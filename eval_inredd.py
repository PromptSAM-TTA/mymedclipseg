"""Evaluate INREDD FDI tooth segmentation with 5 metrics.

Metrics (all in pixel units, per (image, tooth) pair then averaged):
  - Dice       : 2*|P&G| / (|P| + |G|)
  - mIoU       : |P&G| / |P|G|
  - IA         : strict instance accuracy. For each GT tooth prompted with its
                 own FDI, correct iff IoU(pred, this_tooth) > 0.5 AND it is
                 larger than IoU(pred, any_other_tooth_in_same_image).
  - NSD        : normalized surface Dice with tolerance tau (default 2 px)
  - HD95       : 95th percentile of boundary Hausdorff distance (px)

Run after training, e.g.:
  python eval_inredd.py --config-file configs/INREDD.yaml --output-dir ./
"""
import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.ndimage import binary_erosion, distance_transform_edt

from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import (
    build_medclipseg_unimedclip,
    build_medclipseg_biomedclip,
    build_medclipseg_clip,
    build_medclipseg_pubmedclip,
)
from utils.main_utils import load_cfg_from_cfg_file, read_text


# --------------------------------------------------------------------------- #
# Metric helpers (numpy bool masks, same HxW)
# --------------------------------------------------------------------------- #
def _surface(mask):
    """One-pixel-thick boundary: foreground with at least one 8-connected bg neighbour."""
    m = mask.astype(bool)
    eroded = binary_erosion(m, structure=np.ones((3, 3), dtype=bool))
    return m & ~eroded


def dice(pred, gt):
    num = 2.0 * np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    return float(num / den) if den > 0 else 0.0


def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def nsd(pred, gt, tau=2):
    """Normalized surface Dice: boundary points within `tau` px of the other surface count as matched."""
    bP, bG = _surface(pred), _surface(gt)
    if bP.sum() == 0 and bG.sum() == 0:
        return 1.0
    d_to_G = distance_transform_edt(~bG)   # distance to nearest GT boundary px
    d_to_P = distance_transform_edt(~bP)
    matchP = np.logical_and(bP, d_to_G <= tau).sum()
    matchG = np.logical_and(bG, d_to_P <= tau).sum()
    den = bP.sum() + bG.sum()
    return float((matchP + matchG) / den) if den > 0 else 0.0


def hd95(pred, gt):
    """95th percentile of the pooled two-way surface Hausdorff distance (px)."""
    bP, bG = _surface(pred), _surface(gt)
    if bP.sum() == 0 or bG.sum() == 0:
        return float("nan")
    d_to_G = distance_transform_edt(~bG)
    d_to_P = distance_transform_edt(~bP)
    dists = np.concatenate([d_to_G[bP], d_to_P[bG]])
    return float(np.percentile(dists, 95))


def build_model(cfg):
    if cfg.MODEL.CLIP_MODEL == "unimedclip":
        return build_medclipseg_unimedclip(cfg)
    if cfg.MODEL.CLIP_MODEL == "biomedclip":
        return build_medclipseg_biomedclip(cfg)
    if cfg.MODEL.CLIP_MODEL == "clip":
        return build_medclipseg_clip(cfg)
    if cfg.MODEL.CLIP_MODEL == "pubmedclip":
        return build_medclipseg_pubmedclip(cfg)
    raise NotImplementedError(cfg.MODEL.CLIP_MODEL)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, type=str)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data_percentage", type=int, default=100)
    parser.add_argument("--source_dataset", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--tau", type=int, default=2, help="NSD surface tolerance (px)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def main():
    cfg = get_arguments()
    device = cfg.MODEL.DEVICE

    cfg.DATASET.NAME = cfg.DATASET.NAME + (f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else "")
    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"

    checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best_dice"
    src_dataset = cfg.source_dataset or cfg.DATASET.NAME
    checkpoint_path = os.path.join(
        cfg.output_dir,
        src_dataset,
        "trained_models",
        f"seed{cfg.seed}",
        f"{results_name}_{checkpoint_type}.pth",
    )
    assert os.path.isfile(checkpoint_path), f"checkpoint not found: {checkpoint_path}"

    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
    model.eval().to(device)

    test_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    test_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + "Test_text_original.xlsx")
    test_dataset = DatasetSegmentation(
        cfg.DATASET.TEST_PATH, cfg.DATASET.NAME, test_text, test_tf,
        image_size=cfg.DATASET.SIZE,
        multi_class_to_binary=getattr(cfg.DATASET, "MULTI_CLASS_TO_BINARY", False),
        cls_to_fdi=getattr(cfg.DATASET, "CLS_TO_FDI_JSON", None),
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # image_name -> list of {fdi, pred, gt}
    by_image = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval"):
            seg_samples = model(
                image=batch["image"].to(device),
                text=batch["text_prompt"],
                num_samples=cfg.TEST.NUM_SAMPLES,
            )                                   # [N, B, H, W]
            seg_logits = torch.sigmoid(seg_samples).mean(dim=0)   # [B, H, W]
            preds = (seg_logits > 0.5).cpu().numpy()
            gts = batch["ground_truth_mask"].cpu().numpy()

            for i in range(len(batch["image_name"])):
                img = batch["image_name"][i]
                fdi = batch["text_prompt"][i].split(",")[0].split()[-1].strip()
                by_image.setdefault(img, []).append({
                    "fdi": fdi, "pred": preds[i].astype(bool), "gt": (gts[i] > 0),
                })

    # ---- aggregate -------------------------------------------------------- #
    dice_list, iou_list, nsd_list, hd95_list = [], [], [], []
    per_fdi = {}
    ia_correct = ia_total = 0

    for img, items in by_image.items():
        for it in items:
            d = dice(it["pred"], it["gt"])
            dice_list.append(d)
            iou_list.append(iou(it["pred"], it["gt"]))
            nsd_list.append(nsd(it["pred"], it["gt"], tau=cfg.tau))
            h = hd95(it["pred"], it["gt"])
            if not np.isnan(h):
                hd95_list.append(h)
            per_fdi.setdefault(it["fdi"], []).append(d)

        # IA (strict): correct FDI localization within this image
        n = len(items)
        for j in range(n):
            iou_self = iou(items[j]["pred"], items[j]["gt"])
            other = [iou(items[j]["pred"], items[k]["gt"]) for k in range(n) if k != j]
            max_other = max(other) if other else 0.0
            if iou_self > 0.5 and iou_self > max_other:
                ia_correct += 1
            ia_total += 1

    def _s(a):
        return float(np.mean(a))

    print("\n================ INREDD evaluation ================")
    print(f"test images        : {len(by_image)}")
    print(f"tooth instances    : {ia_total}")
    print(f"mDice              : {_s(dice_list):.4f} +- {float(np.std(dice_list)):.4f}")
    print(f"mIoU               : {_s(iou_list):.4f} +- {float(np.std(iou_list)):.4f}")
    print(f"IA (strict)        : {ia_correct / ia_total:.4f}  ({ia_correct}/{ia_total})")
    print(f"mNSD (tau={cfg.tau}px)   : {_s(nsd_list):.4f} +- {float(np.std(nsd_list)):.4f}")
    print(f"HD95 (px, mean)    : {_s(hd95_list):.4f} +- {float(np.std(hd95_list)):.4f}")
    if hd95_list:
        print(f"HD95 (px, median)  : {float(np.median(hd95_list)):.4f}")
    print("---------------------------------------------------")

    # per-FDI Dice breakdown
    print("\nper-FDI mDice (sorted):")
    for fdi in sorted(per_fdi, key=lambda k: int(k)):
        vals = per_fdi[fdi]
        print(f"  FDI {fdi:>2}  n={len(vals):>2}  Dice={float(np.mean(vals)):.4f}")

    # save per-instance results for reproducibility
    out = {
        "mDice": _s(dice_list), "mIoU": _s(iou_list),
        "IA": ia_correct / ia_total, "IA_correct": ia_correct, "IA_total": ia_total,
        "mNSD": _s(nsd_list), "HD95_mean": _s(hd95_list),
        "HD95_median": float(np.median(hd95_list)) if hd95_list else None,
        "per_fdi": {k: float(np.mean(v)) for k, v in per_fdi.items()},
    }
    out_path = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results",
                            f"seed{cfg.seed}", f"metrics_{results_name}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved metrics -> {out_path}")


if __name__ == "__main__":
    main()
