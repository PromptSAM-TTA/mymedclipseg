"""Re-test the three 100-epoch best checkpoints (INREDD / KAGGLE / DUAL).

- Metrics unchanged: mDice, mIoU, IA (strict), mNSD (tau=2px), HD95.
- Visualization unchanged: 10 worst samples, single overlay figure each
  (grayscale X-ray + GT red + prediction green).
- NEW: every test sample with Dice < 0.2 is additionally saved to a `low_dice`
  folder, named `{dataset}-{image_stem}-{dice}-{fdi}.png` for easy inspection.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, distance_transform_edt

from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import (
    build_medclipseg_unimedclip,
    build_medclipseg_biomedclip,
    build_medclipseg_clip,
    build_medclipseg_pubmedclip,
)
from utils.main_utils import load_cfg_from_cfg_file, read_text

CLIP_MEAN = np.array([0.485, 0.456, 0.406])
CLIP_STD = np.array([0.229, 0.224, 0.225])

# (display name, config, output dir)
RUNS = [
    ("INREDD", "configs/INREDD.yaml", "./inredd100_lr2e4"),
    ("KAGGLE", "configs/KAGGLE.yaml", "./kaggle100_lr2e4"),
    ("DUAL", "configs/DUAL.yaml", "./dual100_lr2e4"),
]

LOW_DICE_THRESHOLD = 0.2


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


def _surface(mask):
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
    bP, bG = _surface(pred), _surface(gt)
    if bP.sum() == 0 and bG.sum() == 0:
        return 1.0
    d_to_G = distance_transform_edt(~bG)
    d_to_P = distance_transform_edt(~bP)
    matchP = np.logical_and(bP, d_to_G <= tau).sum()
    matchG = np.logical_and(bG, d_to_P <= tau).sum()
    den = bP.sum() + bG.sum()
    return float((matchP + matchG) / den) if den > 0 else 0.0


def hd95(pred, gt):
    bP, bG = _surface(pred), _surface(gt)
    if bP.sum() == 0 or bG.sum() == 0:
        return float("nan")
    d_to_G = distance_transform_edt(~bG)
    d_to_P = distance_transform_edt(~bP)
    dists = np.concatenate([d_to_G[bP], d_to_P[bG]])
    return float(np.percentile(dists, 95))


def denormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * CLIP_STD + CLIP_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def save_overlay(s, path, title):
    img = denormalize(s["image"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_where(~s["gt"], s["gt"]), cmap="Reds", alpha=0.55, vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(~s["pred"], s["pred"]), cmap="Greens", alpha=0.55, vmin=0, vmax=1)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def evaluate_one(dataset_name, config_file, output_dir, seed=1):
    cfg = load_cfg_from_cfg_file(config_file)
    device = cfg.MODEL.DEVICE

    backbone = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone}"
    ckpt = os.path.join(output_dir, cfg.DATASET.NAME, "trained_models",
                        f"seed{seed}", f"{results_name}_best_dice.pth")
    assert os.path.isfile(ckpt), f"checkpoint not found: {ckpt}"

    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.eval().to(device)

    test_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    test_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + "Test_text_original.xlsx")
    test_ds = DatasetSegmentation(
        cfg.DATASET.TEST_PATH, cfg.DATASET.NAME, test_text, test_tf,
        image_size=cfg.DATASET.SIZE,
        multi_class_to_binary=getattr(cfg.DATASET, "MULTI_CLASS_TO_BINARY", False),
        cls_to_fdi=getattr(cfg.DATASET, "CLS_TO_FDI_JSON", None))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    samples = []
    dice_list, iou_list, nsd_list, hd95_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"{dataset_name} test"):
            seg_samples = model(
                image=batch["image"].to(device),
                text=batch["text_prompt"],
                num_samples=cfg.TEST.NUM_SAMPLES)                       # [N, B, H, W]
            mean_prob = torch.sigmoid(seg_samples).mean(dim=0)           # [B, H, W]
            preds = (mean_prob > 0.5).cpu().numpy()
            gts = batch["ground_truth_mask"].cpu().numpy()

            for i in range(len(batch["image_name"])):
                gt = (gts[i] > 0)
                pred = preds[i].astype(bool)
                d = dice(pred, gt)
                s = {
                    "fdi": str(batch["fdi"][i].item()),
                    "img_name": batch["image_name"][i],
                    "image": batch["image"][i],
                    "pred": pred,
                    "gt": gt,
                    "dice": d,
                    "iou": iou(pred, gt),
                    "nsd": nsd(pred, gt, tau=2),
                    "hd95": hd95(pred, gt),
                }
                samples.append(s)
                dice_list.append(d)
                iou_list.append(s["iou"])
                nsd_list.append(s["nsd"])
                if not np.isnan(s["hd95"]):
                    hd95_list.append(s["hd95"])

    # IA (strict): correct FDI localization within each image
    by_image = {}
    for s in samples:
        by_image.setdefault(s["img_name"], []).append(s)
    ia_correct = ia_total = 0
    for items in by_image.values():
        for j in range(len(items)):
            self_iou = items[j]["iou"]
            other = [items[k]["iou"] for k in range(len(items)) if k != j]
            if self_iou > 0.5 and self_iou > (max(other) if other else 0.0):
                ia_correct += 1
            ia_total += 1

    metrics = {
        "mDice": float(np.mean(dice_list)),
        "mIoU": float(np.mean(iou_list)),
        "IA": ia_correct / ia_total if ia_total else 0.0,
        "IA_correct": ia_correct, "IA_total": ia_total,
        "mNSD": float(np.mean(nsd_list)),
        "HD95_mean": float(np.mean(hd95_list)),
        "HD95_median": float(np.median(hd95_list)),
    }

    out_root = os.path.join(output_dir, cfg.DATASET.NAME, "seg_results", f"seed{seed}")
    os.makedirs(out_root, exist_ok=True)

    # 1) save metrics json
    with open(os.path.join(out_root, f"metrics_{results_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 2) 10 worst single-figure visualizations (unchanged)
    samples_sorted = sorted(samples, key=lambda s: s["dice"])
    vis_dir = os.path.join(out_root, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    for idx, s in enumerate(samples_sorted[:10]):
        title = f"FDI {s['fdi']}  |  Dice {s['dice']:.3f}"
        save_overlay(s, os.path.join(vis_dir, f"worst_{idx + 1}_fdi{s['fdi']}_dice{s['dice']:.3f}.png"), title)

    # 3) low-dice samples (< 0.2) -> separate folder
    low_dir = os.path.join(out_root, "low_dice")
    os.makedirs(low_dir, exist_ok=True)
    n_low = 0
    for s in samples:
        if s["dice"] < LOW_DICE_THRESHOLD:
            stem = os.path.splitext(s["img_name"])[0]
            fname = f"{dataset_name}-{stem}-{s['dice']:.3f}-{s['fdi']}.png"
            save_overlay(s, os.path.join(low_dir, fname), f"FDI {s['fdi']}  |  Dice {s['dice']:.3f}")
            n_low += 1

    metrics["n_low_dice"] = n_low
    return metrics


def main():
    print(f"\n{'=' * 60}")
    for name, cfg, outdir in RUNS:
        m = evaluate_one(name, cfg, outdir)
        print(f"\n[{name}] mDice={m['mDice']:.4f}  mIoU={m['mIoU']:.4f}  "
              f"IA={m['IA']:.4f} ({m['IA_correct']}/{m['IA_total']})  "
              f"mNSD={m['mNSD']:.4f}  HD95={m['HD95_mean']:.2f}px  "
              f"low_dice(<0.2)={m['n_low_dice']}")
        print(f"  -> {outdir}/{cfg.replace('configs/', '').replace('.yaml', '')}/seg_results/seed1/")
    print(f"{'=' * 60}\nDone.")


if __name__ == "__main__":
    main()
