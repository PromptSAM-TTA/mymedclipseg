"""Visualize the WORST test samples as single overlay figures.

For each of the lowest-Dice test samples, save ONE figure: the X-ray with the
ground-truth mask (red) and the prediction (green) overlaid. No uncertainty
map, no best samples.

Run after training, e.g.:
  python visualize_test.py --config-file configs/INREDD.yaml --output-dir ./ --seed 1
"""
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def dice(pred, gt):
    num = 2.0 * np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    return float(num / den) if den > 0 else 0.0


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
    parser.add_argument("--source_dataset", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--num-hard", type=int, default=10, help="number of worst samples to save")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update({k: v for k, v in vars(args).items()})
    return cfg


def denormalize(img_tensor):
    """CLIP-normalized (3,H,W) tensor -> (H,W,3) uint8 for display."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * CLIP_STD + CLIP_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def main():
    cfg = get_arguments()
    device = cfg.MODEL.DEVICE

    backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}"
    checkpoint_type = "latest" if cfg.TEST.USE_LATEST else "best_dice"
    src_dataset = cfg.source_dataset or cfg.DATASET.NAME
    checkpoint_path = os.path.join(
        cfg.output_dir, src_dataset, "trained_models",
        f"seed{cfg.seed}", f"{results_name}_{checkpoint_type}.pth",
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

    samples = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval"):
            seg_samples = model(
                image=batch["image"].to(device),
                text=batch["text_prompt"],
                num_samples=cfg.TEST.NUM_SAMPLES,
            )                                   # [N, B, H, W]
            mean_prob = torch.sigmoid(seg_samples).mean(dim=0)   # [B, H, W]
            preds = (mean_prob > 0.5).cpu().numpy()
            gts = batch["ground_truth_mask"].cpu().numpy()

            for i in range(len(batch["image_name"])):
                gt = (gts[i] > 0)
                samples.append({
                    "fdi": str(batch["fdi"][i].item()),
                    "image": batch["image"][i],
                    "gt": gt,
                    "pred": preds[i].astype(bool),
                    "dice": dice(preds[i].astype(bool), gt),
                })

    # rank by Dice: worst = lowest
    samples.sort(key=lambda s: s["dice"])
    hard = samples[:cfg.num_hard]

    out_root = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results",
                            f"seed{cfg.seed}", "visualizations")
    os.makedirs(out_root, exist_ok=True)

    for idx, s in enumerate(hard):
        img = denormalize(s["image"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(img, cmap="gray")
        ax.imshow(np.ma.masked_where(~s["gt"], s["gt"]), cmap="Reds", alpha=0.55, vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(~s["pred"], s["pred"]), cmap="Greens", alpha=0.55, vmin=0, vmax=1)
        ax.set_title(f"FDI {s['fdi']}  |  Dice {s['dice']:.3f}", fontsize=13)
        ax.axis("off")
        out_path = os.path.join(out_root, f"worst_{idx + 1}_fdi{s['fdi']}_dice{s['dice']:.3f}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"saved {out_path}")

    print(f"\nDone. saved {len(hard)} worst samples -> {out_root}")


if __name__ == "__main__":
    main()
