"""Evaluate binary (all-teeth) segmentation on the test set: mDice / mIoU."""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataloader import DatasetSegmentation, ValGenerator
from trainers import build_medclipseg_unimedclip
from utils.main_utils import load_cfg_from_cfg_file, read_text


def dice(pred, gt):
    num = 2.0 * np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    return float(num / den) if den > 0 else 0.0


def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--output-dir", default="./inredd_binary_336_lr2e4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)
    device = torch.device(args.device)

    backbone = cfg.MODEL.BACKBONE.replace("/", "-")
    results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone}"
    ckpt = os.path.join(args.output_dir, cfg.DATASET.NAME, "trained_models",
                        f"seed{args.seed}", f"{results_name}_best_dice.pth")
    assert os.path.isfile(ckpt), f"checkpoint not found: {ckpt}"

    model = build_medclipseg_unimedclip(cfg)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.eval().to(device)

    test_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    test_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + "Test_text_original.xlsx")
    test_ds = DatasetSegmentation(cfg.DATASET.TEST_PATH, cfg.DATASET.NAME, test_text, test_tf,
                                  image_size=cfg.DATASET.SIZE)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    dice_list, iou_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            seg = model(image=batch["image"].to(device), text=batch["text_prompt"],
                        num_samples=cfg.TEST.NUM_SAMPLES)          # [N, B, H, W]
            prob = torch.sigmoid(seg).mean(dim=0)                   # [B, H, W]
            preds = (prob > 0.5).cpu().numpy()
            gts = batch["ground_truth_mask"].cpu().numpy()
            for i in range(len(batch["image_name"])):
                p = preds[i] > 0
                g = gts[i] > 0
                dice_list.append(dice(p, g))
                iou_list.append(iou(p, g))

    metrics = {
        "mDice": float(np.mean(dice_list)),
        "mIoU": float(np.mean(iou_list)),
        "n": len(dice_list),
    }
    print(json.dumps(metrics, indent=2))

    out = os.path.join(args.output_dir, cfg.DATASET.NAME, "seg_results", f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics_binary.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
