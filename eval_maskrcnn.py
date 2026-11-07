"""Evaluate a trained Mask R-CNN on the DUAL test set.

Metrics aligned with MedCLIPSeg (eval_inredd.py), per GT tooth:
  - mDice / mIoU / mNSD : computed against the same-FDI prediction (0 if missed)
  - HD95 : boundary Hausdorff distance (95th percentile), NaN-excluded
  - IA   : strict instance accuracy (IoU>0.5 AND IoU greater than any other tooth)
  - recall@0.5 : fraction of GT teeth matched by any pred (bonus, class-agnostic)

Also saves a few GT-vs-prediction overlays and a metrics JSON.
"""
import os
import json
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import binary_erosion, distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.tooth_detect import ToothDetectionDataset, collate_fn, NUM_CLASSES
from train_maskrcnn import get_model


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def mask_dice(a, b):
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom > 0 else 0.0


def _surface(mask):
    m = mask.astype(bool)
    eroded = binary_erosion(m, structure=np.ones((3, 3), dtype=bool))
    return m & ~eroded


def mask_nsd(pred, gt, tau=2):
    bP, bG = _surface(pred), _surface(gt)
    if bP.sum() == 0 and bG.sum() == 0:
        return 1.0
    d_to_G = distance_transform_edt(~bG)
    d_to_P = distance_transform_edt(~bP)
    matchP = np.logical_and(bP, d_to_G <= tau).sum()
    matchG = np.logical_and(bG, d_to_P <= tau).sum()
    den = bP.sum() + bG.sum()
    return float((matchP + matchG) / den) if den > 0 else 0.0


def mask_hd95(pred, gt):
    bP, bG = _surface(pred), _surface(gt)
    if bP.sum() == 0 or bG.sum() == 0:
        return float("nan")
    d_to_G = distance_transform_edt(~bG)
    d_to_P = distance_transform_edt(~bP)
    dists = np.concatenate([d_to_G[bP], d_to_P[bG]])
    return float(np.percentile(dists, 95))


def predict(model, loader, device, score_thresh=0.5):
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = [im.to(device) for im in images]
            out = model(images)
            for o in out:
                keep = o["scores"] > score_thresh
                masks = o["masks"][keep].squeeze(1).cpu().numpy()  # [N,H,W]
                masks = (masks > 0.5).astype(np.uint8)
                preds.append({
                    "labels": o["labels"][keep].cpu().numpy(),
                    "masks": masks,
                })
    return preds


def load_gt(ds):
    gts = []
    for i in range(len(ds)):
        _, t = ds[i]
        gts.append({"labels": t["labels"].numpy(), "masks": t["masks"].numpy()})
    return gts


def compute_metrics(preds, gts, tau=2):
    """Metrics aligned with MedCLIPSeg, per GT tooth.

    For every GT tooth f:
      - find the same-FDI prediction with highest IoU -> "pred_f"
      - if none (missed): mDice/mIoU/mNSD = 0, HD95 = NaN (excluded), IA = False
      - mDice/mIoU/mNSD/HD95 computed directly on (pred_f, gt_f)
      - IA (strict): IoU(pred_f, gt_f) > 0.5 AND > IoU(pred_f, any other GT tooth)
    """
    dice_list, iou_list, nsd_list, hd95_list = [], [], [], []
    ia_correct = ia_total = 0
    recall_correct = recall_total = 0
    for pred, gt in zip(preds, gts):
        pm, pl = pred["masks"], pred["labels"]
        gm, gl = gt["masks"], gt["labels"]
        ng = len(gm)

        for j in range(ng):
            ia_total += 1
            recall_total += 1

            # best same-class prediction for this GT tooth
            best_iou = 0.0
            best_idx = -1
            for i in range(len(pm)):
                if pl[i] != gl[j]:
                    continue
                iou = mask_iou(pm[i], gm[j])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            # recall (any class, reference)
            any_iou = 0.0
            for i in range(len(pm)):
                any_iou = max(any_iou, mask_iou(pm[i], gm[j]))
            if any_iou > 0.5:
                recall_correct += 1

            if best_idx < 0:
                dice_list.append(0.0)
                iou_list.append(0.0)
                nsd_list.append(0.0)
                # HD95 stays NaN -> excluded
                continue

            pf = pm[best_idx]
            dice_list.append(mask_dice(pf, gm[j]))
            iou_list.append(best_iou)
            nsd_list.append(mask_nsd(pf, gm[j], tau))
            h = mask_hd95(pf, gm[j])
            if not np.isnan(h):
                hd95_list.append(h)

            # IA (strict): IoU>0.5 AND self > any other tooth
            self_iou = best_iou
            other = [mask_iou(pf, gm[k]) for k in range(ng) if k != j]
            max_other = max(other) if other else 0.0
            if self_iou > 0.5 and self_iou > max_other:
                ia_correct += 1

    mDice = float(np.mean(dice_list)) if dice_list else 0.0
    mIoU = float(np.mean(iou_list)) if iou_list else 0.0
    mNSD = float(np.mean(nsd_list)) if nsd_list else 0.0
    HD95 = float(np.mean(hd95_list)) if hd95_list else float("nan")
    ia = ia_correct / ia_total if ia_total else 0.0
    recall = recall_correct / recall_total if recall_total else 0.0
    return {
        "mDice": mDice, "mIoU": mIoU, "mNSD": mNSD, "HD95": HD95,
        "IA": ia, "IA_correct": ia_correct, "IA_total": ia_total,
        "recall@0.5": recall, "recall_correct": recall_correct, "recall_total": recall_total,
    }


def save_overlays(ds, preds, outdir, n=6, score_thresh=0.5):
    os.makedirs(outdir, exist_ok=True)
    for k in range(min(n, len(ds))):
        img = cv2.imread(os.path.join(ds.img_dir, ds.files[k]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = ds[k][1]["masks"].numpy()          # [Ng,H,W]
        pred = preds[k]["masks"]                # [Np,H,W]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap="gray"); axes[0].set_title("X-ray"); axes[0].axis("off")
        gtm = gt.sum(0) > 0
        axes[1].imshow(img, cmap="gray")
        axes[1].imshow(np.ma.masked_where(~gtm, gtm), cmap="Reds", alpha=0.6)
        axes[1].set_title(f"GT ({len(gt)} teeth)"); axes[1].axis("off")
        pm = pred.sum(0) > 0
        axes[2].imshow(img, cmap="gray")
        axes[2].imshow(np.ma.masked_where(~pm, pm), cmap="Greens", alpha=0.6)
        axes[2].set_title(f"Pred ({len(pred)} teeth)"); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"pred_{k}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--ckpt", type=str, default="./maskrcnn_dual/maskrcnn_dual_best.pth")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--outdir", type=str, default="./maskrcnn_dual/eval")
    args = ap.parse_args()

    device = torch.device(args.device)
    num_classes = NUM_CLASSES + 1

    test_ds = ToothDetectionDataset("data/DUAL/Test_Folder", size=args.size)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                             collate_fn=collate_fn, num_workers=4)
    print(f"test={len(test_ds)}", flush=True)

    model = get_model(num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.to(device).eval()

    preds = predict(model, test_loader, device, args.score_thresh)
    gts = load_gt(test_ds)
    metrics = compute_metrics(preds, gts)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_overlays(test_ds, preds, os.path.join(args.outdir, "visualizations"))

    print(json.dumps(metrics, indent=2), flush=True)
    print(f"saved -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
