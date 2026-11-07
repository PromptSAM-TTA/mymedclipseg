"""Fine-tune Mask R-CNN (ResNet-50-FPN, COCO pretrained) on DUAL tooth detection.

Standard process: train -> validate (val loss) -> save best model (lowest val loss)
and periodic checkpoints. Test-set metrics live in eval_maskrcnn.py.
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from datasets.tooth_detect import ToothDetectionDataset, collate_fn, NUM_CLASSES


def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def validate(model, val_loader, device):
    """Validation loss. torchvision detection models return losses only in
    train mode, so keep model.train() but no gradients."""
    model.train()
    total, nb = 0.0, 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total += sum(v for v in loss_dict.values()).item()
            nb += 1
    return total / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--outdir", type=str, default="./maskrcnn_dual")
    args = ap.parse_args()

    device = torch.device(args.device)
    num_classes = NUM_CLASSES + 1  # 32 teeth + background

    train_ds = ToothDetectionDataset("data/DUAL/Train_Folder", size=args.size)
    val_ds = ToothDetectionDataset("data/DUAL/Val_Folder", size=args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)
    print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)

    model = get_model(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    os.makedirs(args.outdir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total, nb = 0.0, 0
        for images, targets in train_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(v for v in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total += losses.item()
            nb += 1
        scheduler.step()

        val_loss = validate(model, val_loader, device)
        msg = (f"EPOCH {epoch + 1}/{args.epochs} | train_loss {total / max(nb, 1):.4f} "
               f"| val_loss {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
        print(msg, flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.outdir, "maskrcnn_dual_best.pth"))
            print(f"  new best (val_loss {best_val:.4f}) -> saved best", flush=True)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.outdir, f"maskrcnn_dual_ep{epoch + 1}.pth"))
            print(f"  saved ep{epoch + 1}", flush=True)

    torch.save(model.state_dict(), os.path.join(args.outdir, "maskrcnn_dual_last.pth"))
    print("training done", flush=True)


if __name__ == "__main__":
    main()
