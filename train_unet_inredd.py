"""Train a standard UNet (from bilateralsymmetrybasedaugmentation) on INREDD.

Two tasks:
  --task binary      : full-teeth coarse segmentation (1 channel, tooth vs bg)
  --task multiclass  : per-tooth segmentation (33 channels: 32 FDI teeth + bg)

Usage:
  python train_unet_inredd.py --task binary --epochs 100 --lr 2e-4
  python train_unet_inredd.py --task multiclass --epochs 100 --lr 2e-4
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

REPO = "/workspace/code/bilateralsymmetrybasedaugmentation-main"
sys.path.insert(0, REPO)
from models.unet.UNet import UNet  # noqa: E402

NUM_CLASSES = 32


class INREDDDataset(Dataset):
    def __init__(self, img_dir, label_dir, image_size=224, flip=False, binary=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.flip = flip
        self.binary = binary
        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fn), 0)          # grayscale
        img = cv2.resize(img, (self.image_size, self.image_size))
        lab = cv2.imread(os.path.join(self.label_dir, fn), 0)        # 0..32
        lab = cv2.resize(lab, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.flip and np.random.random() > 0.5:
            img = cv2.flip(img, 1)
            lab = cv2.flip(lab, 1)

        if self.binary:
            lab = (lab > 0).astype(np.uint8)                         # tooth vs bg

        img = (img.astype(np.float32) / 255.0)
        img = torch.from_numpy(img).unsqueeze(0)                     # (1, H, W)
        lab = torch.from_numpy(lab).long()                           # (H, W)
        return img, lab


def binary_dice_loss(prob, target, eps=1e-6):
    target = target.float()
    num = 2.0 * (prob * target).sum() + eps
    den = prob.sum() + target.sum() + eps
    return 1.0 - num / den


def multiclass_dice_loss(logits, target, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    tgt = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    probs = probs[:, 1:]
    tgt = tgt[:, 1:]
    num = 2.0 * (probs * tgt).sum(dim=(0, 2, 3)) + eps
    den = probs.sum(dim=(0, 2, 3)) + tgt.sum(dim=(0, 2, 3)) + eps
    # only average over classes present in this batch's GT (skip absent ones)
    valid = tgt.sum(dim=(0, 2, 3)) > 0
    if valid.any():
        return 1.0 - (num[valid] / den[valid]).mean()
    return torch.zeros((), device=logits.device)


@torch.no_grad()
def evaluate(model, loader, device, task):
    model.eval()
    if task == "binary":
        num = den = 0.0
        for img, lab in tqdm(loader, desc="Eval"):
            img, lab = img.to(device), lab.to(device)
            pred = (model(img) > 0.5).squeeze(1)                     # (B, H, W) bool
            gt = lab > 0
            num += 2.0 * (pred & gt).sum().item()
            den += pred.sum().item() + gt.sum().item()
        dice = num / (den + 1e-7)
        model.train()
        return dice
    else:
        per_num = torch.zeros(NUM_CLASSES, device=device)
        per_den = torch.zeros(NUM_CLASSES, device=device)
        for img, lab in tqdm(loader, desc="Eval"):
            img, lab = img.to(device), lab.to(device)
            pred = model(img).argmax(dim=1)
            for c in range(1, NUM_CLASSES + 1):
                pc = (pred == c)
                gc = (lab == c)
                per_num[c - 1] += 2.0 * (pc & gc).sum()
                per_den[c - 1] += pc.sum() + gc.sum()
        valid = per_den > 0
        dice = per_num[valid] / (per_den[valid] + 1e-7)
        model.train()
        return dice.mean().item() if valid.any() else 0.0


@torch.no_grad()
def evaluate_test(model, loader, device, task):
    """Final metrics on the test set. Binary -> whole-tooth Dice;
    multiclass -> per-class mean Dice + IA (fraction of tooth instances with Dice > 0.5)."""
    model.eval()
    if task == "binary":
        num = den = 0.0
        for img, lab in tqdm(loader, desc="Test"):
            img, lab = img.to(device), lab.to(device)
            pred = (model(img) > 0.5).squeeze(1)
            gt = lab > 0
            num += 2.0 * (pred & gt).sum().item()
            den += pred.sum().item() + gt.sum().item()
        return {"mDice": num / (den + 1e-7)}
    else:
        per_num = torch.zeros(NUM_CLASSES, device=device)
        per_den = torch.zeros(NUM_CLASSES, device=device)
        instance_dices = []
        for img, lab in tqdm(loader, desc="Test"):
            img, lab = img.to(device), lab.to(device)
            pred = model(img).argmax(dim=1)                      # (B, H, W)
            for b in range(img.shape[0]):
                for c in range(1, NUM_CLASSES + 1):
                    gc = (lab[b] == c)
                    if gc.sum().item() == 0:                     # absent instance
                        continue
                    pc = (pred[b] == c)
                    n = 2.0 * (pc & gc).sum().item()
                    d = pc.sum().item() + gc.sum().item()
                    per_num[c - 1] += n
                    per_den[c - 1] += d
                    instance_dices.append(n / (d + 1e-7))
        valid = per_den > 0
        mdice = (per_num[valid] / (per_den[valid] + 1e-7)).mean().item() if valid.any() else 0.0
        ia = float(np.mean([d > 0.5 for d in instance_dices])) if instance_dices else 0.0
        return {"mDice": mdice, "IA": ia, "n_instances": len(instance_dices)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--data-root", default="/workspace/code/MedCLIPSeg-main/data/INREDD")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="/workspace/code/MedCLIPSeg-main/unet_inredd")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    binary = (args.task == "binary")
    train_ds = INREDDDataset(
        os.path.join(args.data_root, "Train_Folder", "img"),
        os.path.join(args.data_root, "Train_Folder", "label"), flip=True, binary=binary)
    val_ds = INREDDDataset(
        os.path.join(args.data_root, "Val_Folder", "img"),
        os.path.join(args.data_root, "Val_Folder", "label"), flip=False, binary=binary)
    test_ds = INREDDDataset(
        os.path.join(args.data_root, "Test_Folder", "img"),
        os.path.join(args.data_root, "Test_Folder", "label"), flip=False, binary=binary)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    out_channels = 1 if binary else NUM_CLASSES + 1
    model = UNet(in_channels=1, out_channels=out_channels, init_features=64).to(device)
    print(f"task={args.task}  UNet params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    ce = nn.BCELoss() if binary else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best = 0.0
    logf = open(os.path.join(args.output_dir, f"log_{args.task}.txt"), "w")

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for img, lab in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            img, lab = img.to(device), lab.to(device)
            out = model(img)
            if binary:
                loss = ce(out, lab.float().unsqueeze(1)) + binary_dice_loss(out.squeeze(1), lab)
            else:
                loss = ce(out, lab) + multiclass_dice_loss(out, lab, NUM_CLASSES + 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()

        val = evaluate(model, val_loader, device, args.task)
        line = f"EPOCH {epoch + 1}/{args.epochs} | Train Loss {total / len(train_loader):.4f} | Val Dice {val:.4f}"
        print(line)
        logf.write(line + "\n")
        logf.flush()

        if val > best:
            best = val
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_dice_{args.task}.pth"))
            logf.write(f"  -> new best Dice {best:.4f}\n")
            logf.flush()

    logf.close()

    # Final test evaluation with the best checkpoint
    best_path = os.path.join(args.output_dir, f"best_dice_{args.task}.pth")
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_metrics = evaluate_test(model, test_loader, device, args.task)
        print(f"TEST ({args.task}): {test_metrics}")
        logf = open(os.path.join(args.output_dir, f"log_{args.task}.txt"), "a")
        logf.write(f"TEST {test_metrics}\n")
        logf.close()

    print(f"Done. best Val Dice ({args.task}) = {best:.4f}")


if __name__ == "__main__":
    main()
