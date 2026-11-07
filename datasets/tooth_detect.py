"""DUAL/KAGGLE/INREDD style detection dataset: multi-class mask -> per-tooth instances.

For each image, the multi-class label (pixel value = class index 1..32, 0 = bg)
is split into one instance per present tooth:
    boxes  [xmin, ymin, xmax, ymax]
    labels class index 1..32 (FDI = cls_to_fdi[cls])
    masks  binary uint8 [H, W]
"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

NUM_CLASSES = 32  # tooth classes (labels 1..32; 0 = background)


class ToothDetectionDataset(Dataset):
    def __init__(self, root, size=224):
        self.img_dir = os.path.join(root, "img")
        self.lbl_dir = os.path.join(root, "label")
        self.files = sorted(os.listdir(self.img_dir))
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (img.shape[0], img.shape[1]) != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        mask = cv2.imread(os.path.join(self.lbl_dir, fname), 0)
        if (mask.shape[0], mask.shape[1]) != (self.size, self.size):
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        boxes, labels, masks = [], [], []
        for cls in range(1, NUM_CLASSES + 1):
            binm = (mask == cls)
            if binm.sum() == 0:
                continue
            ys, xs = np.where(binm)
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
            labels.append(cls)
            masks.append(binm.astype(np.uint8))

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)

        target = {"boxes": boxes, "labels": labels, "masks": masks}
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
