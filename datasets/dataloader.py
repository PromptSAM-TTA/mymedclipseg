import torch
import os
import json
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

import random
import cv2

# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2

# Normalization used by CLIP models
CLIP_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

# FDI (11..48) -> anatomical tooth name, used to build descriptive text prompts.
FDI_TO_NAME = {
    11: "maxillary right central incisor",
    12: "maxillary right lateral incisor",
    13: "maxillary right canine",
    14: "maxillary right first premolar",
    15: "maxillary right second premolar",
    16: "maxillary right first molar",
    17: "maxillary right second molar",
    18: "maxillary right third molar",
    21: "maxillary left central incisor",
    22: "maxillary left lateral incisor",
    23: "maxillary left canine",
    24: "maxillary left first premolar",
    25: "maxillary left second premolar",
    26: "maxillary left first molar",
    27: "maxillary left second molar",
    28: "maxillary left third molar",
    31: "mandibular left central incisor",
    32: "mandibular left lateral incisor",
    33: "mandibular left canine",
    34: "mandibular left first premolar",
    35: "mandibular left second premolar",
    36: "mandibular left first molar",
    37: "mandibular left second molar",
    38: "mandibular left third molar",
    41: "mandibular right central incisor",
    42: "mandibular right lateral incisor",
    43: "mandibular right canine",
    44: "mandibular right first premolar",
    45: "mandibular right second premolar",
    46: "mandibular right first molar",
    47: "mandibular right second molar",
    48: "mandibular right third molar",
}

# FDI tens digit (1..4) -> region in the X-ray image. Left/right is mirrored vs
# anatomy (panoramic radiograph convention): maxillary right -> image left, etc.
FDI_TO_REGION = {
    1: "upper-left region",    # FDI 11-18 (maxillary right  -> image left)
    2: "upper-right region",   # FDI 21-28 (maxillary left   -> image right)
    3: "lower-right region",   # FDI 31-38 (mandibular left  -> image right)
    4: "lower-left region",    # FDI 41-48 (mandibular right -> image left)
}
                                      
def to_long_tensor(pic):
    if isinstance(pic, Image.Image):
        pic = np.array(pic, dtype=np.uint8)
    return torch.from_numpy(pic).long()

def random_rotate(image, label):
    angle = random.randint(-20, 20)
    image = image.rotate(angle)
    label = label.rotate(angle)
    return image, label

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  # (width, height)

    def __call__(self, sample):
        image, mask = sample['image'], sample['ground_truth_mask']
        text = sample['text_prompt']

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))

        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = np.squeeze(mask, axis=2)
            mask = Image.fromarray(mask.astype(np.uint8))

        # Resize
        if image.size != self.output_size:
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        if mask.size != self.output_size:
            mask = mask.resize(self.output_size, resample=Image.NEAREST)

        # To tensor and normalize
        image = F.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)

        # Update sample
        sample['image'] = image
        sample['ground_truth_mask'] = mask
        sample['text_prompt'] = text  # unchanged

        return sample


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size  # (width, height)

    def __call__(self, sample):
        image, mask = sample['image'], sample['ground_truth_mask']
        text = sample['text_prompt']

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            image = Image.fromarray(image.astype(np.uint8))

        if isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = np.squeeze(mask, axis=2)
            mask = Image.fromarray(mask.astype(np.uint8))

        if random.random() > 0.5:
            image, mask = random_rotate(image, mask)

        # Resize
        if image.size != self.output_size:
            image = image.resize(self.output_size, resample=Image.BICUBIC)
        if mask.size != self.output_size:
            mask = mask.resize(self.output_size, resample=Image.NEAREST)

        # To tensor and normalize
        image = F.to_tensor(image)
        image = CLIP_NORMALIZE(image)
        mask = to_long_tensor(mask)

        # Update sample
        sample['image'] = image
        sample['ground_truth_mask'] = mask
        sample['text_prompt'] = text  # unchanged

        return sample

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class DatasetSegmentation(Dataset):

    def __init__(
        self,
        dataset_path: str,
        task_name: str,
        row_text: list,
        joint_transform: Callable = None,
        one_hot_mask: int = False,
        image_size: int = 224,
        multi_class_to_binary: bool = False,
        cls_to_fdi=None,
    ) -> None:

        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'label')
        self.one_hot_mask = one_hot_mask
        self.task_name = task_name
        self.multi_class_to_binary = multi_class_to_binary

        self.data_pairs = [
            (row['Image'], row['Ground Truth'], row['Description'])
            for row in row_text
        ]

        self.data_pairs = sorted(self.data_pairs, key=lambda x: x[0])

        # class-index (1..32) -> FDI tooth number (11..48), for "tooth <FDI>" prompts.
        if cls_to_fdi is None:
            self.cls_to_fdi = {}
        elif isinstance(cls_to_fdi, str):
            with open(cls_to_fdi) as f:
                raw = json.load(f)
            self.cls_to_fdi = {int(k): v for k, v in raw.items()}
        else:
            self.cls_to_fdi = {int(k): v for k, v in dict(cls_to_fdi).items()}

        if self.multi_class_to_binary:
            self._build_binary_samples()
        else:
            # one sample per image row, prompt taken from the xlsx Description.
            self.samples = [(img, msk, desc) for (img, msk, desc) in self.data_pairs]

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x: x

    def _build_binary_samples(self):
        """Expand each image into one sample per present tooth.

        The multi-class mask (pixel value = class index, 0 = background) is
        split into per-tooth binary masks at load time. Each sample keeps the
        same image but gets a 0/1 mask and a "tooth <FDI>" prompt, so the
        model stays 1-channel and the binary loss is unchanged.
        """
        self.samples = []
        for image_filename, mask_filename, _ in self.data_pairs:
            mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
            if mask is None:
                continue
            classes = np.unique(mask)
            classes = classes[classes != 0]
            for cls in classes:
                self.samples.append((image_filename, mask_filename, int(cls)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        if self.multi_class_to_binary:
            image_filename, mask_filename, class_id = self.samples[idx]
        else:
            image_filename, mask_filename, text = self.samples[idx]

        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.multi_class_to_binary:
            # Binarize: keep only this tooth, background everywhere else.
            mask = (mask == class_id).astype(np.uint8)
            fdi = self.cls_to_fdi.get(class_id, class_id)
            if fdi in FDI_TO_NAME:
                region = FDI_TO_REGION[fdi // 10]
                text = f"tooth {fdi}, located in the image's {region}"
            else:
                text = f"tooth {fdi}"
            stem = os.path.splitext(mask_filename)[0]
            mask_filename = f"{stem}_tooth{fdi}.png"

        image, mask = correct_dims(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        inputs = {
            "image": image,
            "ground_truth_mask": mask,
            "image_name": image_filename,
            "mask_name": mask_filename,
            "text_prompt": text,
            "dataset_name": self.task_name
        }

        if self.joint_transform:
            inputs = self.joint_transform(inputs)

        return inputs