"""Build KAGGLE multi-class (per-tooth FDI) dataset @336 from the ORIGINAL data.

- Source images: /workspace/dataset/teeth-seg-humansintheloop/img/*.jpg (1024x2041)
- Source anns:   /workspace/dataset/teeth-seg-humansintheloop/ann/*.jpg.json (supervisely)
- Split:         same as data/KAGGLE/Prompts_Folder/{Train,Val,Test}_text*.xlsx
- Output:        data/KAGGLE_336/{Train,Val,Test}_Folder/{img,label} (multi-class, resized to 336)

NOTE: classTitle is already the class index 1..32 (not FDI), so fill the mask directly.
"""
import os
import json
import shutil
import cv2
import numpy as np
import pandas as pd

SRC_IMG = "/workspace/dataset/teeth-seg-humansintheloop/img"
SRC_ANN = "/workspace/dataset/teeth-seg-humansintheloop/ann"
PROMPT_DIR = "data/KAGGLE/Prompts_Folder"
OUT = "data/KAGGLE_336"
SIZE = 336

SPLITS = {
    "Train": "Train_text.xlsx",
    "Val": "Val_text.xlsx",
    "Test": "Test_text_original.xlsx",
}

for split, xlsx in SPLITS.items():
    df = pd.read_excel(os.path.join(PROMPT_DIR, xlsx))
    out_img = os.path.join(OUT, f"{split}_Folder", "img")
    out_lbl = os.path.join(OUT, f"{split}_Folder", "label")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    for _, r in df.iterrows():
        stem = os.path.splitext(r["Image"])[0]  # strip .png

        img = None
        for ext in (".jpg", ".png"):
            p = os.path.join(SRC_IMG, f"{stem}{ext}")
            if os.path.isfile(p):
                img = cv2.imread(p)
                break
        if img is None:
            print(f"[skip] image not found: {stem}")
            continue

        H, W = img.shape[:2]
        with open(os.path.join(SRC_ANN, f"{stem}.jpg.json")) as f:
            ann = json.load(f)

        mask = np.zeros((H, W), dtype=np.uint8)
        for obj in ann["objects"]:
            cls = int(obj["classTitle"])  # class index 1..32 directly
            pts = np.array(obj["points"]["exterior"], dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts.astype(np.int32)], cls)

        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(out_img, f"{stem}.png"), img)
        cv2.imwrite(os.path.join(out_lbl, f"{stem}.png"), mask)

    print(f"{split}: {len(df)} images -> {OUT}/{split}_Folder/")

os.makedirs(os.path.join(OUT, "Prompts_Folder"), exist_ok=True)
for xlsx in ["Train_text.xlsx", "Val_text.xlsx", "Test_text_original.xlsx"]:
    shutil.copy2(os.path.join(PROMPT_DIR, xlsx), os.path.join(OUT, "Prompts_Folder", xlsx))
shutil.copy2("data/KAGGLE/cls_to_fdi.json", os.path.join(OUT, "cls_to_fdi.json"))
print("copied Prompts_Folder + cls_to_fdi.json")
