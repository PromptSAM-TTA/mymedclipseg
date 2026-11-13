"""Build INREDD multi-class (per-tooth FDI) dataset @384 from the ORIGINAL data.

- Source images: /workspace/dataset/inredd-dataset-pan924-yolo-format-main/.../images/*.jpg (1536x2903)
- Source anns:   .../annotations/teeth_fdi_labels/*.json (COCO, per-tooth polygons)
- Split/cls_to_fdi: taken from data/INREDD_336 (the original data/INREDD was deleted)
- Output:        data/INREDD_384/{Train,Val,Test}_Folder/{img,label} (multi-class, resized to 384)
"""
import os
import json
import shutil
import cv2
import numpy as np
import pandas as pd

SRC_IMG = "/workspace/dataset/inredd-dataset-pan924-yolo-format-main/inredd-dataset-pan924-yolo-format-main/images"
SRC_ANN = "/workspace/dataset/inredd-dataset-pan924-yolo-format-main/inredd-dataset-pan924-yolo-format-main/annotations/teeth_fdi_labels"
PROMPT_DIR = "data/INREDD_336/Prompts_Folder"
OUT = "data/INREDD_384"
SIZE = 384

cls_to_fdi = {int(k): int(v) for k, v in json.load(open("data/INREDD_336/cls_to_fdi.json")).items()}
fdi_to_cls = {v: k for k, v in cls_to_fdi.items()}

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
        stem = os.path.splitext(r["Image"])[0]
        img = cv2.imread(os.path.join(SRC_IMG, f"{stem}.jpg"))
        if img is None:
            print(f"[skip] image not found: {stem}.jpg")
            continue

        with open(os.path.join(SRC_ANN, f"{stem}.json")) as f:
            ann = json.load(f)
        H = ann["images"][0]["height"]
        W = ann["images"][0]["width"]

        cat_fdi = {c["id"]: int(c["name"]) for c in ann["categories"]}
        mask = np.zeros((H, W), dtype=np.uint8)
        for a in ann["annotations"]:
            fdi = cat_fdi.get(a["category_id"])
            if fdi is None or fdi not in fdi_to_cls:
                continue
            cls = fdi_to_cls[fdi]
            for seg in a["segmentation"]:
                poly = np.array(seg, dtype=np.float32).reshape(-1, 2)
                cv2.fillPoly(mask, [poly.astype(np.int32)], cls)

        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(out_img, f"{stem}.png"), img)
        cv2.imwrite(os.path.join(out_lbl, f"{stem}.png"), mask)

    print(f"{split}: {len(df)} images -> {OUT}/{split}_Folder/")

os.makedirs(os.path.join(OUT, "Prompts_Folder"), exist_ok=True)
for xlsx in ["Train_text.xlsx", "Val_text.xlsx", "Test_text_original.xlsx"]:
    shutil.copy2(os.path.join(PROMPT_DIR, xlsx), os.path.join(OUT, "Prompts_Folder", xlsx))
shutil.copy2("data/INREDD_336/cls_to_fdi.json", os.path.join(OUT, "cls_to_fdi.json"))
print("copied Prompts_Folder + cls_to_fdi.json")
