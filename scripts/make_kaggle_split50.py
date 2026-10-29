"""Build a reproducible 50% subset for KAGGLE, test fixed.

Halves the ORIGINAL split with a fixed seed:
  - train: 50% of Train_Folder  (418 -> 209)
  - val  : 50% of Val_Folder    ( 90 ->  45)
  - test : unchanged (87)

Writes new prompt tables under Prompts_Folder_50/. No image copying — the
config keeps pointing TRAIN_PATH/VAL_PATH at the original Train_Folder/Val_Folder.
"""
import os
import json
import shutil
import numpy as np
import pandas as pd

BASE = "data/KAGGLE"
SPLIT_SEED = 1  # fixed for reproducibility
PROMPT_DIR = os.path.join(BASE, "Prompts_Folder_50")


def stems(folder):
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(BASE, folder, "img"))
    )


def main():
    train = stems("Train_Folder")   # 418
    val = stems("Val_Folder")       # 90

    rng = np.random.default_rng(SPLIT_SEED)
    train_idx = rng.permutation(len(train))
    val_idx = rng.permutation(len(val))
    train_50 = [train[i] for i in train_idx[: len(train) // 2]]
    val_50 = [val[i] for i in val_idx[: len(val) // 2]]

    os.makedirs(PROMPT_DIR, exist_ok=True)
    for name, items in [("Train", train_50), ("Val", val_50)]:
        df = pd.DataFrame({
            "Image": [f"{s}.png" for s in items],
            "Ground Truth": [f"{s}.png" for s in items],
            "Description": ["teeth"] * len(items),
        })
        df.to_excel(os.path.join(PROMPT_DIR, f"{name}_text.xlsx"), index=False)

    shutil.copy2(
        os.path.join(BASE, "Prompts_Folder", "Test_text_original.xlsx"),
        os.path.join(PROMPT_DIR, "Test_text_original.xlsx"),
    )

    manifest = {"split_seed": SPLIT_SEED, "train": train_50, "val": val_50}
    with open(os.path.join(PROMPT_DIR, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"train {len(train)} -> {len(train_50)}")
    print(f"val   {len(val)} -> {len(val_50)}")
    print(f"train ∩ val = {len(set(train_50) & set(val_50))}")


if __name__ == "__main__":
    main()
