"""Build a reproducible N% subset for a dataset (KAGGLE / DUAL / ...), test fixed.

Randomly samples PERCENT% from the ORIGINAL Train_Folder and PERCENT% from the
ORIGINAL Val_Folder with a fixed seed, writes new prompt tables under
Prompts_Folder_{PERCENT}/. No image copying — train/val keep pointing at the
original Train_Folder/Val_Folder.

Usage:
  python scripts/make_split.py --dataset KAGGLE --percent 50 --seed 1
  python scripts/make_split.py --dataset DUAL   --percent 50 --seed 1

Nested subsets: with the same seed, the PERCENT% set is a prefix of the same
permutation, so 10% ⊂ 25% ⊂ 50% ⊂ 100% (clean data-efficiency comparison).
"""
import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd


def stems(base, folder):
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(base, folder, "img"))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="KAGGLE")
    ap.add_argument("--percent", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    base = f"data/{args.dataset}"
    prompt_dir = os.path.join(base, f"Prompts_Folder_{args.percent}")

    train = stems(base, "Train_Folder")
    val = stems(base, "Val_Folder")

    rng = np.random.default_rng(args.seed)
    train_idx = rng.permutation(len(train))
    val_idx = rng.permutation(len(val))
    n_train = len(train) * args.percent // 100
    n_val = len(val) * args.percent // 100
    train_sub = [train[i] for i in train_idx[:n_train]]
    val_sub = [val[i] for i in val_idx[:n_val]]

    os.makedirs(prompt_dir, exist_ok=True)
    for name, items in [("Train", train_sub), ("Val", val_sub)]:
        df = pd.DataFrame({
            "Image": [f"{s}.png" for s in items],
            "Ground Truth": [f"{s}.png" for s in items],
            "Description": ["teeth"] * len(items),
        })
        df.to_excel(os.path.join(prompt_dir, f"{name}_text.xlsx"), index=False)

    shutil.copy2(
        os.path.join(base, "Prompts_Folder", "Test_text_original.xlsx"),
        os.path.join(prompt_dir, "Test_text_original.xlsx"),
    )

    manifest = {
        "dataset": args.dataset,
        "percent": args.percent,
        "split_seed": args.seed,
        "train": train_sub,
        "val": val_sub,
    }
    with open(os.path.join(prompt_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"dataset={args.dataset} percent={args.percent}% seed={args.seed}")
    print(f"train {len(train)} -> {n_train}")
    print(f"val   {len(val)} -> {n_val}")


if __name__ == "__main__":
    main()
