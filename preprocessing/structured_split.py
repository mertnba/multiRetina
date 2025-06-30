import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import List


def load_labels(csv_path: str, valid_classes: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop extra columns (optional)
    drop_cols = ["sex", "year", "image_width", "image_hight", "subcategory", "condition", "eye"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    df = df[df["class"].isin(valid_classes)].copy()
    return df


def resolve_multi_class_patients(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("patient_id")["class"].unique().reset_index()
    freq = df["class"].value_counts().to_dict()

    def pick_least_frequent(classes):
        return min(classes, key=lambda c: freq.get(c, 0))

    grouped["chosen_class"] = grouped["class"].apply(pick_least_frequent)
    df = df.merge(grouped[["patient_id", "chosen_class"]], on="patient_id")
    df = df[df["class"] == df["chosen_class"]].drop(columns="chosen_class")
    return df


def stratified_split(df: pd.DataFrame, seed=42):
    unique_patients = df[["patient_id", "class"]].drop_duplicates("patient_id")
    train_ids, temp_ids = train_test_split(unique_patients, test_size=0.3, stratify=unique_patients["class"], random_state=seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, stratify=temp_ids["class"], random_state=seed)

    train_ids["split"] = "train"
    val_ids["split"] = "val"
    test_ids["split"] = "test"
    split_df = pd.concat([train_ids, val_ids, test_ids])

    return df.merge(split_df[["patient_id", "split"]], on="patient_id")


def copy_images(df: pd.DataFrame, source_paths: dict, output_dir: str):
    for _, row in df.iterrows():
        fname = row["file_name"] + ".jpg"
        label = row["class"]
        split = row["split"]

        src_dir = source_paths.get(label)
        if src_dir is None:
            continue

        src_path = os.path.join(src_dir, fname)
        dst_folder = os.path.join(output_dir, split, label)
        os.makedirs(dst_folder, exist_ok=True)
        dst_path = os.path.join(dst_folder, fname)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)


def verify_counts(base_dir: str, classes: List[str]):
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base_dir, split)
        counts = {}
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            counts[cls] = len([f for f in os.listdir(cls_dir) if f.lower().endswith(".jpg")]) if os.path.exists(cls_dir) else 0
        print(f"{split.upper()} counts:", counts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file with labels")
    parser.add_argument("--img-dirs", nargs="+", required=True, help="Image directories (in order of class)")
    parser.add_argument("--classes", nargs="+", required=True, help="List of valid class names")
    parser.add_argument("--output", default="data/processed", help="Output base directory")
    args = parser.parse_args()

    source_paths = dict(zip(args.classes, args.img_dirs))

    print("[INFO] Loading and cleaning label data...")
    df = load_labels(args.csv, valid_classes=args.classes)
    df = resolve_multi_cl
