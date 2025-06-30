#!/usr/bin/env python3

import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Preprocess and split dataset into train/val/test.")
    parser.add_argument("--csv", required=True, help="Path to input CSV with columns: patient_id, class, file_name")
    parser.add_argument("--img-dir", required=True, help="Directory where class folders are located")
    parser.add_argument("--classes", nargs="+", required=True, help="Valid class names (must match folders inside --img-dir)")
    parser.add_argument("--out-dir", default="data/processed", help="Where to save split images")

    args = parser.parse_args()
    class_dirs = [os.path.join(args.img_dir, cls) for cls in args.classes]

    cmd = [
        "python", "src/preprocessing/structured_split.py",
        "--csv", args.csv,
        "--img-dirs", *class_dirs,
        "--classes", *args.classes,
        "--output", args.out_dir,
    ]

    print("\n[INFO] Running structured split...")
    print("Command:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
