#!/usr/bin/env python3

import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Dataset Split Pipeline")
    parser.add_argument("--csv", required=True, help="Path to label CSV")
    parser.add_argument("--img-dir", required=True, help="Directory containing all image folders")
    parser.add_argument("--classes", nargs="+", required=True, help="List of valid class names (must match folder names)")
    parser.add_argument("--out-dir", default="data/processed", help="Where to write processed train/val/test folders")
    args = parser.parse_args()

    class_dirs = [f"{args.img_dir}/{cls}" for cls in args.classes]

    cmd = [
        "python", "src/preprocessing/structured_split.py",
        "--csv", args.csv,
        "--img-dirs", *class_dirs,
        "--classes", *args.classes,
        "--output", args.out_dir,
    ]

    print("[INFO] Running split with command:")
    print("       " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
