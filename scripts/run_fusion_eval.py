#!/usr/bin/env python3

import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Run fusion evaluation (RetFound + AutoMorph)")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all", help="Which split to run")
    parser.add_argument("--cls-dir", default="features/cls_vectors", help="Directory with RetFound CLS .csv files")
    parser.add_argument("--morph-dir", default="features/morphology", help="Directory with AutoMorph .csv files")
    parser.add_argument("--suffix", default="cls_vectors.csv", help="Filename suffix (default: cls_vectors.csv)")
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        cls_csv = os.path.join(args.cls_dir, f"{split}_{args.suffix}")
        morph_csv = os.path.join(args.morph_dir, f"{split}.csv")

        print(f"\n[INFO] Running fusion classifier for {split} set")
        cmd = [
            "python", "src/fusion/run_classifier.py",
            "--cls", cls_csv,
            "--morph", morph_csv
        ]

        print("Command:\n  " + " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
