#!/usr/bin/env python3

import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Run statistical tests on morphology features.")
    parser.add_argument("--csv", required=True, help="Path to feature CSV file (e.g. features/morphology/train.csv)")
    parser.add_argument("--out-dir", default="results/stats", help="Directory to save results")
    parser.add_argument("--label-col", default="Class", help="Name of label column (default: Class)")
    args = parser.parse_args()

    cmd = [
        "python", "src/statistics/vessel_tests.py",
        "--csv", args.csv,
        "--label-col", args.label_col,
        "--out-dir", args.out_dir,
    ]

    print("[INFO] Running statistical tests...")
    print("Command:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
