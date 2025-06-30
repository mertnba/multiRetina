#!/usr/bin/env python3

import argparse
from src.features.uno import run_automorph_segmentation

def main():
    parser = argparse.ArgumentParser(description="Run AutoMorph segmentation.")
    parser.add_argument("--image-dir", required=True, help="Directory of input images")
    parser.add_argument("--out", required=True, help="Directory to save segmentation outputs")
    parser.add_argument("--automorph-dir", default="models/AutoMorph", help="Path to AutoMorph repo")
    parser.add_argument("--config", default="models/AutoMorph/configs/infer_macular.yaml", help="Path to AutoMorph YAML config")
    args = parser.parse_args()

    run_automorph_segmentation(
        image_dir=args.image_dir,
        output_dir=args.out,
        automorph_dir=args.automorph_dir,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
