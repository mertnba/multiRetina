#!/usr/bin/env python3

import argparse
from src.features.uno import extract_cls_features

def main():
    parser = argparse.ArgumentParser(description="Extract CLS vectors using RetFound.")
    parser.add_argument("--image-dir", required=True, help="Directory containing images (e.g. data/processed/train)")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained RetFound checkpoint (.pth)")
    parser.add_argument("--out", required=True, help="Output path to save .csv with CLS vectors")
    parser.add_argument("--model-name", default="RETFound_mae", help="Model name for timm.create_model")
    parser.add_argument("--image-size", type=int, default=224, help="Image size for resizing")
    args = parser.parse_args()

    extract_cls_features(
        image_dir=args.image_dir,
        checkpoint_path=args.checkpoint,
        output_csv=args.out,
        model_name=args.model_name,
        image_size=args.image_size
    )


if __name__ == "__main__":
    main()
