import os
import torch
import timm
import shutil
import subprocess
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import List
from tqdm import tqdm


def extract_cls_features(
    image_dir: str,
    checkpoint_path: str,
    output_csv: str,
    model_name: str = "RETFound_mae",
    image_size: int = 224,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="token")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print("[INFO] Loaded checkpoint:", msg)

    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    image_paths = list(Path(image_dir).rglob("*.jpg"))
    features = []
    filenames = []

    for img_path in tqdm(image_paths, desc="Extracting CLS features"):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(tensor).squeeze().cpu().numpy()
            features.append(feat)
            filenames.append(img_path.name)
        except Exception as e:
            print(f"[WARN] Skipping {img_path.name}: {e}")

    df = pd.DataFrame(features)
    df.insert(0, "filename", filenames)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("[DONE] CLS features saved to:", output_csv)


def run_automorph_segmentation(
    image_dir: str,
    output_dir: str,
    automorph_dir: str = "models/AutoMorph",
    config_path: str = "models/AutoMorph/configs/infer_macular.yaml",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    assert os.path.isdir(automorph_dir), "AutoMorph directory not found."

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "python", os.path.join(automorph_dir, "infer.py"),
        "--input_path", image_dir,
        "--output_path", output_dir,
        "--config", config_path,
        "--gpu", "0" if device == "cuda" else "-1"
    ]

    print("[INFO] Running AutoMorph segmentation via subprocess...")
    subprocess.run(cmd, check=True)
    print("[DONE] Segmentation saved to:", output_dir)
