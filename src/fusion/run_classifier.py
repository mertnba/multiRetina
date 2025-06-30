import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.utils.data as D
from pathlib import Path


class TabDataset(D.Dataset):
    def __init__(self, X, y):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, out_dim)
        )
    def forward(self, x): return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, num_classes):
    model = MLP(in_dim=X_train.shape[1], out_dim=num_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    train_ds = TabDataset(X_train, y_train)
    val_ds   = TabDataset(X_val, y_val)
    train_dl = D.DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = D.DataLoader(val_ds, batch_size=128)

    best_auc, best_state = 0, None
    for epoch in range(1, 51):
        model.train()
        for x, y in train_dl:
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad(); loss.backward(); opt.step()

        # Evaluate
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for x, y in val_dl:
                out = model(x)
                y_true.append(y)
                y_prob.append(out.softmax(1))
        y_true = torch.cat(y_true).numpy()
        y_prob = torch.cat(y_prob).numpy()
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model


def evaluate(model, X_test, y_test):
    model.eval()
    x = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_prob = model(x).softmax(1).numpy()
    y_pred = y_prob.argmax(1)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Cohen Kappa:", cohen_kappa_score(y_test, y_pred, weights="quadratic"))
    print("ROC AUC (ovr):", roc_auc_score(y_test, y_prob, multi_class="ovr"))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def load_and_align(cls_csv: str, morph_csv: str):
    cls_df = pd.read_csv(cls_csv)
    morph_df = pd.read_csv(morph_csv)

    # Join on filename
    cls_df = cls_df.set_index("filename")
    morph_df = morph_df.set_index("filename")

    common = cls_df.index.intersection(morph_df.index)
    cls_df = cls_df.loc[common]
    morph_df = morph_df.loc[common]

    merged = pd.concat([cls_df, morph_df], axis=1)
    merged["label"] = merged.index.str.split("_").str[1].astype(int)  # e.g., train_2_xxx.jpg → 2
    return merged


def main():
    parser = argparse.ArgumentParser(description="Run MLP classifier on fused features.")
    parser.add_argument("--cls", required=True, help="Path to CLS vector .csv (RetFound)")
    parser.add_argument("--morph", required=True, help="Path to morphology .csv (AutoMorph)")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-split", type=float, default=0.25, help="Validation split ratio from train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_and_align(args.cls, args.morph)
    y = df["label"]
    X = df.drop(columns=["label"])

    # Split: train → val → test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=args.test_split, stratify=y, random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=args.val_split, stratify=y_trainval, random_state=args.seed)

    # Normalize
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    num_classes = y.nunique()

    print("\n[INFO] Training MLP on fused features...")
    model = train_mlp(X_train, y_train, X_val, y_val, num_classes)

    print("\n[INFO] Evaluating on test set...")
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
