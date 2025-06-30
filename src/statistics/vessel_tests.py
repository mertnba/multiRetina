import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from typing import List, Tuple
from pathlib import Path


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x), np.asarray(y)
    gt = np.sum(x[:, None] > y)
    lt = np.sum(x[:, None] < y)
    return (gt - lt) / (len(x) * len(y))


def run_kruskal(df: pd.DataFrame, features: List[str], label_col="Class") -> pd.DataFrame:
    results = []
    for feat in features:
        groups = [df.loc[df[label_col] == c, feat].values for c in sorted(df[label_col].unique())]
        stat, p = kruskal(*groups)
        results.append({"feature": feat, "KW_stat": stat, "p_raw": p})
    df_kw = pd.DataFrame(results)
    df_kw["p_adj"] = multipletests(df_kw["p_raw"], method="fdr_bh")[1]
    return df_kw.sort_values("p_adj")


def run_pairwise_tests(df: pd.DataFrame, features: List[str], label_col="Class", pairs=[(0, 1), (1, 2), (2, 3), (3, 4)]) -> pd.DataFrame:
    rows, all_pvals = [], []
    for feat in features:
        for a, b in pairs:
            x = df.loc[df[label_col] == a, feat].values
            y = df.loc[df[label_col] == b, feat].values
            u, p_raw = mannwhitneyu(x, y, alternative="two-sided")
            delta = cliffs_delta(x, y)
            rows.append({"feature": feat, "pair": f"{a} vs {b}", "U": u, "p_raw": p_raw, "delta": delta})
            all_pvals.append(p_raw)
    p_adj = multipletests(all_pvals, method="fdr_bh")[1]
    for i, row in enumerate(rows):
        row["p_adj"] = p_adj[i]
    return pd.DataFrame(rows).sort_values(["feature", "pair"])


def run_trend_test(df: pd.DataFrame, features: List[str], label_col="Class") -> pd.DataFrame:
    rows, pvals = [], []
    for feat in features:
        res = pg.trend(df, dv=feat, by=label_col, order=sorted(df[label_col].unique()))
        rows.append({
            "feature": feat,
            "JT_stat": res.loc[0, "T"],
            "p_raw": res.loc[0, "p-val"],
            "direction": "increasing" if res.loc[0, "T"] > 0 else "decreasing"
        })
        pvals.append(res.loc[0, "p-val"])
    p_adj = multipletests(pvals, method="fdr_bh")[1]
    for i in range(len(rows)):
        rows[i]["p_adj"] = p_adj[i]
    return pd.DataFrame(rows).sort_values("p_adj")


def run_all_tests(input_csv: str, label_col="Class", output_dir="results/stats", save_csv=True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    features = [c for c in df.columns if c != label_col]

    print("[INFO] Running Kruskal–Wallis tests...")
    df_kw = run_kruskal(df, features, label_col)
    if save_csv:
        df_kw.to_csv(os.path.join(output_dir, "kruskal_results.csv"), index=False)

    print("[INFO] Running Mann–Whitney + Cliff’s δ tests...")
    df_mw = run_pairwise_tests(df, features, label_col)
    if save_csv:
        df_mw.to_csv(os.path.join(output_dir, "mannwhitney_results.csv"), index=False)

    print("[INFO] Running Jonckheere–Terpstra trend test...")
    df_jt = run_trend_test(df, features, label_col)
    if save_csv:
        df_jt.to_csv(os.path.join(output_dir, "trend_test_results.csv"), index=False)

    print("[DONE] All statistical tests completed.")
    return df_kw, df_mw, df_jt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to input .csv file with features and class labels")
    parser.add_argument("--label-col", default="Class", help="Column name for target labels (default: Class)")
    parser.add_argument("--out-dir", default="results/stats", help="Where to save output .csvs")
    args = parser.parse_args()

    run_all_tests(args.csv, label_col=args.label_col, output_dir=args.out_dir)
