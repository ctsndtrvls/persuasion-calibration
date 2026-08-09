"""
5-fold grouped cross-validation for persuasion uncertainty scores.

Fits train-fold parameters for:
  - argument quality A (z-scores + min-max on flipped dialogues)
  - epistemic marker normalization

Produces out-of-fold (OOF) scores and metrics.

Outputs under composite_score/04_cross_validation/:
  csv/oof_scores.csv
  csv/cv_metrics.csv
  csv/cv_vs_insample.csv
  png/cv_metrics_comparison.png

Example:
  cd src && python3 cross_validate_persuasion_uncertainty.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_persuasion_uncertainty_scores import (
    QUALITY_FLIP_COLS,
    apply_argument_quality_A,
    apply_minmax_01,
    compute_persuasion_scores,
    fit_argument_quality_A,
    fit_minmax_params,
)
from evaluate_persuasion_uncertainty import METHODS, evaluate_methods
from persuasion_uncertainty_viz import BG, METHOD_COLORS, METHOD_LABELS

FIXED_METHODS = ("U_token",)
CV_METHODS = [m for m in METHODS if m not in FIXED_METHODS]

COMPOSITE_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever" / "composite_score"
DEFAULT_FEATURES = COMPOSITE_DIR / "01_features" / "csv" / "persuasion_uncertainty_features.csv"
DEFAULT_SCORES = COMPOSITE_DIR / "02_scores" / "csv" / "persuasion_uncertainty_scores.csv"
DEFAULT_OUT = COMPOSITE_DIR / "04_cross_validation"
N_FOLDS = 5
RNG = np.random.default_rng(42)


def grouped_kfold_indices(groups: np.ndarray, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    shuffled = unique_groups.copy()
    RNG.shuffle(shuffled)
    folds = np.array_split(shuffled, n_folds)
    train_test: list[tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(groups))
    for test_groups in folds:
        test_mask = np.isin(groups, test_groups)
        train_mask = ~test_mask
        train_test.append((all_idx[train_mask], all_idx[test_mask]))
    return train_test


def run_oof_cv(features: pd.DataFrame, scores_insample: pd.DataFrame, n_folds: int = N_FOLDS) -> pd.DataFrame:
    score_cols = ["dialogue_id", "total_marker_n", *FIXED_METHODS]
    score_cols = [c for c in score_cols if c in scores_insample.columns]
    df = features.merge(
        scores_insample[score_cols],
        on="dialogue_id",
        how="left",
        suffixes=("", "_dup"),
    )
    if "total_marker_n_dup" in df.columns:
        df["total_marker_n"] = df["total_marker_n"].fillna(df["total_marker_n_dup"])
        df = df.drop(columns=["total_marker_n_dup"])

    groups = df["original_index"].to_numpy()
    oof = df.copy()
    for col in CV_METHODS:
        oof[col] = np.nan
    oof["cv_fold"] = -1

    for fold_id, (train_idx, test_idx) in enumerate(grouped_kfold_indices(groups, n_folds)):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx].copy()

        train_flipped = train[train["F"] == 1]
        if len(train_flipped) >= 2:
            a_params = fit_argument_quality_A(train_flipped)
        else:
            a_params = fit_argument_quality_A(train)
        a_test = apply_argument_quality_A(test, a_params)

        lo, hi = fit_minmax_params(train["total_marker_n"].fillna(0))
        u_marker = apply_minmax_01(test["total_marker_n"].fillna(0), lo, hi)

        scored = compute_persuasion_scores(test, a=a_test, u_marker=u_marker)
        for col in CV_METHODS:
            oof.loc[test.index, col] = scored[col].to_numpy()
        oof.loc[test.index, "cv_fold"] = fold_id

    return oof


def plot_cv_vs_insample(comparison: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    methods = comparison["method"].tolist()
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width / 2, comparison["brier_insample"], width, label="In-sample", color="#94A3B8")
    ax.bar(x + width / 2, comparison["brier_oof"], width, label="5-fold OOF", color="#1D4ED8")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Cross-validation: in-sample vs out-of-fold Brier")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grouped 5-fold CV for uncertainty scores.")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    features = pd.read_csv(args.features)
    scores_insample = pd.read_csv(args.scores)

    oof = run_oof_cv(features, scores_insample, n_folds=args.n_folds)
    metrics_oof = evaluate_methods(oof, METHODS)
    metrics_oof["evaluation"] = "oof_5fold"
    metrics_insample = evaluate_methods(scores_insample, METHODS)
    metrics_insample["evaluation"] = "insample"

    comparison = metrics_insample[["method", "label", "brier"]].rename(columns={"brier": "brier_insample"})
    comparison = comparison.merge(
        metrics_oof[["method", "brier"]].rename(columns={"brier": "brier_oof"}),
        on="method",
    )
    comparison["delta_brier"] = comparison["brier_oof"] - comparison["brier_insample"]

    out_csv = args.out_dir / "csv"
    out_png = args.out_dir / "png"
    out_csv.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)

    oof_cols = [
        "dialogue_id", "original_index", "E_i", "F", "S", "C0", "C_flip", "cv_fold",
        *METHODS,
    ]
    oof[oof_cols].to_csv(out_csv / "oof_scores.csv", index=False)
    pd.concat([metrics_insample, metrics_oof], ignore_index=True).to_csv(
        out_csv / "cv_metrics.csv", index=False
    )
    comparison.to_csv(out_csv / "cv_vs_insample.csv", index=False)
    plot_cv_vs_insample(comparison, out_png / "cv_metrics_comparison.png")

    summary = [
        "5-fold grouped cross-validation (by original_index)",
        f"n = {len(oof)} dialogues, folds = {args.n_folds}",
        "",
        "Brier (lower is better):",
    ]
    for _, r in comparison.sort_values("brier_oof").iterrows():
        summary.append(
            f"  {r['label']:30s}  in-sample={r['brier_insample']:.4f}  "
            f"OOF={r['brier_oof']:.4f}  Δ={r['delta_brier']:+.4f}"
        )
    summary_path = args.out_dir / "cv_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Saved OOF scores: {out_csv / 'oof_scores.csv'}")
    print(f"Saved CV metrics: {out_csv / 'cv_metrics.csv'}")
    print(f"Saved figure:     {out_png / 'cv_metrics_comparison.png'}")
    print(f"Saved summary:    {summary_path}")
    print()
    print("\n".join(summary[3:]))


if __name__ == "__main__":
    main()
