"""
Post-hoc recalibration of persuasion uncertainty scores (supervisor protocol).

Uses existing 5-fold grouped OOF raw scores. For each method and fold:
  - fit calibrator g on train folds only (Platt scaling or isotonic regression)
  - apply g(U) on held-out fold → OOF calibrated probabilities

Compares raw OOF vs calibrated OOF Brier / UCE (primary: isotonic; Platt reported too).

Outputs under composite_score/06_recalibration/:
  csv/oof_calibrated_scores.csv
  csv/recalibration_metrics.csv
  png/recalibration_brier_comparison.png
  png/reliability_raw_vs_calibrated.png
  recalibration_summary.txt

Example:
  cd src && python3 recalibrate_persuasion_uncertainty.py
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
from sklearn.isotonic import IsotonicRegression

from cross_validate_persuasion_uncertainty import N_FOLDS, grouped_kfold_indices
from evaluate_persuasion_uncertainty import (
    CLIP,
    METHODS,
    _metric_row,
    reliability_bins,
)
from incremental_persuasion_regression import fit_logistic, predict_proba
from persuasion_uncertainty_viz import BG, METHOD_COLORS, METHOD_LABELS

COMPOSITE_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever" / "composite_score"
DEFAULT_OOF = COMPOSITE_DIR / "04_cross_validation" / "csv" / "oof_scores.csv"
DEFAULT_SCORES = COMPOSITE_DIR / "02_scores" / "csv" / "persuasion_uncertainty_scores.csv"
DEFAULT_OUT = COMPOSITE_DIR / "06_recalibration"

CALIBRATORS = ("platt", "isotonic")
RELIABILITY_METHODS = ("U_self", "U_token", "U_hybrid")


def _logit(u: np.ndarray) -> np.ndarray:
    u = np.clip(u.astype(float), CLIP, 1.0 - CLIP)
    return np.log(u / (1.0 - u))


def _constant_calibrator(y_train: np.ndarray) -> float:
    return float(np.clip(y_train.mean(), CLIP, 1.0 - CLIP))


def fit_platt(u_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Return beta for logit(p) = X @ beta with X = [1, logit(U)]."""
    y = y_train.astype(float)
    if len(np.unique(y)) < 2:
        p = _constant_calibrator(y)
        return np.array([np.log(p / (1.0 - p)), 0.0])
    x = np.column_stack([np.ones(len(u_train)), _logit(u_train)])
    return fit_logistic(x, y)


def apply_platt(u: np.ndarray, beta: np.ndarray) -> np.ndarray:
    if len(beta) >= 2 and abs(beta[1]) < 1e-12:
        p = 1.0 / (1.0 + np.exp(-beta[0]))
        return np.full(len(u), np.clip(p, CLIP, 1.0 - CLIP))
    x = np.column_stack([np.ones(len(u)), _logit(u)])
    return np.clip(predict_proba(x, beta), CLIP, 1.0 - CLIP)


def fit_isotonic(u_train: np.ndarray, y_train: np.ndarray) -> IsotonicRegression | float:
    y = y_train.astype(float)
    if len(np.unique(y)) < 2:
        return _constant_calibrator(y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=CLIP, y_max=1.0 - CLIP)
    iso.fit(u_train.astype(float), y)
    return iso


def apply_isotonic(u: np.ndarray, model: IsotonicRegression | float) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(u), model)
    return np.clip(model.predict(u.astype(float)), CLIP, 1.0 - CLIP)


def load_oof(oof_csv: Path, scores_csv: Path) -> pd.DataFrame:
    oof = pd.read_csv(oof_csv)
    scores = pd.read_csv(scores_csv)
    merge_cols = ["dialogue_id", *[c for c in METHODS if c in scores.columns]]
    oof = oof.drop(columns=[c for c in METHODS if c in oof.columns], errors="ignore")
    oof = oof.merge(scores[merge_cols], on="dialogue_id", how="left")
    if "cv_fold" not in oof.columns or oof["cv_fold"].isna().any():
        groups = oof["original_index"].to_numpy()
        oof["cv_fold"] = -1
        for fold_id, (_, test_idx) in enumerate(grouped_kfold_indices(groups, N_FOLDS)):
            oof.loc[oof.index[test_idx], "cv_fold"] = fold_id
    return oof


def calibrate_oof(
    df: pd.DataFrame,
    method: str,
    calibrator: str,
    n_folds: int = N_FOLDS,
) -> np.ndarray:
    u_raw = df[method].astype(float).to_numpy()
    y = df["E_i"].astype(int).to_numpy()
    u_cal = np.full(len(df), np.nan)
    folds = df["cv_fold"].astype(int).to_numpy()

    for fold_id in range(n_folds):
        train_mask = folds != fold_id
        test_mask = folds == fold_id
        if not test_mask.any():
            continue
        u_tr, y_tr = u_raw[train_mask], y[train_mask]
        if calibrator == "platt":
            beta = fit_platt(u_tr, y_tr)
            u_cal[test_mask] = apply_platt(u_raw[test_mask], beta)
        elif calibrator == "isotonic":
            model = fit_isotonic(u_tr, y_tr)
            u_cal[test_mask] = apply_isotonic(u_raw[test_mask], model)
        else:
            raise ValueError(f"Unknown calibrator: {calibrator}")

    return u_cal


def evaluate_raw_and_calibrated(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    y = df["E_i"].astype(int).to_numpy()
    rows: list[dict] = []
    for method in methods:
        if method not in df.columns:
            continue
        u_raw = df[method].astype(float).to_numpy()
        raw_row = _metric_row(method, u_raw, y)
        raw_row["calibrator"] = "raw_oof"
        raw_row["brier_cal"] = raw_row["brier"]
        raw_row["uce_cal"] = raw_row["uce"]
        raw_row["delta_brier"] = 0.0
        raw_row["delta_uce"] = 0.0
        rows.append(raw_row)

        for cal in CALIBRATORS:
            col = f"{method}__{cal}"
            if col not in df.columns:
                continue
            u_cal = df[col].astype(float).to_numpy()
            cal_row = _metric_row(method, u_cal, y)
            cal_row["calibrator"] = cal
            cal_row["brier_raw"] = raw_row["brier"]
            cal_row["uce_raw"] = raw_row["uce"]
            cal_row["brier_cal"] = cal_row["brier"]
            cal_row["uce_cal"] = cal_row["uce"]
            cal_row["delta_brier"] = cal_row["brier_cal"] - raw_row["brier"]
            cal_row["delta_uce"] = cal_row["uce_cal"] - raw_row["uce"]
            rows.append(cal_row)
    return pd.DataFrame(rows)


def plot_brier_comparison(metrics: pd.DataFrame, out_path: Path) -> None:
    iso = metrics[metrics["calibrator"].isin(["raw_oof", "isotonic"])].copy()
    methods = [m for m in METHODS if m in iso["method"].unique()]
    x = np.arange(len(methods))
    width = 0.35
    raw = iso[iso["calibrator"] == "raw_oof"].set_index("method")
    cal = iso[iso["calibrator"] == "isotonic"].set_index("method")

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG)
    ax.bar(x - width / 2, [raw.loc[m, "brier"] for m in methods], width, label="Raw OOF", color="#94A3B8")
    ax.bar(x + width / 2, [cal.loc[m, "brier_cal"] for m in methods], width, label="Isotonic OOF", color="#1D4ED8")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Recalibration: raw vs isotonic (5-fold OOF)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_reliability_raw_vs_cal(
    df: pd.DataFrame,
    methods: tuple[str, ...],
    out_path: Path,
) -> None:
    y = df["E_i"].astype(int).to_numpy()
    n = len(methods)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), squeeze=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle("Reliability: raw OOF (top) vs isotonic OOF (bottom)", fontsize=13, weight="bold")

    for j, method in enumerate(methods):
        if method not in df.columns:
            continue
        u_raw = df[method].astype(float).to_numpy()
        u_cal = df[f"{method}__isotonic"].astype(float).to_numpy()
        for row, u, title_suffix in ((0, u_raw, "raw"), (1, u_cal, "isotonic")):
            ax = axes[row, j]
            bins = reliability_bins(u, y)
            if not bins.empty:
                ax.plot([0, 1], [0, 1], "--", color="#CBD5E1", lw=1)
                ax.scatter(bins["mean_u"], bins["mean_e"], s=bins["count"] * 3, alpha=0.85,
                           color=METHOD_COLORS.get(method, "#64748B"), edgecolors="white")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{METHOD_LABELS.get(method, method)} ({title_suffix})", fontsize=9)
            if j == 0:
                ax.set_ylabel("Observed error rate")
            if row == 1:
                ax.set_xlabel("Predicted uncertainty")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recalibrate OOF persuasion uncertainty scores.")
    ap.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    df = load_oof(args.oof, args.scores)
    methods = [m for m in METHODS if m in df.columns and df[m].notna().all()]
    if not methods:
        raise RuntimeError("No complete method columns found for recalibration.")

    out = df[["dialogue_id", "original_index", "E_i", "cv_fold", *methods]].copy()
    for method in methods:
        for cal in CALIBRATORS:
            out[f"{method}__{cal}"] = calibrate_oof(df, method, cal, n_folds=args.n_folds)

    metrics = evaluate_raw_and_calibrated(out, methods)
    metrics = metrics.sort_values(["method", "calibrator"])

    out_csv = args.out_dir / "csv"
    out_png = args.out_dir / "png"
    out_csv.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_csv / "oof_calibrated_scores.csv", index=False)
    metrics.to_csv(out_csv / "recalibration_metrics.csv", index=False)
    plot_brier_comparison(metrics, out_png / "recalibration_brier_comparison.png")
    rel_methods = tuple(m for m in RELIABILITY_METHODS if m in methods)
    if rel_methods:
        plot_reliability_raw_vs_cal(out, rel_methods, out_png / "reliability_raw_vs_calibrated.png")

    iso = metrics[metrics["calibrator"] == "isotonic"].sort_values("brier_cal")
    lines = [
        "Recalibration summary (5-fold grouped OOF, train-fold fit only)",
        f"n = {len(df)} dialogues, methods = {len(methods)}",
        "",
        "Isotonic recalibration — Brier (lower is better):",
    ]
    for _, r in iso.iterrows():
        lines.append(
            f"  {r['label']:30s}  raw={r['brier_raw']:.4f}  "
            f"cal={r['brier_cal']:.4f}  Δ={r['delta_brier']:+.4f}"
        )
    lines += ["", "Key comparison (isotonic):"]
    for base, comp in (("U_self", "U_hybrid"), ("U_self", "U_pers"), ("U_token", "U_hybrid")):
        if base not in methods or comp not in methods:
            continue
        b = iso[iso["method"] == base].iloc[0]["brier_cal"]
        c = iso[iso["method"] == comp].iloc[0]["brier_cal"]
        winner = comp if c < b else base
        lines.append(f"  {METHOD_LABELS.get(comp, comp)} vs {METHOD_LABELS.get(base, base)}: "
                     f"cal Brier {c:.4f} vs {b:.4f} → {METHOD_LABELS.get(winner, winner)} better")

    summary_path = args.out_dir / "recalibration_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved calibrated OOF: {out_csv / 'oof_calibrated_scores.csv'}")
    print(f"Saved metrics:        {out_csv / 'recalibration_metrics.csv'}")
    print(f"Saved figures:          {out_png}")
    print(f"Saved summary:          {summary_path}")
    print()
    print("\n".join(lines[3:]))


if __name__ == "__main__":
    main()
