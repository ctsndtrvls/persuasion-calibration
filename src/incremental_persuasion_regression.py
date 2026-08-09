"""
Incremental logistic regression: do persuasion signals add beyond baselines?

Supervisor protocol:
  baseline (self):  logit P(E) = a0 + a1*(1-C0)
  extended (self):  + F, F*S, F*(1-A), F*(1-C_flip)

  baseline (token): logit P(E) = b0 + b1*(1-token_conf)   [where available]
  extended (token): + persuasion signals

Uses grouped 5-fold CV for out-of-fold predictions.

Outputs under composite_score/05_incremental_regression/

Example:
  cd src && python3 incremental_persuasion_regression.py
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

from cross_validate_persuasion_uncertainty import grouped_kfold_indices
from persuasion_tokenprob import PERSUASION214_TOKENPROB, load_persuasion_tokenprob
from evaluate_persuasion_uncertainty import (
    auroc_score,
    brier_score,
    log_loss,
    uce_score,
)
from persuasion_uncertainty_viz import BG

COMPOSITE_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever" / "composite_score"
DEFAULT_FEATURES = COMPOSITE_DIR / "01_features" / "csv" / "persuasion_uncertainty_features.csv"
DEFAULT_SCORES = COMPOSITE_DIR / "02_scores" / "csv" / "persuasion_uncertainty_scores.csv"
DEFAULT_TOKEN = PERSUASION214_TOKENPROB
DEFAULT_OUT = COMPOSITE_DIR / "05_incremental_regression"
N_FOLDS = 5
MAX_ITER = 100
CLIP = 1e-6


def load_with_token(features: pd.DataFrame, scores: pd.DataFrame, token_csv: Path) -> pd.DataFrame:
    df = features.merge(scores[["dialogue_id", "total_marker_n"]], on="dialogue_id", how="left")
    tok = load_persuasion_tokenprob(token_csv=token_csv, original_indices=features["original_index"])
    df = df.merge(tok[["original_index", "U_token", "confidence"]], on="original_index", how="left")
    df["U_self"] = 1.0 - df["C0"]
    return df


def persuasion_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    f = df["F"].astype(float).to_numpy()
    s = df["S"].fillna(0.0).astype(float).to_numpy()
    a = df["A"].fillna(0.5).astype(float).to_numpy()
    c_flip = df["C_flip"].fillna(0.5).astype(float).to_numpy()
    return np.column_stack([f, f * s, f * (1.0 - a), f * (1.0 - c_flip)])


def build_design(df: pd.DataFrame, baseline: str, extended: bool) -> np.ndarray:
    n = len(df)
    if baseline == "self":
        base = (1.0 - df["C0"].astype(float)).to_numpy().reshape(-1, 1)
    elif baseline == "token":
        base = df["U_token"].astype(float).to_numpy().reshape(-1, 1)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    parts = [np.ones((n, 1)), base]
    if extended:
        parts.append(persuasion_feature_matrix(df))
    return np.hstack(parts)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-6) -> np.ndarray:
    """IRLS for binary logistic regression with tiny L2."""
    beta = np.zeros(X.shape[1])
    y = y.astype(float)
    for _ in range(MAX_ITER):
        eta = np.clip(X @ beta, -20, 20)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-8, None)
        z = eta + (y - mu) / w
        xw = X * np.sqrt(w)[:, None]
        zw = z * np.sqrt(w)
        reg = np.eye(X.shape[1]) * l2
        reg[0, 0] = 0.0
        beta_new = np.linalg.lstsq(xw.T @ xw + reg, xw.T @ zw, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
    return beta


def predict_proba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    eta = np.clip(X @ beta, -20, 20)
    return 1.0 / (1.0 + np.exp(-eta))


def _metric_row(model: str, y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, CLIP, 1.0 - CLIP)
    return {
        "model": model,
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
        "auroc": auroc_score(p, y),
        "uce": uce_score(p, y),
        "n": len(y),
        "error_rate": float(y.mean()),
    }


def oof_logistic_cv(
    df: pd.DataFrame,
    baseline: str,
    extended: bool,
    n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return OOF probabilities and fold ids."""
    if baseline == "token":
        work = df[df["U_token"].notna()].copy()
    else:
        work = df.copy()

    y = work["E_i"].astype(int).to_numpy()
    groups = work["original_index"].to_numpy()
    oof_p = np.full(len(work), np.nan)
    fold_ids = np.full(len(work), -1, dtype=int)

    for fold_id, (train_idx, test_idx) in enumerate(grouped_kfold_indices(groups, n_folds)):
        train = work.iloc[train_idx]
        test = work.iloc[test_idx]
        X_train = build_design(train, baseline, extended)
        X_test = build_design(test, baseline, extended)
        y_train = train["E_i"].astype(int).to_numpy()
        beta = fit_logistic(X_train, y_train)
        oof_p[test_idx] = predict_proba(X_test, beta)
        fold_ids[test_idx] = fold_id

    return oof_p, y, fold_ids, work


def fit_full_sample(df: pd.DataFrame, baseline: str, extended: bool) -> tuple[np.ndarray, pd.DataFrame]:
    if baseline == "token":
        work = df[df["U_token"].notna()].copy()
    else:
        work = df.copy()
    X = build_design(work, baseline, extended)
    y = work["E_i"].astype(int).to_numpy()
    beta = fit_logistic(X, y)
    if baseline == "self":
        names = ["intercept", "1-C0"]
    else:
        names = ["intercept", "U_token"]
    if extended:
        names += ["F", "F*S", "F*(1-A)", "F*(1-C_flip)"]
    coef = pd.DataFrame({"term": names, "coef": beta})
    return predict_proba(X, beta), work, coef


def plot_incremental(metrics: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle(title, fontsize=12, weight="bold")
    specs = [("brier", "Brier ↓", True), ("auroc", "AUROC ↑", False), ("log_loss", "Log loss ↓", True)]
    for ax, (col, label, asc) in zip(axes, specs):
        sub = metrics.sort_values(col, ascending=asc)
        ax.barh(sub["model"], sub[col], color="#3B82F6")
        ax.set_title(label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Incremental persuasion logistic regression.")
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--token-csv", type=Path, default=DEFAULT_TOKEN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    features = pd.read_csv(args.features)
    scores = pd.read_csv(args.scores)
    df = load_with_token(features, scores, args.token_csv)

    models = [
        ("self_baseline", "self", False),
        ("self_extended", "self", True),
        ("token_baseline", "token", False),
        ("token_extended", "token", True),
    ]

    metric_rows = []
    coef_rows = []
    oof_frames = []

    for model_name, baseline, extended in models:
        oof_p, y, fold_ids, work = oof_logistic_cv(df, baseline, extended, n_folds=args.n_folds)
        row = _metric_row(model_name, y, oof_p)
        row["baseline_type"] = baseline
        row["extended"] = extended
        metric_rows.append(row)

        oof_df = work[["dialogue_id", "original_index", "E_i"]].copy()
        oof_df["model"] = model_name
        oof_df["p_hat"] = oof_p
        oof_df["cv_fold"] = fold_ids
        oof_frames.append(oof_df)

        _, _, coef = fit_full_sample(df, baseline, extended)
        coef["model"] = model_name
        coef_rows.append(coef)

    metrics = pd.DataFrame(metric_rows)
    coefs = pd.concat(coef_rows, ignore_index=True)
    oof_all = pd.concat(oof_frames, ignore_index=True)

    # Incremental deltas (extended - baseline)
    deltas = []
    for baseline in ("self", "token"):
        base = metrics[metrics["model"] == f"{baseline}_baseline"].iloc[0]
        ext = metrics[metrics["model"] == f"{baseline}_extended"].iloc[0]
        deltas.append({
            "comparison": f"{baseline}: extended vs baseline",
            "n": ext["n"],
            "delta_brier": ext["brier"] - base["brier"],
            "delta_log_loss": ext["log_loss"] - base["log_loss"],
            "delta_auroc": ext["auroc"] - base["auroc"],
            "delta_uce": ext["uce"] - base["uce"],
        })
    delta_df = pd.DataFrame(deltas)

    out_csv = args.out_dir / "csv"
    out_png = args.out_dir / "png"
    out_csv.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(out_csv / "incremental_model_metrics.csv", index=False)
    delta_df.to_csv(out_csv / "incremental_deltas.csv", index=False)
    coefs.to_csv(out_csv / "incremental_coefficients.csv", index=False)
    oof_all.to_csv(out_csv / "incremental_oof_predictions.csv", index=False)

    plot_incremental(
        metrics,
        out_png / "incremental_self_metrics.png",
        "Incremental regression — self-report baseline (OOF, n=214)",
    )
    token_n = int(df["U_token"].notna().sum())
    token_metrics = metrics[metrics["baseline_type"] == "token"]
    plot_incremental(
        token_metrics,
        out_png / "incremental_token_metrics.png",
        f"Incremental regression — token-prob baseline (OOF, n={token_n})",
    )

    summary = [
        "Incremental logistic regression (5-fold grouped OOF)",
        "",
        f"Token-prob coverage: {token_n} / {len(df)} dialogues",
        "",
        "Self-report baseline (n=214):",
    ]
    for _, r in metrics[metrics["baseline_type"] == "self"].iterrows():
        summary.append(
            f"  {r['model']:18s}  Brier={r['brier']:.4f}  AUROC={r['auroc']:.3f}  logloss={r['log_loss']:.3f}"
        )
    summary += ["", f"Token-prob baseline (n={token_n}):"]
    for _, r in metrics[metrics["baseline_type"] == "token"].iterrows():
        summary.append(
            f"  {r['model']:18s}  Brier={r['brier']:.4f}  AUROC={r['auroc']:.3f}  logloss={r['log_loss']:.3f}"
        )
    summary += ["", "Incremental gain (negative delta Brier = extended better):"]
    for _, r in delta_df.iterrows():
        summary.append(
            f"  {r['comparison']:30s}  ΔBrier={r['delta_brier']:+.4f}  "
            f"ΔAUROC={r['delta_auroc']:+.3f}  n={int(r['n'])}"
        )

    summary_path = args.out_dir / "incremental_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Saved metrics: {out_csv / 'incremental_model_metrics.csv'}")
    print(f"Saved figure:  {out_png}")
    print(f"Saved summary: {summary_path}")
    print()
    print("\n".join(summary[4:]))


if __name__ == "__main__":
    main()
