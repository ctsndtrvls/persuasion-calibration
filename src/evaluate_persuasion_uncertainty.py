"""
Evaluate persuasion uncertainty scores vs baselines (supervisor protocol).

Reads persuasion_uncertainty_scores.csv, computes:
  - Brier, AUROC, AUPRC, UCE, risk at 80% coverage
  - Ablation variants (F → F+S → F+S+A → full U^pers)
  - Paired cluster-bootstrap CIs (by original_index)

Outputs under composite_score/03_evaluation/:
  csv/  — metrics, ablation, bootstrap, risk-coverage tables
  png/  — comparison bars, ROC, reliability, risk-coverage, ablation, bootstrap

Example:
  cd src && python3 evaluate_persuasion_uncertainty.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from build_persuasion_uncertainty_scores import W_A, W_C, W_F, W_S
from persuasion_uncertainty_viz import (
    METHOD_COLORS,
    METHOD_LABELS,
    plot_ablation,
    plot_bootstrap_ci,
    plot_metrics_bar,
    plot_reliability,
    plot_risk_coverage,
    plot_roc_curves,
)

COMPOSITE_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever" / "composite_score"
DEFAULT_SCORES = COMPOSITE_DIR / "02_scores" / "csv" / "persuasion_uncertainty_scores.csv"
DEFAULT_OUT = COMPOSITE_DIR / "03_evaluation"

METHODS = ["U_self", "U_token", "U_marker", "U_flip", "U_pers", "U_hybrid"]
N_BINS = 5
N_BOOT = 5000
CLIP = 1e-6
RNG = np.random.default_rng(42)


def _clip_u(u: np.ndarray) -> np.ndarray:
    return np.clip(u.astype(float), CLIP, 1.0 - CLIP)


def brier_score(u: np.ndarray, e: np.ndarray) -> float:
    u, e = _clip_u(u), e.astype(float)
    return float(np.mean((u - e) ** 2))


def log_loss(u: np.ndarray, e: np.ndarray) -> float:
    u, e = _clip_u(u), e.astype(float)
    return float(-np.mean(e * np.log(u) + (1.0 - e) * np.log(1.0 - u)))


def auroc_score(u: np.ndarray, e: np.ndarray) -> float:
    u, e = u.astype(float), e.astype(int)
    n_pos = int(e.sum())
    n_neg = len(e) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(u)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(u) + 1)
    sum_ranks_pos = ranks[e == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc_score(u: np.ndarray, e: np.ndarray) -> float:
    u, e = u.astype(float), e.astype(int)
    if e.sum() == 0:
        return float("nan")
    order = np.argsort(-u)
    e_sorted = e[order]
    tp = np.cumsum(e_sorted)
    fp = np.cumsum(1 - e_sorted)
    prec = tp / np.maximum(tp + fp, 1)
    recall = tp / e.sum()
    # trapezoid on precision-recall
    recall = np.concatenate([[0.0], recall, [1.0]])
    prec = np.concatenate([[1.0], prec, [0.0]])
    return float(np.trapz(prec, recall))


def uce_score(u: np.ndarray, e: np.ndarray, n_bins: int = N_BINS) -> float:
    u, e = u.astype(float), e.astype(float)
    try:
        bins = pd.qcut(u, q=n_bins, duplicates="drop")
    except ValueError:
        return float("nan")
    grouped = pd.DataFrame({"u": u, "e": e, "bin": bins})
    n = len(u)
    total = 0.0
    for _, g in grouped.groupby("bin", observed=True):
        total += len(g) / n * abs(g["e"].mean() - g["u"].mean())
    return float(total)


def reliability_bins(u: np.ndarray, e: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    u, e = u.astype(float), e.astype(float)
    try:
        bins = pd.qcut(u, q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["mean_u", "mean_e", "count"])
    grouped = pd.DataFrame({"u": u, "e": e, "bin": bins})
    rows = []
    for _, g in grouped.groupby("bin", observed=True):
        rows.append({"mean_u": g["u"].mean(), "mean_e": g["e"].mean(), "count": len(g)})
    return pd.DataFrame(rows)


def risk_coverage_curve(u: np.ndarray, e: np.ndarray) -> pd.DataFrame:
    u, e = u.astype(float), e.astype(int)
    order = np.argsort(u)  # least uncertain first
    e_sorted = e[order]
    n = len(e)
    coverages = []
    risks = []
    for k in range(1, n + 1):
        coverages.append(k / n)
        risks.append(e_sorted[:k].mean())
    return pd.DataFrame({"coverage": coverages, "risk": risks})


def risk_at_coverage(u: np.ndarray, e: np.ndarray, coverage: float = 0.8) -> float:
    curve = risk_coverage_curve(u, e)
    idx = (curve["coverage"] - coverage).abs().idxmin()
    return float(curve.loc[idx, "risk"])


def roc_curve_points(u: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, e = u.astype(float), e.astype(int)
    thresholds = np.unique(u)[::-1]
    tprs, fprs = [0.0], [0.0]
    n_pos, n_neg = e.sum(), len(e) - e.sum()
    for t in thresholds:
        pred = (u >= t).astype(int)
        tp = ((pred == 1) & (e == 1)).sum()
        fp = ((pred == 1) & (e == 0)).sum()
        tprs.append(tp / n_pos if n_pos else 0.0)
        fprs.append(fp / n_neg if n_neg else 0.0)
    tprs.append(1.0)
    fprs.append(1.0)
    return np.array(fprs), np.array(tprs)


def compute_ablation_scores(df: pd.DataFrame) -> pd.DataFrame:
    f = df["F"].astype(float)
    s = df["S"].astype(float).fillna(0.0)
    a = df["A"].astype(float)
    c_flip = df["C_flip"].astype(float)

    out = df[["dialogue_id", "E_i"]].copy()
    out["U_F"] = f
    out["U_F_S"] = W_F * f + W_S * f * s
    out.loc[f == 0, "U_F_S"] = 0.0
    out["U_F_S_A"] = out["U_F_S"] + W_A * f * (1.0 - a)
    out.loc[f == 0, "U_F_S_A"] = 0.0
    out["U_F_S_A_C"] = (
        W_F * f + W_S * f * s + W_A * f * (1.0 - a) + W_C * f * (1.0 - c_flip)
    )
    out.loc[f == 0, "U_F_S_A_C"] = 0.0
    return out


def _metric_row(method: str, u: np.ndarray, e: np.ndarray) -> dict:
    return {
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "brier": brier_score(u, e),
        "log_loss": log_loss(u, e),
        "auroc": auroc_score(u, e),
        "auprc": auprc_score(u, e),
        "uce": uce_score(u, e),
        "risk_at_80": risk_at_coverage(u, e, 0.8),
        "n": len(e),
        "error_rate": float(e.mean()),
    }


def evaluate_methods(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    e = df["E_i"].to_numpy(dtype=int)
    rows = []
    for m in methods:
        if m not in df.columns:
            continue
        u = df[m].fillna(0.0).to_numpy(dtype=float)
        rows.append(_metric_row(m, u, e))
    return pd.DataFrame(rows)


def evaluate_ablation(ablation_df: pd.DataFrame) -> pd.DataFrame:
    e = ablation_df["E_i"].to_numpy(dtype=int)
    variants = [
        ("Flip only (F)", "U_F"),
        ("+ Speed (F·S)", "U_F_S"),
        ("+ Arg. quality (F·S·A)", "U_F_S_A"),
        ("+ Conf. at flip (full U^pers)", "U_F_S_A_C"),
    ]
    rows = []
    for label, col in variants:
        u = ablation_df[col].to_numpy(dtype=float)
        row = _metric_row(col, u, e)
        row["variant"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_bootstrap_brier(
    df: pd.DataFrame,
    composite_col: str,
    baseline_col: str,
    cluster_col: str = "original_index",
    n_boot: int = N_BOOT,
) -> dict:
    e = df["E_i"].to_numpy(dtype=int)
    u_comp = df[composite_col].fillna(0.0).to_numpy(dtype=float)
    u_base = df[baseline_col].fillna(0.0).to_numpy(dtype=float)
    clusters = df[cluster_col].to_numpy()
    unique_clusters = np.unique(clusters)

    deltas = []
    for _ in range(n_boot):
        sampled = RNG.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx_parts = [np.where(clusters == c)[0] for c in sampled]
        idx = np.concatenate(idx_parts) if idx_parts else np.array([], dtype=int)
        if len(idx) == 0:
            continue
        b_base = brier_score(u_base[idx], e[idx])
        b_comp = brier_score(u_comp[idx], e[idx])
        deltas.append(b_base - b_comp)

    deltas = np.array(deltas)
    return {
        "comparison": f"{METHOD_LABELS.get(composite_col, composite_col)} vs {METHOD_LABELS.get(baseline_col, baseline_col)}",
        "composite": composite_col,
        "baseline": baseline_col,
        "mean_delta": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_improve": float((deltas > 0).mean()),
        "n_boot": len(deltas),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate persuasion uncertainty scores.")
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    e = df["E_i"].to_numpy(dtype=int)
    out_csv = args.out_dir / "csv"
    out_png = args.out_dir / "png"
    out_csv.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)

    # ── Main metrics ──────────────────────────────────────────────────────
    metrics = evaluate_methods(df, METHODS)
    metrics.to_csv(out_csv / "uncertainty_metrics.csv", index=False)
    plot_metrics_bar(metrics, out_png / "metrics_comparison.png", "Step 3 — Method comparison (primary metrics)")

    # ── ROC curves ──────────────────────────────────────────────────────────
    roc_data = {}
    for m in METHODS:
        if m in df.columns:
            roc_data[m] = roc_curve_points(df[m].fillna(0.0).to_numpy(), e)
    plot_roc_curves(roc_data, out_png / "roc_curves.png")

    # ── Reliability diagrams ────────────────────────────────────────────────
    bin_stats = {}
    for m in METHODS:
        if m in df.columns:
            bin_stats[m] = reliability_bins(df[m].fillna(0.0).to_numpy(), e)
    plot_reliability(bin_stats, out_png / "reliability_diagrams.png")
    rel_df = pd.concat(
        [s.assign(method=m) for m, s in bin_stats.items()],
        ignore_index=True,
    )
    rel_df.to_csv(out_csv / "reliability_bins.csv", index=False)

    # ── Risk–coverage ───────────────────────────────────────────────────────
    rc_curves = {}
    rc_summary_rows = []
    for m in METHODS:
        if m not in df.columns:
            continue
        u = df[m].fillna(0.0).to_numpy(dtype=float)
        rc_curves[m] = risk_coverage_curve(u, e)
        rc_summary_rows.append({
            "method": m,
            "label": METHOD_LABELS.get(m, m),
            "risk_at_70": risk_at_coverage(u, e, 0.7),
            "risk_at_80": risk_at_coverage(u, e, 0.8),
            "risk_at_90": risk_at_coverage(u, e, 0.9),
        })
    plot_risk_coverage(rc_curves, out_png / "risk_coverage_curves.png")
    pd.DataFrame(rc_summary_rows).to_csv(out_csv / "risk_coverage_summary.csv", index=False)
    rc_df = pd.concat(
        [c.assign(method=m) for m, c in rc_curves.items()],
        ignore_index=True,
    )
    rc_df.to_csv(out_csv / "risk_coverage_curves.csv", index=False)

    # ── Ablation ────────────────────────────────────────────────────────────
    ablation_scores = compute_ablation_scores(df)
    ablation_metrics = evaluate_ablation(ablation_scores)
    ablation_metrics.to_csv(out_csv / "ablation_metrics.csv", index=False)
    plot_ablation(ablation_metrics, out_png / "ablation_comparison.png")

    # ── Bootstrap (U^pers vs baselines) ─────────────────────────────────────
    boot_rows = []
    for baseline in ("U_self", "U_marker", "U_flip"):
        if baseline in df.columns:
            boot_rows.append(cluster_bootstrap_brier(df, "U_pers", baseline, n_boot=args.n_boot))
    if "U_self" in df.columns:
        boot_rows.append(cluster_bootstrap_brier(df, "U_hybrid", "U_self", n_boot=args.n_boot))
    bootstrap = pd.DataFrame(boot_rows)
    bootstrap.to_csv(out_csv / "bootstrap_brier_ci.csv", index=False)
    if not bootstrap.empty:
        plot_bootstrap_ci(bootstrap, out_png / "bootstrap_brier_ci.png")

    # ── Summary text ──────────────────────────────────────────────────────
    lines = [
        "Persuasion uncertainty evaluation summary",
        f"n = {len(df)} dialogues, initial error rate = {e.mean():.3f}",
        "",
        "Primary metric (Brier, lower is better):",
    ]
    for _, r in metrics.sort_values("brier").iterrows():
        lines.append(f"  {r['label']:30s}  Brier={r['brier']:.4f}  AUROC={r['auroc']:.3f}  UCE={r['uce']:.4f}")
    lines += ["", "Ablation (Brier):"]
    for _, r in ablation_metrics.iterrows():
        lines.append(f"  {r['variant']:35s}  Brier={r['brier']:.4f}  AUROC={r['auroc']:.3f}")
    if not bootstrap.empty:
        lines += ["", "Bootstrap Brier improvement (positive = composite better):"]
        for _, r in bootstrap.iterrows():
            lines.append(
                f"  {r['comparison']:50s}  Δ={r['mean_delta']:+.4f}  "
                f"95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  P(Δ>0)={r['p_improve']:.3f}"
            )
    summary_path = args.out_dir / "evaluation_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved metrics:    {out_csv / 'uncertainty_metrics.csv'}")
    print(f"Saved figures:    {out_png}")
    print(f"Saved summary:    {summary_path}")
    print()
    print("\n".join(lines[3:]))


if __name__ == "__main__":
    main()
