"""
Significance panels for metrics_comparison and ablation_comparison (by dataset).

Computes paired cluster-bootstrap 95% CIs on ΔBrier for:
  - method pairs (metrics page)
  - consecutive ablation steps (ablation page)
Plus between-model CI-overlap matrices.

Outputs under output_wood/persuasion/uncertainty_scores_compare/:
  metrics_significance_{fever,popqa,debateqa}.png
  ablation_significance_{fever,popqa,debateqa}.png
  csv/metrics_bootstrap_{dataset}.csv
  csv/ablation_bootstrap_{dataset}.csv

Example:
  cd src && python3 plot_metrics_ablation_significance.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_persuasion_uncertainty import (
    brier_score,
    compute_ablation_scores,
)
from persuasion_uncertainty_viz import BG

PERSUASION_ROOT = _PROJECT_ROOT / "output_wood" / "persuasion"
OUT_DIR = PERSUASION_ROOT / "uncertainty_scores_compare"

DATASETS = ("fever", "popqa", "debateqa")
DATASET_TITLES = {"fever": "FEVER", "popqa": "PopQA", "debateqa": "DebateQA"}
MODEL_ORDER = ("GPT-4o", "DeepSeek", "Gemma", "Qwen")
MODEL_LABELS = {
    "GPT-4o": "GPT",
    "DeepSeek": "DeepSeek",
    "Gemma": "Gemma",
    "Qwen": "Qwen",
}
DISPLAY_MODELS = [MODEL_LABELS[m] for m in MODEL_ORDER]

N_BOOT = 2000
RNG = np.random.default_rng(42)

# Metrics page: paired method comparisons (Δ = Brier(base) − Brier(comp))
METRICS_PAIRS = [
    ("U_hybrid", "U_self", "Hybrid vs Self"),
    ("U_pers", "U_self", "U^pers vs Self"),
    ("U_flip", "U_self", "Flip vs Self"),
    ("U_hybrid", "U_flip", "Hybrid vs Flip"),
    ("U_pers", "U_flip", "U^pers vs Flip"),
    ("U_self", "U_marker", "Self vs Markers"),
]

# Ablation: consecutive / overall (Δ = Brier(earlier) − Brier(later); + = later better)
ABLATION_PAIRS = [
    ("U_F", "U_F_S", "Flip → +Speed"),
    ("U_F_S", "U_F_S_A", "+Speed → +Arg.Q"),
    ("U_F_S_A", "U_F_S_A_C", "+Arg.Q → Full"),
    ("U_F", "U_F_S_A_C", "Flip → Full U^pers"),
]

STATUS_COLOR = {
    "better": "#10B981",
    "worse": "#EF4444",
    "ns": "#94A3B8",
}


def _status(lo: float, hi: float) -> str:
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    return "ns"


def paired_cohen_d(e: np.ndarray, u_comp: np.ndarray, u_base: np.ndarray) -> float:
    """Cohen's d on paired per-dialogue squared-error differences.

    d_i = (u_base_i − E_i)² − (u_comp_i − E_i)²; positive mean ⇒ composite better.
    """
    se_base = (u_base - e.astype(float)) ** 2
    se_comp = (u_comp - e.astype(float)) ** 2
    diff = se_base - se_comp
    sd = float(diff.std(ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan")
    return float(diff.mean() / sd)


def cluster_bootstrap_delta(
    e: np.ndarray,
    u_comp: np.ndarray,
    u_base: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float, float, float, float]:
    """Δ = Brier(base) − Brier(comp); positive = composite better.

    Returns mean_delta, ci_low, ci_high, p_improve (= P(Δ>0)),
    p_two_sided (= 2·min(P(Δ>0), P(Δ≤0))), and paired Cohen's d.
    """
    unique = np.unique(clusters)
    # Pre-index rows per cluster for speed
    idx_map = {c: np.where(clusters == c)[0] for c in unique}
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = RNG.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_map[c] for c in sampled])
        b_base = brier_score(u_base[idx], e[idx])
        b_comp = brier_score(u_comp[idx], e[idx])
        deltas[i] = b_base - b_comp
    p_improve = float((deltas > 0).mean())
    p_two_sided = float(min(1.0, 2.0 * min(p_improve, 1.0 - p_improve)))
    return (
        float(deltas.mean()),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
        p_improve,
        p_two_sided,
        paired_cohen_d(e, u_comp, u_base),
    )


def load_scores(model: str, dataset: str) -> pd.DataFrame | None:
    path = (
        PERSUASION_ROOT
        / model
        / dataset
        / "uncertainty_scores"
        / "csv"
        / "persuasion_uncertainty_scores.csv"
    )
    if not path.exists():
        return None
    return pd.read_csv(path)


def bootstrap_metrics_for_condition(model: str, dataset: str, n_boot: int) -> list[dict]:
    df = load_scores(model, dataset)
    if df is None:
        return []
    e = df["E_i"].to_numpy(dtype=int)
    clusters = df["original_index"].to_numpy()
    rows = []
    for comp, base, label in METRICS_PAIRS:
        if comp not in df.columns or base not in df.columns:
            continue
        u_comp = df[comp].fillna(0.0).to_numpy(dtype=float)
        u_base = df[base].fillna(0.0).to_numpy(dtype=float)
        mu, lo, hi, p_imp, p2, d = cluster_bootstrap_delta(
            e, u_comp, u_base, clusters, n_boot=n_boot
        )
        rows.append(
            {
                "model": model,
                "display": MODEL_LABELS[model],
                "dataset": dataset,
                "comparison": label,
                "composite": comp,
                "baseline": base,
                "mean_delta": mu,
                "ci_low": lo,
                "ci_high": hi,
                "p_improve": p_imp,
                "p_two_sided": p2,
                "cohen_d": d,
                "status": _status(lo, hi),
            }
        )
    return rows


def bootstrap_ablation_for_condition(model: str, dataset: str, n_boot: int) -> list[dict]:
    df = load_scores(model, dataset)
    if df is None:
        return []
    need = {"F", "S", "A", "C_flip", "E_i", "original_index"}
    if not need.issubset(df.columns):
        return []
    abl = compute_ablation_scores(df)
    abl["original_index"] = df["original_index"].to_numpy()
    e = abl["E_i"].to_numpy(dtype=int)
    clusters = abl["original_index"].to_numpy()
    rows = []
    for earlier, later, label in ABLATION_PAIRS:
        # Δ = Brier(earlier) − Brier(later); later is "composite"
        u_comp = abl[later].to_numpy(dtype=float)
        u_base = abl[earlier].to_numpy(dtype=float)
        mu, lo, hi, p_imp, p2, d = cluster_bootstrap_delta(
            e, u_comp, u_base, clusters, n_boot=n_boot
        )
        rows.append(
            {
                "model": model,
                "display": MODEL_LABELS[model],
                "dataset": dataset,
                "comparison": label,
                "composite": later,
                "baseline": earlier,
                "mean_delta": mu,
                "ci_low": lo,
                "ci_high": hi,
                "p_improve": p_imp,
                "p_two_sided": p2,
                "cohen_d": d,
                "status": _status(lo, hi),
            }
        )
    return rows


def pairwise_overlap(df: pd.DataFrame, comparison: str) -> dict[str, dict[str, str]]:
    """Between-model CI overlap for one comparison label."""
    sub = df[df["comparison"] == comparison].set_index("display")
    mat: dict[str, dict[str, str]] = {}
    for a in DISPLAY_MODELS:
        mat[a] = {}
        for b in DISPLAY_MODELS:
            if a == b:
                mat[a][b] = "—"
            elif a not in sub.index or b not in sub.index:
                mat[a][b] = "?"
            else:
                A, B = sub.loc[a], sub.loc[b]
                overlap = not (A["ci_high"] < B["ci_low"] or B["ci_high"] < A["ci_low"])
                mat[a][b] = "n.s." if overlap else "sig"
    return mat


def draw_significance_page(
    *,
    title: str,
    within_df: pd.DataFrame,
    comparisons: list[str],
    matrix_a_label: str,
    matrix_b_label: str,
    out_path: Path,
    panel_a_note: str,
) -> None:
    fig = plt.figure(figsize=(12.5, 8.4))
    fig.patch.set_facecolor(BG)
    fig.suptitle(title, fontsize=13, weight="bold", y=0.985)

    # Keep panel A lower to avoid overlap with two-line suptitle.
    ax1 = fig.add_axes([0.05, 0.37, 0.90, 0.49])
    ax1.set_facecolor(BG)
    ax1.set_title("A. Within-model paired ΔBrier (95% CI)", loc="left", fontsize=11, weight="bold", pad=2)
    ax1.set_xlim(0, 5.2)
    n_rows = len(comparisons)
    ax1.set_ylim(0, n_rows + 1.8)
    ax1.invert_yaxis()
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    col_w = [1.55] + [0.88] * 4
    x_pos = [0.05]
    for w in col_w[:-1]:
        x_pos.append(x_pos[-1] + w)

    headers = ["Comparison"] + DISPLAY_MODELS
    for j, h in enumerate(headers):
        ax1.text(x_pos[j] + col_w[j] / 2, 0.35, h, ha="center", va="center", fontsize=9, weight="bold")

    row_h = 0.95 if n_rows <= 4 else 0.78
    for i, lab in enumerate(comparisons):
        y = 0.35 + (i + 1) * row_h
        ax1.text(x_pos[0] + 0.04, y, lab, ha="left", va="center", fontsize=8.5)
        for j, m in enumerate(DISPLAY_MODELS):
            r = within_df[(within_df["display"] == m) & (within_df["comparison"] == lab)]
            if r.empty:
                continue
            row = r.iloc[0]
            st = row["status"]
            mu, lo, hi = float(row["mean_delta"]), float(row["ci_low"]), float(row["ci_high"])
            x, w = x_pos[j + 1], col_w[j + 1]
            rect = mpatches.FancyBboxPatch(
                (x + 0.04, y - 0.30),
                w - 0.08,
                0.60,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=STATUS_COLOR[st],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.88,
            )
            ax1.add_patch(rect)
            sign = "+" if mu >= 0 else ""
            ax1.text(
                x + w / 2,
                y,
                f"{sign}{mu:.3f}\n[{lo:+.2f},{hi:+.2f}]",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white",
                weight="bold",
            )

    ax1.text(
        0.05,
        n_rows * row_h + 1.35,
        "Green = significant improvement (CI>0)  ·  Red = significant worsening (CI<0)  ·  Grey = n.s. (CI crosses 0)",
        fontsize=8,
        color="#475569",
    )
    ax1.text(0.05, n_rows * row_h + 1.65, panel_a_note, fontsize=7.5, color="#64748B")

    def draw_matrix(ax, comparison: str, subtitle: str) -> None:
        ax.set_facecolor(BG)
        ax.set_title(subtitle, loc="left", fontsize=10, weight="bold")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        mat = pairwise_overlap(within_df, comparison)
        for j, m in enumerate(DISPLAY_MODELS):
            ax.text(1.15 + j * 0.95, 0.4, m, ha="center", va="center", fontsize=8, weight="bold")
        for i, a in enumerate(DISPLAY_MODELS):
            ax.text(0.12, 1.15 + i * 0.9, a, ha="left", va="center", fontsize=8, weight="bold")
            for j, b in enumerate(DISPLAY_MODELS):
                x, y = 0.72 + j * 0.95, 0.8 + i * 0.9
                v = mat[a][b]
                if v == "—":
                    color, txt = "#F1F5F9", "—"
                elif v == "sig":
                    color, txt = "#F59E0B", "sig"
                else:
                    color, txt = "#CBD5E1", "n.s."
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (x, y),
                        0.85,
                        0.7,
                        boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.6,
                    )
                )
                ax.text(x + 0.425, y + 0.35, txt, ha="center", va="center", fontsize=8, weight="bold")

    ax2 = fig.add_axes([0.06, 0.05, 0.42, 0.30])
    draw_matrix(ax2, matrix_a_label, f"B. Between models — {matrix_a_label}")
    ax3 = fig.add_axes([0.54, 0.05, 0.42, 0.30])
    draw_matrix(ax3, matrix_b_label, f"C. Between models — {matrix_b_label}")

    fig.text(
        0.5,
        0.01,
        "Between-model: orange = 95% CIs on ΔBrier do not overlap; grey = overlap. Heuristic, not a formal cross-model test.",
        ha="center",
        fontsize=7.5,
        color="#64748B",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Metrics/ablation significance panels.")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    out = args.out_dir
    csv_dir = out / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[pd.DataFrame] = []
    for dataset in DATASETS:
        print(f"\n=== {dataset} (n_boot={args.n_boot}) ===")
        metrics_rows: list[dict] = []
        abl_rows: list[dict] = []
        for model in MODEL_ORDER:
            print(f"  bootstrap {MODEL_LABELS[model]} …")
            metrics_rows.extend(bootstrap_metrics_for_condition(model, dataset, args.n_boot))
            abl_rows.extend(bootstrap_ablation_for_condition(model, dataset, args.n_boot))

        mdf = pd.DataFrame(metrics_rows)
        adf = pd.DataFrame(abl_rows)
        mdf.to_csv(csv_dir / f"metrics_bootstrap_{dataset}.csv", index=False)
        adf.to_csv(csv_dir / f"ablation_bootstrap_{dataset}.csv", index=False)

        draw_significance_page(
            title=(
                f"Metrics significance — {DATASET_TITLES[dataset]}\n"
                f"Paired cluster-bootstrap 95% CI on ΔBrier  ·  GPT, DeepSeek, Gemma, Qwen"
            ),
            within_df=mdf,
            comparisons=[c[2] for c in METRICS_PAIRS],
            matrix_a_label="Hybrid vs Self",
            matrix_b_label="Flip vs Self",
            out_path=out / f"metrics_significance_{dataset}.png",
            panel_a_note="Δ = Brier(baseline) − Brier(method). Matches differences visible on metrics_comparison_*.png.",
        )
        draw_significance_page(
            title=(
                f"Ablation significance — {DATASET_TITLES[dataset]}\n"
                f"Paired cluster-bootstrap 95% CI on ΔBrier  ·  GPT, DeepSeek, Gemma, Qwen"
            ),
            within_df=adf,
            comparisons=[c[2] for c in ABLATION_PAIRS],
            matrix_a_label="Flip → +Speed",
            matrix_b_label="Flip → Full U^pers",
            out_path=out / f"ablation_significance_{dataset}.png",
            panel_a_note="Δ = Brier(earlier step) − Brier(later step). Positive = adding the component improves Brier.",
        )
        all_metrics.append(mdf)

    # Compact summary: key pairwise comparisons with ΔBrier, CI, p, Cohen's d
    summary = pd.concat(all_metrics, ignore_index=True)
    key = ["Hybrid vs Self", "U^pers vs Self", "Flip vs Self"]
    summary = summary[summary["comparison"].isin(key)].copy()
    summary_cols = [
        "dataset",
        "display",
        "comparison",
        "mean_delta",
        "ci_low",
        "ci_high",
        "p_two_sided",
        "cohen_d",
        "status",
    ]
    out_csv = csv_dir / "significance_summary.csv"
    summary[summary_cols].to_csv(out_csv, index=False)
    print(f"\nSignificance summary → {out_csv}")
    print(f"All significance pages → {out}")


if __name__ == "__main__":
    main()
