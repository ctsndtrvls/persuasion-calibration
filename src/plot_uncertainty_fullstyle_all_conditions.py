"""
Full-style metrics_comparison / ablation_comparison grids for all models × datasets.

Matches the horizontal-bar look of per-condition
  {Model}/{dataset}/uncertainty_scores/figures/metrics_comparison.png
  {Model}/{dataset}/uncertainty_scores/figures/ablation_comparison.png

Outputs (thesis-ready):
  png/uncertainty_metrics_comparison_all_models_all_datasets.png
  png/uncertainty_ablation_comparison_all_models_all_datasets.png
  png/uncertainty_metrics_comparison_{fever,popqa,debateqa}.png   (4 models × 4 metrics)
  png/uncertainty_ablation_comparison_{fever,popqa,debateqa}.png  (4 models × 3 metrics)

Example:
  python3 plot_uncertainty_fullstyle_all_conditions.py
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from persuasion_uncertainty_viz import BG, METHOD_COLORS, METHOD_LABELS  # noqa: E402

ROOT = _PROJECT_ROOT
PERSUASION = ROOT / "output_wood" / "persuasion"
OUT_PNG = PERSUASION / "png"
OUT_CSV = PERSUASION / "csv"

MODEL_ORDER = ["GPT-4o", "DeepSeek", "Gemma", "Qwen"]
MODEL_DISPLAY = {"GPT-4o": "GPT", "DeepSeek": "DeepSeek", "Gemma": "Gemma", "Qwen": "Qwen"}
DATASETS = [("fever", "FEVER"), ("popqa", "PopQA"), ("debateqa", "DebateQA")]

METRICS_METHODS = ["U_self", "U_marker", "U_flip", "U_pers", "U_hybrid"]
METRIC_SPECS = [
    ("brier", "Brier ↓", True),
    ("auroc", "AUROC ↑", False),
    ("auprc", "AUPRC ↑", False),
    ("uce", "UCE ↓", True),
]
ABLATION_ORDER = [
    "Flip only (F)",
    "+ Speed (F·S)",
    "+ Arg. quality (F·S·A)",
    "+ Conf. at flip (full U^pers)",
]
ABLATION_SPECS = [
    ("brier", "Brier ↓", True),
    ("auroc", "AUROC ↑", False),
    ("uce", "UCE ↓", True),
]
SHORT_METHOD = {
    "Self-report (1−C₀)": "Self-report",
    "Epistemic markers": "Markers",
    "Flip only": "Flip only",
    "Persuasion (U^pers)": r"$U^{\mathrm{pers}}$",
    "Hybrid (U^hybrid)": "Hybrid",
}
SHORT_ABLATION = {
    "Flip only (F)": "Flip only (F)",
    "+ Speed (F·S)": "+ Speed (F·S)",
    "+ Arg. quality (F·S·A)": "+ Arg. quality",
    "+ Conf. at flip (full U^pers)": r"Full $U^{\mathrm{pers}}$",
}


def load_metrics(model: str, dataset_key: str) -> pd.DataFrame | None:
    path = PERSUASION / model / dataset_key / "uncertainty_scores" / "csv" / "uncertainty_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_ablation(model: str, dataset_key: str) -> pd.DataFrame | None:
    path = PERSUASION / model / dataset_key / "uncertainty_scores" / "csv" / "ablation_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _barh_methods(ax: plt.Axes, df: pd.DataFrame, col: str, *, lower_better: bool, show_ylabels: bool) -> None:
    present = [m for m in METRICS_METHODS if m in set(df["method"])]
    sub = df.set_index("method").loc[present].reset_index()
    sub = sub.sort_values(col, ascending=lower_better)
    labels = [SHORT_METHOD.get(METHOD_LABELS.get(m, m), METHOD_LABELS.get(m, m)) for m in sub["method"]]
    colors = [METHOD_COLORS.get(m, "#64748B") for m in sub["method"]]
    ax.barh(labels, sub[col].to_numpy(dtype=float), color=colors, edgecolor="white", linewidth=0.3)
    ax.set_xlim(left=0)
    if not show_ylabels:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)


def _barh_ablation(ax: plt.Axes, df: pd.DataFrame, col: str, *, lower_better: bool, show_ylabels: bool) -> None:
    sub = df.set_index("variant").loc[ABLATION_ORDER].reset_index()
    sub = sub.sort_values(col, ascending=lower_better)
    labels = [SHORT_ABLATION.get(v, v) for v in sub["variant"]]
    ax.barh(labels, sub[col].to_numpy(dtype=float), color="#3B82F6", edgecolor="white", linewidth=0.3)
    ax.set_xlim(left=0)
    if not show_ylabels:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)


def plot_metrics_dataset_page(dataset_key: str, dataset_label: str, out_path: Path) -> None:
    """4 models (rows) × 4 metrics (cols), original horizontal-bar style."""
    fig, axes = plt.subplots(4, 4, figsize=(15.5, 11.5), sharex=False)
    fig.patch.set_facecolor(BG)
    ns = []
    for r, model in enumerate(MODEL_ORDER):
        df = load_metrics(model, dataset_key)
        if df is None:
            for c in range(4):
                axes[r, c].axis("off")
            continue
        ns.append(int(df["n"].iloc[0]))
        err = float(df["error_rate"].iloc[0])
        for c, (col, title, lower) in enumerate(METRIC_SPECS):
            ax = axes[r, c]
            ax.set_facecolor(BG)
            _barh_methods(ax, df, col, lower_better=lower, show_ylabels=(c == 0))
            if r == 0:
                ax.set_title(title, fontsize=10, pad=4)
            if c == 0:
                n = int(df["n"].iloc[0])
                ax.set_ylabel(
                    f"{MODEL_DISPLAY[model]}\nn={n}, err={err:.0%}",
                    fontsize=9,
                    rotation=0,
                    labelpad=42,
                    va="center",
                )
    n_note = ", ".join(f"{MODEL_DISPLAY[m]}={n}" for m, n in zip(MODEL_ORDER, ns)) if ns else ""
    fig.suptitle(
        f"Method comparison — {dataset_label} (Brier primary)",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    fig.text(0.5, 0.005, f"Sample sizes: {n_note}", ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_ablation_dataset_page(dataset_key: str, dataset_label: str, out_path: Path) -> None:
    """4 models (rows) × 3 metrics (cols), original ablation horizontal-bar style."""
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 11.5), sharex=False)
    fig.patch.set_facecolor(BG)
    for r, model in enumerate(MODEL_ORDER):
        df = load_ablation(model, dataset_key)
        if df is None:
            for c in range(3):
                axes[r, c].axis("off")
            continue
        n = int(df["n"].iloc[0])
        err = float(df["error_rate"].iloc[0])
        for c, (col, title, lower) in enumerate(ABLATION_SPECS):
            ax = axes[r, c]
            ax.set_facecolor(BG)
            _barh_ablation(ax, df, col, lower_better=lower, show_ylabels=(c == 0))
            if r == 0:
                ax.set_title(title, fontsize=10, pad=4)
            if c == 0:
                ax.set_ylabel(
                    f"{MODEL_DISPLAY[model]}\nn={n}, err={err:.0%}",
                    fontsize=9,
                    rotation=0,
                    labelpad=42,
                    va="center",
                )
    fig.suptitle(
        f"Ablation — {dataset_label} (F → +S → +A → full $U^{{\\mathrm{{pers}}}}$)",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.01, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_metrics_combined(out_path: Path) -> None:
    """All datasets stacked: 12 model-rows × 4 metrics."""
    n_rows = len(MODEL_ORDER) * len(DATASETS)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15.5, 3.0 * n_rows), sharex=False)
    fig.patch.set_facecolor(BG)
    row = 0
    for dataset_key, dataset_label in DATASETS:
        for model in MODEL_ORDER:
            df = load_metrics(model, dataset_key)
            for c, (col, title, lower) in enumerate(METRIC_SPECS):
                ax = axes[row, c]
                ax.set_facecolor(BG)
                if df is None:
                    ax.axis("off")
                    continue
                _barh_methods(ax, df, col, lower_better=lower, show_ylabels=(c == 0))
                if row == 0:
                    ax.set_title(title, fontsize=10, pad=4)
                if c == 0:
                    n = int(df["n"].iloc[0])
                    err = float(df["error_rate"].iloc[0])
                    ax.set_ylabel(
                        f"{dataset_label} · {MODEL_DISPLAY[model]}\nn={n}, err={err:.0%}",
                        fontsize=8,
                        rotation=0,
                        labelpad=58,
                        va="center",
                    )
            row += 1
    fig.suptitle(
        "Method comparison — all models and datasets (Brier primary)",
        fontsize=14,
        weight="bold",
        y=0.998,
    )
    fig.tight_layout(rect=[0.03, 0.01, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_ablation_combined(out_path: Path) -> None:
    n_rows = len(MODEL_ORDER) * len(DATASETS)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13.5, 3.0 * n_rows), sharex=False)
    fig.patch.set_facecolor(BG)
    row = 0
    for dataset_key, dataset_label in DATASETS:
        for model in MODEL_ORDER:
            df = load_ablation(model, dataset_key)
            for c, (col, title, lower) in enumerate(ABLATION_SPECS):
                ax = axes[row, c]
                ax.set_facecolor(BG)
                if df is None:
                    ax.axis("off")
                    continue
                _barh_ablation(ax, df, col, lower_better=lower, show_ylabels=(c == 0))
                if row == 0:
                    ax.set_title(title, fontsize=10, pad=4)
                if c == 0:
                    n = int(df["n"].iloc[0])
                    err = float(df["error_rate"].iloc[0])
                    ax.set_ylabel(
                        f"{dataset_label} · {MODEL_DISPLAY[model]}\nn={n}, err={err:.0%}",
                        fontsize=8,
                        rotation=0,
                        labelpad=58,
                        va="center",
                    )
            row += 1
    fig.suptitle(
        r"Ablation of $U^{\mathrm{pers}}$ — all models and datasets",
        fontsize=14,
        weight="bold",
        y=0.998,
    )
    fig.tight_layout(rect=[0.03, 0.01, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_summary_csv() -> None:
    rows = []
    for model in MODEL_ORDER:
        for key, label in DATASETS:
            df = load_metrics(model, key)
            if df is None:
                continue
            best = df.loc[df["brier"].idxmin()]
            rows.append(
                {
                    "model": MODEL_DISPLAY[model],
                    "dataset": label,
                    "n": int(df["n"].iloc[0]),
                    "error_rate": round(float(df["error_rate"].iloc[0]), 3),
                    "best_method": best["label"],
                    "best_brier": round(float(best["brier"]), 4),
                    "U_self_brier": round(float(df.loc[df["method"] == "U_self", "brier"].iloc[0]), 4),
                    "U_pers_brier": round(float(df.loc[df["method"] == "U_pers", "brier"].iloc[0]), 4),
                    "U_hybrid_brier": round(float(df.loc[df["method"] == "U_hybrid", "brier"].iloc[0]), 4),
                    "U_flip_brier": round(float(df.loc[df["method"] == "U_flip", "brier"].iloc[0]), 4),
                }
            )
    out = OUT_CSV / "uncertainty_chapter_summary.csv"
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


def main() -> None:
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    for key, label in DATASETS:
        plot_metrics_dataset_page(
            key, label, OUT_PNG / f"uncertainty_metrics_comparison_{key}_all_models.png"
        )
        plot_ablation_dataset_page(
            key, label, OUT_PNG / f"uncertainty_ablation_comparison_{key}_all_models.png"
        )
    plot_metrics_combined(OUT_PNG / "uncertainty_metrics_comparison_all_models_all_datasets.png")
    plot_ablation_combined(OUT_PNG / "uncertainty_ablation_comparison_all_models_all_datasets.png")
    write_summary_csv()


if __name__ == "__main__":
    main()
