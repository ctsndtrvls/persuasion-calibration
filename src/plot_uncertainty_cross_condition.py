"""
Cross-model comparison pages for ablation, metrics, and bootstrap (by dataset).

For each dataset writes one page with all available models side-by-side:
  output_wood/persuasion/uncertainty_scores_compare/
    ablation_comparison_{fever,popqa,debateqa}.png
    metrics_comparison_{fever,popqa,debateqa}.png
    bootstrap_brier_ci_{fever,popqa,debateqa}.png

Example:
  cd src && python3 plot_uncertainty_cross_condition.py
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

from persuasion_uncertainty_viz import BG, METHOD_LABELS

PERSUASION_ROOT = _PROJECT_ROOT / "output_wood" / "persuasion"
OUT_DIR = PERSUASION_ROOT / "uncertainty_scores_compare"

DATASETS = ("fever", "popqa", "debateqa")
DATASET_TITLES = {"fever": "FEVER", "popqa": "PopQA", "debateqa": "DebateQA"}
# Internal folder names → display order/labels on compare pages
MODEL_ORDER = ("GPT-4o", "DeepSeek", "Gemma", "Qwen")
MODEL_LABELS = {
    "GPT-4o": "GPT",
    "DeepSeek": "DeepSeek",
    "Gemma": "Gemma",
    "Qwen": "Qwen",
}
MODEL_COLORS = {
    "GPT-4o": "#2563EB",
    "DeepSeek": "#0F766E",
    "Gemma": "#D97706",
    "Qwen": "#7C3AED",
}


def model_display(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def models_display_list(models: list[str]) -> str:
    return ", ".join(model_display(m) for m in models)

ABLATION_ORDER = [
    "Flip only (F)",
    "+ Speed (F·S)",
    "+ Arg. quality (F·S·A)",
    "+ Conf. at flip (full U^pers)",
]
ABLATION_SHORT = {
    "Flip only (F)": "Flip only",
    "+ Speed (F·S)": "+ Speed",
    "+ Arg. quality (F·S·A)": "+ Arg. quality",
    "+ Conf. at flip (full U^pers)": "Full U^pers",
}

# Shared methods across models (exclude U_token — DeepSeek/FEVER only)
METRICS_METHODS = ["U_self", "U_marker", "U_flip", "U_pers", "U_hybrid"]

# Canonical bootstrap comparisons (shared across models)
BOOTSTRAP_KEYS = [
    ("U_pers", "U_self", "U^pers vs Self"),
    ("U_pers", "U_marker", "U^pers vs Markers"),
    ("U_pers", "U_flip", "U^pers vs Flip"),
    ("U_hybrid", "U_self", "Hybrid vs Self"),
]


def models_for_dataset(dataset: str) -> list[str]:
    found = []
    for model in MODEL_ORDER:
        csv_dir = PERSUASION_ROOT / model / dataset / "uncertainty_scores" / "csv"
        if (csv_dir / "ablation_metrics.csv").exists() and (csv_dir / "uncertainty_metrics.csv").exists():
            found.append(model)
    return found


def load_ablation(dataset: str) -> pd.DataFrame:
    rows = []
    for model in models_for_dataset(dataset):
        path = PERSUASION_ROOT / model / dataset / "uncertainty_scores" / "csv" / "ablation_metrics.csv"
        df = pd.read_csv(path)
        df["model"] = model
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_metrics(dataset: str) -> pd.DataFrame:
    rows = []
    for model in models_for_dataset(dataset):
        path = PERSUASION_ROOT / model / dataset / "uncertainty_scores" / "csv" / "uncertainty_metrics.csv"
        df = pd.read_csv(path)
        df["model"] = model
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_bootstrap(dataset: str) -> pd.DataFrame:
    rows = []
    for model in models_for_dataset(dataset):
        path = (
            PERSUASION_ROOT
            / model
            / dataset
            / "uncertainty_scores"
            / "csv"
            / "bootstrap_brier_ci.csv"
        )
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["model"] = model
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _grouped_bars(
    ax: plt.Axes,
    *,
    categories: list[str],
    models: list[str],
    values: dict[str, list[float]],
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(categories))
    n = len(models)
    width = min(0.8 / max(n, 1), 0.22)
    offsets = (np.arange(n) - (n - 1) / 2) * width
    for i, model in enumerate(models):
        ax.bar(
            x + offsets[i],
            values[model],
            width,
            label=model_display(model),
            color=MODEL_COLORS.get(model, "#64748B"),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.set_facecolor(BG)


def plot_ablation_page(dataset: str, out_path: Path) -> None:
    df = load_ablation(dataset)
    if df.empty:
        print(f"SKIP ablation {dataset}: no data")
        return
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    short_cats = [ABLATION_SHORT[v] for v in ABLATION_ORDER]

    specs = [
        ("brier", "Brier ↓", True),
        ("auroc", "AUROC ↑", False),
        ("uce", "UCE ↓", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    fig.patch.set_facecolor(BG)
    title = (
        f"Ablation comparison — {DATASET_TITLES[dataset]} "
        f"({models_display_list(models)})"
    )
    fig.suptitle(title, fontsize=13, weight="bold")

    for ax, (col, label, _lower) in zip(axes, specs):
        values = {}
        for model in models:
            sub = df[df["model"] == model].set_index("variant")
            values[model] = [float(sub.loc[v, col]) for v in ABLATION_ORDER]
        _grouped_bars(
            ax,
            categories=short_cats,
            models=models,
            values=values,
            ylabel=col.upper() if col != "brier" else "Brier",
            title=label,
        )
    axes[0].legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_metrics_page(dataset: str, out_path: Path) -> None:
    df = load_metrics(dataset)
    if df.empty:
        print(f"SKIP metrics {dataset}: no data")
        return
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    methods = [m for m in METRICS_METHODS if m in set(df["method"])]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    # Shorten for axis
    short = {
        "Self-report (1−C₀)": "Self-report",
        "Epistemic markers": "Markers",
        "Flip only": "Flip only",
        "Persuasion (U^pers)": "U^pers",
        "Hybrid (U^hybrid)": "Hybrid",
    }
    cats = [short.get(l, l) for l in labels]

    specs = [
        ("brier", "Brier ↓"),
        ("auroc", "AUROC ↑"),
        ("auprc", "AUPRC ↑"),
        ("uce", "UCE ↓"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.8))
    fig.patch.set_facecolor(BG)
    title = (
        f"Method comparison — {DATASET_TITLES[dataset]} "
        f"({models_display_list(models)}; Brier primary)"
    )
    fig.suptitle(title, fontsize=13, weight="bold")

    for ax, (col, label) in zip(axes, specs):
        values = {}
        for model in models:
            sub = df[df["model"] == model].set_index("method")
            values[model] = [float(sub.loc[m, col]) for m in methods]
        _grouped_bars(
            ax,
            categories=cats,
            models=models,
            values=values,
            ylabel=col.upper() if col != "brier" else "Brier",
            title=label,
        )
        # Optional: color tick labels not needed; legend on first
    axes[0].legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_bootstrap_page(dataset: str, out_path: Path) -> None:
    """Paired ΔBrier with 95% CI; models grouped within each comparison."""
    df = load_bootstrap(dataset)
    if df.empty:
        print(f"SKIP bootstrap {dataset}: no data")
        return
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    # Keep only the canonical comparisons that exist for at least one model
    keys = []
    for comp, base, label in BOOTSTRAP_KEYS:
        if ((df["composite"] == comp) & (df["baseline"] == base)).any():
            keys.append((comp, base, label))
    if not keys:
        print(f"SKIP bootstrap {dataset}: no matching comparisons")
        return

    n_comp = len(keys)
    n_models = len(models)
    fig_h = max(4.5, 1.1 + 0.55 * n_comp * (0.35 + 0.2 * n_models))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    title = (
        f"Bootstrap ΔBrier (95% CI) — {DATASET_TITLES[dataset]} "
        f"({models_display_list(models)})"
    )
    ax.set_title(title, fontsize=13, weight="bold")

    # Layout: each comparison is a block; within block, one bar per model
    block = 1.0
    bar_h = min(0.22, 0.7 / max(n_models, 1))
    y_ticks = []
    y_labels = []

    for i, (comp, base, label) in enumerate(keys):
        y0 = (n_comp - 1 - i) * block
        y_ticks.append(y0)
        y_labels.append(label)
        for j, model in enumerate(models):
            sub = df[
                (df["model"] == model)
                & (df["composite"] == comp)
                & (df["baseline"] == base)
            ]
            if sub.empty:
                continue
            r = sub.iloc[0]
            mean = float(r["mean_delta"])
            lo = float(r["ci_low"])
            hi = float(r["ci_high"])
            y = y0 + (j - (n_models - 1) / 2) * bar_h
            ax.barh(
                y,
                mean,
                height=bar_h * 0.9,
                color=MODEL_COLORS.get(model, "#64748B"),
                edgecolor="white",
                linewidth=0.4,
                label=model_display(model) if i == 0 else None,
                xerr=[[mean - lo], [hi - mean]],
                error_kw={"ecolor": "#334155", "capsize": 3, "lw": 1.0},
            )

    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Δ Brier = Brier(baseline) − Brier(composite); positive = composite better")
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-model ablation/metrics/bootstrap pages by dataset."
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        models = models_for_dataset(dataset)
        print(f"{dataset}: models={models}")
        plot_ablation_page(dataset, out / f"ablation_comparison_{dataset}.png")
        plot_metrics_page(dataset, out / f"metrics_comparison_{dataset}.png")
        plot_bootstrap_page(dataset, out / f"bootstrap_brier_ci_{dataset}.png")

    print(f"\nAll pages → {out}")


if __name__ == "__main__":
    main()