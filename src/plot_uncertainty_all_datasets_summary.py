"""
Compact all-dataset summary figures for persuasion uncertainty scores (Brier).

Writes:
  output_wood/persuasion/png/uncertainty_metrics_brier_all_models_all_datasets.png
  output_wood/persuasion/png/uncertainty_ablation_brier_all_models_all_datasets.png
  output_wood/persuasion/csv/uncertainty_metrics_all_models_summary.csv
  output_wood/persuasion/csv/uncertainty_ablation_all_models_summary.csv

Also copies the fuller per-dataset compare pages into png/ for thesis use.

Example:
  python3 plot_uncertainty_all_datasets_summary.py
"""
from __future__ import annotations

import os
import shutil
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

ROOT = _PROJECT_ROOT
PERSUASION = ROOT / "output_wood" / "persuasion"
COMPARE = PERSUASION / "uncertainty_scores_compare"
OUT_PNG = PERSUASION / "png"
OUT_CSV = PERSUASION / "csv"

MODEL_ORDER = ["GPT-4o", "DeepSeek", "Gemma", "Qwen"]
MODEL_DISPLAY = {"GPT-4o": "GPT", "DeepSeek": "DeepSeek", "Gemma": "Gemma", "Qwen": "Qwen"}
MODEL_COLORS = {
    "GPT-4o": "#2563EB",
    "DeepSeek": "#0F766E",
    "Gemma": "#D97706",
    "Qwen": "#7C3AED",
}
DATASETS = [("fever", "FEVER"), ("popqa", "PopQA"), ("debateqa", "DebateQA")]

METRICS_METHODS = [
    ("U_self", "Self"),
    ("U_marker", "Markers"),
    ("U_flip", "Flip"),
    ("U_pers", r"$U^{\mathrm{pers}}$"),
    ("U_hybrid", "Hybrid"),
]
ABLATION_ORDER = [
    "Flip only (F)",
    "+ Speed (F·S)",
    "+ Arg. quality (F·S·A)",
    "+ Conf. at flip (full U^pers)",
]
ABLATION_SHORT = {
    "Flip only (F)": "F",
    "+ Speed (F·S)": "+S",
    "+ Arg. quality (F·S·A)": "+A",
    "+ Conf. at flip (full U^pers)": "Full",
}


def load_all_metrics() -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for key, label in DATASETS:
            path = PERSUASION / model / key / "uncertainty_scores" / "csv" / "uncertainty_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["model"] = model
            df["model_display"] = MODEL_DISPLAY[model]
            df["dataset"] = label
            df["dataset_key"] = key
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_all_ablation() -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for key, label in DATASETS:
            path = PERSUASION / model / key / "uncertainty_scores" / "csv" / "ablation_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["model"] = model
            df["model_display"] = MODEL_DISPLAY[model]
            df["dataset"] = label
            df["dataset_key"] = key
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _grouped_brier(
    ax: plt.Axes,
    *,
    categories: list[str],
    values: dict[str, list[float]],
    title: str,
    ylabel: bool,
) -> None:
    x = np.arange(len(categories))
    models = list(values.keys())
    n = len(models)
    width = min(0.8 / max(n, 1), 0.18)
    offsets = (np.arange(n) - (n - 1) / 2) * width
    for i, model in enumerate(models):
        ax.bar(
            x + offsets[i],
            values[model],
            width,
            label=MODEL_DISPLAY[model],
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, pad=6)
    if ylabel:
        ax.set_ylabel("Brier ↓")
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.35)


def plot_metrics_brier(metrics: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=False)
    fig.suptitle(
        "Uncertainty method comparison (Brier) across models and datasets",
        fontsize=13,
        weight="bold",
        y=1.02,
    )
    method_ids = [m for m, _ in METRICS_METHODS]
    method_labels = [lab for _, lab in METRICS_METHODS]

    for ax, (_, dataset_label) in zip(axes, DATASETS):
        sub = metrics[metrics["dataset"] == dataset_label]
        values = {}
        n_note = ""
        for model in MODEL_ORDER:
            msub = sub[sub["model"] == model].set_index("method")
            if msub.empty:
                continue
            values[model] = [float(msub.loc[mid, "brier"]) if mid in msub.index else np.nan for mid in method_ids]
            n_note = f"n={int(msub['n'].iloc[0])}"
            if model == "DeepSeek" and dataset_label == "FEVER":
                n_note = f"DeepSeek n={int(msub['n'].iloc[0])}; others n=480"
        _grouped_brier(
            ax,
            categories=method_labels,
            values=values,
            title=f"{dataset_label}",
            ylabel=(ax is axes[0]),
        )
        # annotate n in corner
        err = float(sub["error_rate"].iloc[0]) if len(sub) else float("nan")
        # error rates differ by model slightly only for DeepSeek fever
        ax.text(
            0.98,
            0.98,
            f"err. rate ≈ {err:.0%}" if dataset_label != "FEVER" else "see caption for n",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="#444444",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def plot_ablation_brier(ablation: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=False)
    fig.suptitle(
        "Ablation of $U^{\\mathrm{pers}}$ (Brier) across models and datasets",
        fontsize=13,
        weight="bold",
        y=1.02,
    )
    cats = [ABLATION_SHORT[v] for v in ABLATION_ORDER]
    for ax, (_, dataset_label) in zip(axes, DATASETS):
        sub = ablation[ablation["dataset"] == dataset_label]
        values = {}
        for model in MODEL_ORDER:
            msub = sub[sub["model"] == model].set_index("variant")
            if msub.empty:
                continue
            values[model] = [float(msub.loc[v, "brier"]) for v in ABLATION_ORDER]
        _grouped_brier(
            ax,
            categories=cats,
            values=values,
            title=dataset_label,
            ylabel=(ax is axes[0]),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.text(
        0.5,
        -0.10,
        "Ablation path: F → +S → +A → full $U^{\\mathrm{pers}}$ (adds $C_{\\mathrm{flip}}$).",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def copy_full_compare_pages() -> None:
    """Copy multi-metric per-dataset pages into png/ with thesis-friendly names."""
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    mapping = {
        "metrics_comparison_fever.png": "uncertainty_metrics_comparison_fever.png",
        "metrics_comparison_popqa.png": "uncertainty_metrics_comparison_popqa.png",
        "metrics_comparison_debateqa.png": "uncertainty_metrics_comparison_debateqa.png",
        "ablation_comparison_fever.png": "uncertainty_ablation_comparison_fever.png",
        "ablation_comparison_popqa.png": "uncertainty_ablation_comparison_popqa.png",
        "ablation_comparison_debateqa.png": "uncertainty_ablation_comparison_debateqa.png",
    }
    for src_name, dst_name in mapping.items():
        src = COMPARE / src_name
        if src.exists():
            dst = OUT_PNG / dst_name
            shutil.copy2(src, dst)
            print(f"Copied {dst}")


def main() -> None:
    metrics = load_all_metrics()
    ablation = load_all_ablation()
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_CSV / "uncertainty_metrics_all_models_summary.csv", index=False)
    ablation.to_csv(OUT_CSV / "uncertainty_ablation_all_models_summary.csv", index=False)
    print(f"Wrote {OUT_CSV / 'uncertainty_metrics_all_models_summary.csv'}")
    print(f"Wrote {OUT_CSV / 'uncertainty_ablation_all_models_summary.csv'}")

    plot_metrics_brier(metrics, OUT_PNG / "uncertainty_metrics_brier_all_models_all_datasets.png")
    plot_ablation_brier(ablation, OUT_PNG / "uncertainty_ablation_brier_all_models_all_datasets.png")
    copy_full_compare_pages()


if __name__ == "__main__":
    main()
