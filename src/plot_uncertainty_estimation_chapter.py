"""
Paper figures for the Uncertainty Estimation section.

Reads the learned-logistic tables and writes:
  paper/figures/uncertainty_brier.pdf
  paper/figures/uncertainty_coefficients.pdf
  paper/figures/coefficient_heatmap.pdf

Example:
  cd src && python3 plot_uncertainty_estimation_chapter.py
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = _PROJECT_ROOT / "output_wood" / "persuasion" / "learned_logistic" / "csv" / "metrics.csv"
COEFS = _PROJECT_ROOT / "output_wood" / "persuasion" / "learned_logistic" / "csv" / "coefficients.csv"
OUT_DIR = _PROJECT_ROOT / "paper" / "figures"

# Okabe–Ito, color-blind safe.
COLOR_BASE = "#7F7F7F"
COLOR_MANUAL = "#0072B2"
COLOR_LEARNED = "#E69F00"

SLICE_LABELS = {
    "fever": "FEVER",
    "popqa": "PopQA",
    "debateqa": "DebateQA",
    "all": "All",
}
TERM_LABELS = {
    "intercept": "Intercept",
    "flip": "Flip",
    "flip_x_speed": "Flip × speed",
    "flip_x_weak_argument": "Flip × weak argument",
    "flip_x_uncertainty_at_flip": "Flip × uncertainty at flip",
}
TERM_ORDER = list(TERM_LABELS)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_brier(metrics: pd.DataFrame) -> None:
    # ~3.4 in is one ACL column. Type is 9–11 pt, so it matches the body
    # when the image is placed at \columnwidth. Panels are stacked because
    # a side-by-side pair cannot fit those labels in one column.
    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, axes = plt.subplots(2, 1, figsize=(3.42, 6.35), sharey=True)
        panels = [
            (axes[0], "local", ["fever", "popqa", "debateqa"], "(a) Local: a separate fit for each dataset"),
            (axes[1], "global", ["all", "fever", "popqa", "debateqa"], "(b) Global: one fit, scored on each dataset"),
        ]
        methods = [
            ("intercept_only", "Error rate only", COLOR_BASE),
            ("manual", "Fixed weights", COLOR_MANUAL),
            ("learned", "Learned weights", COLOR_LEARNED),
        ]

        def brier(scope: str, slice_name: str, method: str) -> float:
            # Gray bars are the group's own error rate. On the global
            # panel that is not the pooled intercept scored inside one dataset.
            if method == "intercept_only" and slice_name != "all":
                scope = "local"
            hit = metrics[
                (metrics["scope"] == scope)
                & (metrics["slice"] == slice_name)
                & (metrics["method"] == method)
            ]
            return float(hit["brier"].iloc[0])

        for ax, scope, slices, title in panels:
            x = np.arange(len(slices))
            bar_w = 0.22
            # Gap between bars so the rotated 9 pt values do not touch.
            step = 0.32
            for i, (method, label, color) in enumerate(methods):
                heights = [brier(scope, slice_name, method) for slice_name in slices]
                positions = x + (i - 1) * step
                ax.bar(
                    positions,
                    heights,
                    bar_w,
                    label=label,
                    color=color,
                    edgecolor="none",
                )
                for xpos, height in zip(positions, heights):
                    ax.text(
                        xpos,
                        height + 0.01,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=9,
                        color="#222222",
                        clip_on=False,
                    )
            ax.set_xticks(x)
            ax.set_xticklabels([SLICE_LABELS[s] for s in slices], fontsize=10)
            ax.set_title(title, loc="left", fontsize=11, pad=6)
            ax.set_ylim(0, 0.52)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.6)
            ax.tick_params(axis="y", labelsize=10)
            ax.set_ylabel("")
        fig.supylabel("Brier score (lower is better)", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.56, 0.985),
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=0.9,
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.90))
        _save(fig, "uncertainty_brier")


def plot_coefficients(coefs: pd.DataFrame) -> None:
    panels = [
        ("local", "fever", "Local: FEVER"),
        ("local", "popqa", "Local: PopQA (wider axis)"),
        ("local", "debateqa", "Local: DebateQA"),
        ("global", "all", "Global"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.7))
    for ax, (scope, dataset, title) in zip(axes.ravel(), panels):
        sub = coefs[(coefs["scope"] == scope) & (coefs["dataset"] == dataset)].copy()
        sub["term"] = pd.Categorical(sub["term"], TERM_ORDER, ordered=True)
        sub = sub.sort_values("term")
        y = np.arange(len(sub))
        ax.axvline(0, color="#B0B0B0", linewidth=0.8, zorder=0)
        ax.errorbar(
            sub["fold_mean"],
            y,
            xerr=sub["fold_std"],
            fmt="o",
            color=COLOR_MANUAL,
            markersize=4.5,
            elinewidth=1.0,
            capsize=2.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([TERM_LABELS[t] for t in sub["term"]])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", pad=6)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.6)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.supxlabel(
        "← initial answer more likely correct          more likely wrong →",
        fontsize=8,
        y=0.02,
    )
    _save(fig, "uncertainty_coefficients")


def plot_coefficient_heatmap(coefs: pd.DataFrame) -> None:
    """Fold-mean log-odds coefficients. Replaces the raises/lowers table."""
    terms = [
        "flip",
        "flip_x_speed",
        "flip_x_weak_argument",
        "flip_x_uncertainty_at_flip",
    ]
    row_labels = [
        "Flip",
        "Flip × speed",
        "Flip × weak argument",
        "Flip × low confidence",
    ]
    columns = [
        ("local", "fever", "Local FEVER"),
        ("local", "popqa", "Local PopQA"),
        ("local", "debateqa", "Local DebateQA"),
        ("global", "all", "Global"),
    ]
    means = np.zeros((len(terms), len(columns)))
    stds = np.zeros_like(means)
    for j, (scope, dataset, _) in enumerate(columns):
        sub = coefs[(coefs["scope"] == scope) & (coefs["dataset"] == dataset)]
        by_term = sub.set_index("term")
        for i, term in enumerate(terms):
            means[i, j] = float(by_term.loc[term, "fold_mean"])
            stds[i, j] = float(by_term.loc[term, "fold_std"])

    limit = float(np.max(np.abs(means)))
    fig, ax = plt.subplots(figsize=(7.05, 2.85))
    image = ax.imshow(means, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([label for _, _, label in columns])
    ax.set_yticks(np.arange(len(terms)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    threshold = 0.45 * limit
    for i in range(len(terms)):
        for j in range(len(columns)):
            color = "white" if abs(means[i, j]) > threshold else "#1A1A1A"
            ax.text(
                j,
                i,
                f"{means[i, j]:+.2f}\n± {stds[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                linespacing=1.15,
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Mean log-odds coefficient", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    _save(fig, "coefficient_heatmap")


def main() -> None:
    _style()
    plot_brier(pd.read_csv(METRICS))
    plot_coefficients(pd.read_csv(COEFS))
    plot_coefficient_heatmap(pd.read_csv(COEFS))
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
