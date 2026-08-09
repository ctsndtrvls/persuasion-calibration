"""Shared matplotlib helpers for persuasion uncertainty pipeline."""
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
import seaborn as sns

BG = "#F7F9FC"
METHOD_LABELS = {
    "U_self": "Self-report (1−C₀)",
    "U_token": "Token log-prob (1−p)",
    "U_marker": "Epistemic markers",
    "U_flip": "Flip only",
    "U_pers": "Persuasion (U^pers)",
    "U_hybrid": "Hybrid (U^hybrid)",
}
# Okabe–Ito categorical palette without a red–green pair
# (deuteranopia/protanopia safe).
METHOD_COLORS = {
    "U_self": "#0072B2",   # blue
    "U_token": "#000000",  # black
    "U_marker": "#E69F00",  # orange
    "U_flip": "#CC79A7",   # reddish purple
    "U_pers": "#56B4E9",   # sky blue
    "U_hybrid": "#D55E00",  # vermillion (dark orange; not green)
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_feature_overview(features: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flipped = features[features["F"] == 1].copy()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Step 1 — Uncertainty features (214 dialogues)", fontsize=14, weight="bold")

    axes[0, 0].bar(["No flip", "Flip"], [int((features["F"] == 0).sum()), int((features["F"] == 1).sum())],
                   color=["#94A3B8", "#CC79A7"])
    axes[0, 0].set_title("F — verdict flip")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].bar(["Correct t0", "Error t0"], [int((features["E_i"] == 0).sum()), int((features["E_i"] == 1).sum())],
                   color=["#0072B2", "#E69F00"])
    axes[0, 1].set_title("E_i — initial-answer error")

    axes[0, 2].hist(flipped["T"].dropna(), bins=range(1, 17), color="#56B4E9", edgecolor="white")
    axes[0, 2].set_title("T — flip turn (flipped only)")
    axes[0, 2].set_xlabel("Turn")

    axes[1, 0].hist(features["C0"].dropna(), bins=11, range=(0, 1.01), color="#0072B2", edgecolor="white")
    axes[1, 0].set_title("C₀ — initial confidence / 10")
    axes[1, 0].set_xlabel("C₀")

    if not flipped.empty:
        axes[1, 1].hist(flipped["S"].dropna(), bins=12, color="#009E73", edgecolor="white")
        axes[1, 1].set_title("S — persuasion speed (flipped)")
        axes[1, 1].set_xlabel("S")

        axes[1, 2].scatter(flipped["A"], flipped["C_flip"], alpha=0.6, c="#D55E00", edgecolors="white", linewidths=0.4)
        axes[1, 2].set_title("A vs C_flip (flipped)")
        axes[1, 2].set_xlabel("A — argument quality at flip")
        axes[1, 2].set_ylabel("C_flip")
    else:
        axes[1, 1].axis("off")
        axes[1, 2].axis("off")

    fig.tight_layout()
    _save(fig, out_dir / "feature_overview.png")

    if not flipped.empty:
        fig2, ax = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor(BG)
        parts = ["quality_at_flip", "cogency_at_flip", "effectiveness_at_flip", "reasonableness_at_flip"]
        melt = flipped[parts].melt(var_name="dimension", value_name="score")
        melt["dimension"] = melt["dimension"].str.replace("_at_flip", "").str.replace("quality", "overall")
        sns.boxplot(data=melt, x="dimension", y="score", hue="dimension", palette="Set2", legend=False, ax=ax)
        ax.set_title("Argument quality at flip turn")
        ax.set_xlabel("")
        ax.set_ylabel("Judge score (0–3)")
        fig2.tight_layout()
        _save(fig2, out_dir / "quality_at_flip_boxplot.png")


def plot_scores_overview(scores: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [c for c in METHOD_LABELS if c in scores.columns]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Step 2 — Uncertainty scores & baselines", fontsize=14, weight="bold")

    for ax, col in zip(axes.flat, methods[:4]):
        ax.hist(scores[col].dropna(), bins=15, color=METHOD_COLORS.get(col, "#64748B"), edgecolor="white")
        ax.set_title(METHOD_LABELS.get(col, col))
        ax.set_xlabel("Score")

    fig.tight_layout()
    _save(fig, out_dir / "score_distributions.png")

    fig2, ax = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor(BG)
    plot_df = scores[["E_i"] + methods].melt(id_vars="E_i", var_name="method", value_name="score")
    plot_df["method"] = plot_df["method"].map(METHOD_LABELS)
    sns.boxplot(data=plot_df, x="method", y="score", hue="E_i", palette={0: "#0072B2", 1: "#E69F00"}, ax=ax)
    ax.set_title("Scores by initial-answer correctness")
    ax.set_xlabel("")
    ax.set_ylabel("Uncertainty score")
    ax.tick_params(axis="x", rotation=20)
    fig2.tight_layout()
    _save(fig2, out_dir / "scores_by_initial_error.png")

    fig3, ax = plt.subplots(figsize=(7, 6))
    fig3.patch.set_facecolor(BG)
    corr = scores[methods].corr()
    corr.index = [METHOD_LABELS.get(c, c) for c in corr.index]
    corr.columns = [METHOD_LABELS.get(c, c) for c in corr.columns]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Score correlation matrix")
    fig3.tight_layout()
    _save(fig3, out_dir / "score_correlation.png")


def plot_metrics_bar(metrics: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle(title, fontsize=13, weight="bold")
    metric_specs = [
        ("brier", "Brier ↓", True),
        ("auroc", "AUROC ↑", False),
        ("auprc", "AUPRC ↑", False),
        ("uce", "UCE ↓", True),
    ]
    for ax, (col, label, lower_better) in zip(axes, metric_specs):
        sub = metrics.sort_values(col, ascending=lower_better)
        colors = [METHOD_COLORS.get(m, "#64748B") for m in sub["method"]]
        ax.barh(sub["label"], sub[col], color=colors)
        ax.set_title(label)
        ax.set_xlim(left=0)
    fig.tight_layout()
    _save(fig, out_path)


def plot_roc_curves(roc_data: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)
    for method, (fpr, tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method, "#64748B"), lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — predicting initial error")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def plot_reliability(bin_stats: dict[str, pd.DataFrame], out_path: Path) -> None:
    n = len(bin_stats)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes).flatten()
    for ax, (method, stats) in zip(axes, bin_stats.items()):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.scatter(stats["mean_u"], stats["mean_e"], s=stats["count"] * 3, alpha=0.8,
                   color=METHOD_COLORS.get(method, "#64748B"))
        ax.plot(stats["mean_u"], stats["mean_e"], color=METHOD_COLORS.get(method, "#64748B"), alpha=0.5)
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted U")
        ax.set_ylabel("Observed error rate")
    for ax in axes[len(bin_stats):]:
        ax.axis("off")
    fig.suptitle("Reliability diagrams (5 equal-frequency bins)", fontsize=12, weight="bold")
    fig.tight_layout()
    _save(fig, out_path)


def plot_risk_coverage(curves: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)
    for method, curve in curves.items():
        ax.plot(curve["coverage"], curve["risk"], label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, "#64748B"), lw=2)
    ax.set_xlabel("Coverage (fraction retained, least uncertain first)")
    ax.set_ylabel("Error rate among retained")
    ax.set_title("Risk–coverage curves")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, out_path)


def plot_ablation(ablation: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Ablation — adding components to flip-only baseline", fontsize=12, weight="bold")
    specs = [("brier", "Brier ↓", True), ("auroc", "AUROC ↑", False), ("uce", "UCE ↓", True)]
    for ax, (col, label, asc) in zip(axes, specs):
        sub = ablation.sort_values(col, ascending=asc)
        ax.barh(sub["variant"], sub[col], color="#3B82F6")
        ax.set_title(label)
    fig.tight_layout()
    _save(fig, out_path)


def plot_bootstrap_ci(bootstrap: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    y_pos = np.arange(len(bootstrap))
    ax.barh(y_pos, bootstrap["mean_delta"], xerr=[
        bootstrap["mean_delta"] - bootstrap["ci_low"],
        bootstrap["ci_high"] - bootstrap["mean_delta"],
    ], color="#0072B2", alpha=0.85, capsize=4)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bootstrap["comparison"])
    ax.set_xlabel("Δ metric (composite − baseline); positive Brier Δ = composite better")
    ax.set_title("Paired cluster-bootstrap 95% CI (Brier difference)")
    fig.tight_layout()
    _save(fig, out_path)
