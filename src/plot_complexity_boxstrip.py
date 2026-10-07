"""
Complexity of the 480-instance subsets used in the paper.

FEVER and PopQA are stratified by tertiles of the mean Wood (1986) total
(two LLM judges). DebateQA is stratified by terciles of perspective_count,
so it is drawn on its own axis.

Writes paper/figures/dataset_complexity.pdf and .png.
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
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = _PROJECT_ROOT
SUB = ROOT / "output_wood" / "dataset_subsampling"
OUT_DIR = ROOT / "paper" / "figures"

FEVER_CSV = SUB / "fever" / "csv" / "fever480_160x3_complexity_wood_v1_lr_40_resplit.csv"
POPQA_CSV = SUB / "conflictqa" / "csv" / "conflictqa_popqa480_160x3_wood_v2_resplit.csv"
DEBATEQA_CSV = SUB / "debateqa" / "csv" / "debateqa_480_160x3.csv"

LEVEL_MAP = {1: "Low", 2: "Middle", 3: "High"}
LEVEL_ORDER = ["Low", "Middle", "High"]
PALETTE = {"Low": "#4C78A8", "Middle": "#F58518", "High": "#54A24B"}


def _load_levels(path: Path, value_col: str, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if value_col not in df.columns or "complexity_level" not in df.columns:
        raise KeyError(f"{path}: need {value_col!r} and complexity_level")
    out = pd.DataFrame(
        {
            "dataset": dataset_name,
            "complexity": pd.to_numeric(df["complexity_level"], errors="coerce").map(LEVEL_MAP),
            "score": pd.to_numeric(df[value_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["complexity", "score"])
    return out


def _boxplot(ax, df: pd.DataFrame, order: list[str]) -> None:
    sns.boxplot(
        data=df,
        x="dataset",
        y="score",
        hue="complexity",
        order=order,
        hue_order=LEVEL_ORDER,
        palette=PALETTE,
        width=0.7,
        fliersize=2,
        linewidth=0.8,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.legend_.remove()


def main() -> None:
    wood = pd.concat(
        [
            _load_levels(FEVER_CSV, "wood_total_mean", "FEVER"),
            _load_levels(POPQA_CSV, "wood_total_mean", "PopQA"),
        ],
        ignore_index=True,
    )
    debate = _load_levels(DEBATEQA_CSV, "perspective_count", "DebateQA")

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})

    _boxplot(axes[0], wood, ["FEVER", "PopQA"])
    axes[0].set_ylabel("Wood total (mean of two judges)")
    axes[0].set_title("(a) FEVER and PopQA")
    axes[0].set_ylim(6, 25)

    _boxplot(axes[1], debate, ["DebateQA"])
    axes[1].set_ylabel("Annotated perspectives")
    axes[1].set_title("(b) DebateQA")
    axes[1].set_ylim(2, 12)

    handles = [
        plt.Line2D([0], [0], color=PALETTE[name], lw=8, solid_capstyle="butt", label=name)
        for name in LEVEL_ORDER
    ]
    fig.legend(
        handles=handles,
        labels=LEVEL_ORDER,
        title="Complexity",
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "dataset_complexity.pdf"
    png = OUT_DIR / "dataset_complexity.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print("Saved:", pdf)
    print("Saved:", png)
    for name, frame, col in (
        ("FEVER/PopQA", wood, "score"),
        ("DebateQA", debate, "score"),
    ):
        print(name)
        print(
            frame.groupby(["dataset", "complexity"], observed=True)[col]
            .agg(["count", "min", "median", "max"])
            .round(2)
            .to_string()
        )


if __name__ == "__main__":
    main()
