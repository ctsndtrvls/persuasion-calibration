"""
Mean judge scores by argument-quality dimension (top-3 scheme).

Example:
  python3 plot_persuasion_arg_quality.py
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
import pandas as pd

from persuasion_arg_quality import TOP_LEVEL_QUALITY_DIMENSIONS

FEVER_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_JUDGE = (
    FEVER_DIR
    / "arg_quality"
    / "arg_quality_long_fever214_no_flip_t1_v1__top3__openrouter__openai_gpt-5.4-mini.csv"
)
DEFAULT_OUT = FEVER_DIR / "png" / "arg_quality_dimension_means.png"


def plot_dimension_means(df: pd.DataFrame, out_png: Path) -> None:
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if d in df.columns]
    means = df[dims].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.9 * len(means))))
    median = means.median()
    colors = [
        "#4C78A8"
        if dim == "effectiveness"
        else ("#4C78A8" if v >= median else "#E45756")
        for dim, v in means.items()
    ]
    means.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlim(0, 3)
    ax.set_xlabel("Mean judge score (0–3)")
    ax.set_title("Argument quality by dimension (214 turn-1 counterarguments)")
    ax.axvline(
        means.mean(),
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"overall mean={means.mean():.2f}",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-csv", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_csv(args.judge_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_dimension_means(df, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
