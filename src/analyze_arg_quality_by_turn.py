"""
Summarize and plot LLM-judge argument quality across persuasion turns.

Example:
  python3 analyze_arg_quality_by_turn.py \\
    --judge-csv ../output_wood/persuasion/DeepSeek/fever/arg_quality/arg_quality_long_fever214_all_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv
"""
from __future__ import annotations

import argparse
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
import seaborn as sns

from persuasion_arg_quality import TOP_LEVEL_QUALITY_DIMENSIONS

FEVER_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_JUDGE = (
    FEVER_DIR
    / "arg_quality"
    / "arg_quality_long_fever214_all_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv"
)
MAX_TURN = 15
DIM_COLORS = {
    "cogency": "#4C78A8",
    "effectiveness": "#E45756",
    "reasonableness": "#54A24B",
    "mean_score": "#F58518",
}


def summarize_by_turn(df: pd.DataFrame) -> pd.DataFrame:
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if d in df.columns]
    rows = []
    for turn, g in df.groupby("turn"):
        row = {"turn": int(turn), "n_counterarguments": len(g)}
        for d in dims:
            row[f"mean_{d}"] = float(g[d].mean())
        if "mean_score" in g.columns:
            row["mean_overall"] = float(g["mean_score"].mean())
        else:
            row["mean_overall"] = float(g[dims].mean(axis=1).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("turn")


def summarize_paired_delta(df: pd.DataFrame, base_turn: int = 1) -> pd.DataFrame:
    """Per-dialogue change vs base_turn (only dialogues with both turns)."""
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if d in df.columns]
    sub = df.drop_duplicates(subset=["dialogue_id", "turn"], keep="first")
    rows = []
    base = sub[sub["turn"] == base_turn].set_index("dialogue_id")
    for turn in sorted(sub["turn"].unique()):
        if turn == base_turn:
            continue
        other = sub[sub["turn"] == turn].set_index("dialogue_id")
        paired = base.join(other, lsuffix=f"_t{base_turn}", rsuffix=f"_t{turn}", how="inner")
        if paired.empty:
            continue
        row = {"turn": int(turn), "n_paired_dialogues": len(paired)}
        for d in dims:
            row[f"mean_delta_{d}"] = float(
                (paired[f"{d}_t{turn}"] - paired[f"{d}_t{base_turn}"]).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_quality_by_turn(summary: pd.DataFrame, out_png: Path, *, title_suffix: str = "") -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    turns = summary["turn"].to_numpy()
    for dim in TOP_LEVEL_QUALITY_DIMENSIONS:
        col = f"mean_{dim}"
        if col not in summary.columns:
            continue
        ax.plot(
            turns,
            summary[col],
            marker="o",
            linewidth=1.6,
            label=dim.capitalize(),
            color=DIM_COLORS.get(dim, None),
        )
    if "mean_overall" in summary.columns:
        ax.plot(
            turns,
            summary["mean_overall"],
            marker="s",
            linewidth=1.4,
            linestyle="--",
            label="Overall mean",
            color=DIM_COLORS["mean_score"],
            alpha=0.9,
        )
    for _, r in summary.iterrows():
        ax.annotate(
            str(int(r["n_counterarguments"])),
            (r["turn"], r.get("mean_overall", r[f"mean_{TOP_LEVEL_QUALITY_DIMENSIONS[0]}"])),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=7,
            color="#555555",
        )
    ax.set_xlim(0.5, MAX_TURN + 0.5)
    ax.set_xticks(range(1, MAX_TURN + 1))
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("Persuasion turn")
    ax.set_ylabel("Mean judge score (0–3)")
    title = "Argument quality across persuasion turns (LLM judge, top-3 dimensions)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, pad=10)
    ax.legend(loc="lower right", fontsize=9)
    fig.text(
        0.5,
        -0.02,
        "Numbers above points = counterarguments scored at that turn. "
        "Later turns include fewer dialogues (many end earlier).",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paired_delta_vs_turn1(paired: pd.DataFrame, out_png: Path) -> None:
    if paired.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    turns = paired["turn"]
    for dim in TOP_LEVEL_QUALITY_DIMENSIONS:
        col = f"mean_delta_{dim}"
        if col not in paired.columns:
            continue
        ax.plot(turns, paired[col], marker="o", label=f"Δ {dim} (vs turn 1)", color=DIM_COLORS.get(dim))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean score change vs turn 1 (same dialogue)")
    ax.set_title("Within-dialogue quality change relative to turn 1")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Argument quality by persuasion turn.")
    ap.add_argument("--judge-csv", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--out-dir", type=Path, default=FEVER_DIR)
    args = ap.parse_args()

    if not args.judge_csv.exists():
        raise SystemExit(f"Judge CSV not found: {args.judge_csv}")

    df = pd.read_csv(args.judge_csv)
    summary = summarize_by_turn(df)
    paired = summarize_paired_delta(df, base_turn=1)

    csv_dir = args.out_dir / "arg_quality"
    png_dir = args.out_dir / "png"
    csv_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    out_summary = csv_dir / "arg_quality_by_turn_summary.csv"
    out_paired = csv_dir / "arg_quality_paired_delta_vs_turn1.csv"
    out_png = png_dir / "arg_quality_by_turn.png"
    out_delta_png = png_dir / "arg_quality_paired_delta_vs_turn1.png"

    summary.to_csv(out_summary, index=False)
    paired.to_csv(out_paired, index=False)
    plot_quality_by_turn(
        summary,
        out_png,
        title_suffix=f"n={df['dialogue_id'].nunique()} dialogues, {len(df)} scored counterarguments",
    )
    plot_paired_delta_vs_turn1(paired, out_delta_png)

    print(f"Scored rows: {len(df)}")
    print(f"Turns covered: {sorted(df['turn'].unique())}")
    print(f"Saved summary: {out_summary}")
    print(f"Saved paired deltas: {out_paired}")
    print(f"Saved figure: {out_png}")
    if not paired.empty:
        print(f"Saved figure: {out_delta_png}")
    print("\nMean overall by turn:")
    for _, r in summary.iterrows():
        print(f"  turn {int(r['turn']):2d}: n={int(r['n_counterarguments']):4d}  mean={r['mean_overall']:.3f}")


if __name__ == "__main__":
    main()
