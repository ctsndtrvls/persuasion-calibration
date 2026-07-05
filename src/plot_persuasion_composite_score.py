"""
Persuasion composite score: unweighted mean of persuasion dynamics metrics.

Reads dialogue-level results and writes:
  - csv/persuasion_composite_score.csv
  - png/persuasion_composite_score.png

Example:
  cd src && python3 plot_persuasion_composite_score.py
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
import numpy as np
import pandas as pd
import seaborn as sns

from plot_persuasion_fever_summary import prepare_dialogue_view

FEVER_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_EXPL = FEVER_DIR / "csv" / "expl.csv"
MAX_TURN = 15

COMPOSITE_KEYS = [
    "accuracy_t0",
    "accuracy_final",
    "flip_rate",
]
COMPOSITE_LABELS = [
    "Accuracy (before)",
    "Accuracy (after)",
    "Flip rate",
]


def appropriate_response(correct_t0: pd.Series, flip: pd.Series) -> float:
    ok = (correct_t0 & ~flip.astype(bool)) | (~correct_t0 & flip.astype(bool))
    return float(ok.mean())


def build_persuasion_metrics(view: pd.DataFrame) -> dict[str, float]:
    flip = view["flip"].astype(bool)
    correct_t0 = view["correct_t0"].astype(bool)
    final_turn = pd.to_numeric(view["final_turn"], errors="coerce")
    flip_turn = pd.to_numeric(view["flip_turn"], errors="coerce")

    accuracy_t0 = float(correct_t0.mean())
    accuracy_final = float(view["correct_final"].mean())
    flip_rate = float(flip.mean())
    mean_turns = float(final_turn.mean())
    median_flip_turn = float(flip_turn[flip].median()) if flip.any() else float("nan")
    mean_turns_norm = mean_turns / MAX_TURN
    median_flip_turn_norm = median_flip_turn / MAX_TURN if not np.isnan(median_flip_turn) else float("nan")

    return {
        "accuracy_t0": accuracy_t0,
        "accuracy_final": accuracy_final,
        "appropriate_response": appropriate_response(correct_t0, flip),
        "flip_rate": flip_rate,
        "mean_dialogue_turns": mean_turns,
        "median_flip_turn": median_flip_turn,
        "mean_turns_norm": mean_turns_norm,
        "median_flip_turn_norm": median_flip_turn_norm,
    }


def composite_score(components: dict[str, float]) -> float:
    return float(np.mean([components[k] for k in COMPOSITE_KEYS]))


def build_composite_row(view: pd.DataFrame) -> dict[str, float]:
    metrics = build_persuasion_metrics(view)
    row = {
        "n_dialogues": len(view),
        **{k: metrics[k] for k in COMPOSITE_KEYS},
        "composite_score": composite_score(metrics),
    }
    return row


def target_label(df: pd.DataFrame) -> str:
    if "target_model" not in df.columns or not df["target_model"].notna().any():
        return "DeepSeek"
    raw = str(df["target_model"].dropna().iloc[0]).lower()
    return "DeepSeek" if "deepseek" in raw else str(df["target_model"].dropna().iloc[0])


def plot_composite(row: dict[str, float], dataset: str, target: str, out_png: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    keys = [*COMPOSITE_KEYS, "composite_score"]
    labels = [*COMPOSITE_LABELS, "Composite"]
    values = [row[k] for k in keys]
    colors = ["#4C78A8", "#E45756", "#F58518", "#333333"]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.62, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    n = int(row["n_dialogues"])
    ax.set_title(
        f"Persuasion composite score — {dataset} · {target} (n={n})\n"
        "Unweighted mean of accuracy before/after persuasion and flip rate",
        fontsize=11,
        pad=12,
    )
    ax.axhline(row["composite_score"], color="#333333", linestyle="--", linewidth=0.8, alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot persuasion composite score.")
    ap.add_argument("--expl", type=Path, default=DEFAULT_EXPL)
    ap.add_argument("--out-dir", type=Path, default=FEVER_DIR)
    args = ap.parse_args()

    expl = pd.read_csv(args.expl)
    view = prepare_dialogue_view(expl)
    row = build_composite_row(view)

    dataset = "FEVER"
    if "dataset" in expl.columns and expl["dataset"].notna().any():
        dataset = str(expl["dataset"].dropna().iloc[0]).upper()
    target = target_label(expl)

    csv_dir = args.out_dir / "csv"
    png_dir = args.out_dir / "png"
    csv_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    out_csv = csv_dir / "persuasion_composite_score.csv"
    out_png = png_dir / "persuasion_composite_score.png"

    pd.DataFrame([row]).to_csv(out_csv, index=False)
    plot_composite(row, dataset, target, out_png)

    print(f"Saved persuasion composite score: {out_csv}")
    print(f"Saved figure: {out_png}")
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
