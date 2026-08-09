"""
Four-panel FEVER boxplot (flip-turn style) for all target models.

Filters each rollout to the canonical fever480 original_index set so DeepSeek
(pilot + fill480) is comparable to GPT-4o / Gemma / Qwen.

Example:
  python3 plot_flip_turn_fever_all_models.py
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

from analyze_persuasion_confidence import (  # noqa: E402
    VERDICT_PALETTE,
    _annotate_verdict_turn_boxplot,
    build_dialogue_confidence,
)

ROOT = _PROJECT_ROOT
FEVER480 = (
    ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "fever"
    / "csv"
    / "fever480_160x3_complexity_wood_v1_lr_40_resplit.csv"
)
OUT_PNG = ROOT / "output_wood" / "persuasion" / "flip_turn_by_dialogue_fever_all_models.png"

MODELS = [
    ("DeepSeek", ROOT / "output_wood/persuasion/DeepSeek/fever/csv/expl.csv"),
    ("GPT-4o", ROOT / "output_wood/persuasion/GPT-4o/fever/rollout/csv/expl.csv"),
    ("Gemma", ROOT / "output_wood/persuasion/Gemma/fever/rollout/csv/expl.csv"),
    ("Qwen", ROOT / "output_wood/persuasion/Qwen/fever/rollout/csv/expl.csv"),
]


def load_fever480_dialogues(csv_path: Path, fever480_ids: set[int]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["original_index"].astype(int).isin(fever480_ids)]
    return build_dialogue_confidence(df)


def plot_all_models(out_png: Path = OUT_PNG) -> None:
    fever480_ids = set(pd.read_csv(FEVER480)["original_index"].astype(int))
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.suptitle(
        "FEVER: dialogue length by initial verdict (turn 0)",
        fontsize=14,
        weight="bold",
        y=0.98,
    )

    for ax, (label, csv_path) in zip(axes.flat, MODELS):
        dlg = load_fever480_dialogues(csv_path, fever480_ids)
        n = len(dlg)
        flip_rate = dlg["flip_final"].mean() * 100 if n else float("nan")
        _annotate_verdict_turn_boxplot(ax, dlg, "answer_before")
        ax.set_title(f"{label} · FEVER · flip rate {flip_rate:.1f}% · n={n}", fontsize=11, pad=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    plot_all_models()
