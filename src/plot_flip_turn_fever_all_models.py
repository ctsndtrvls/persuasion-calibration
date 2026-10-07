"""
FEVER dialogue-length boxplots by initial verdict, one panel per model.

Drawn at ACL single-column width with Times at 9–11 pt, so the type stays
close to the paper body when the file is included at \\columnwidth.
Models are stacked: a 2×2 grid cannot fit body-sized verdict labels in one column.

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
    VERDICT_ORDER,
    VERDICT_PALETTE,
    build_dialogue_confidence,
    normalize_label,
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
OUT_PDF = ROOT / "output_wood" / "persuasion" / "flip_turn_by_dialogue_fever_all_models.pdf"
VERDICT_TICKS = ["SUPPORTS", "REFUTES", "NEI"]

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
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # ~3.4 in is one ACL column. Type is 9–11 pt, so it matches the body
    # when the image is placed at \columnwidth.
    fig, axes = plt.subplots(4, 1, figsize=(3.42, 7.7), sharex=True, sharey=True)

    for ax, (label, csv_path) in zip(axes, MODELS):
        dlg = load_fever480_dialogues(csv_path, fever480_ids)
        n = len(dlg)
        flip_rate = dlg["flip_final"].mean() * 100 if n else float("nan")
        plot_df = dlg.dropna(subset=["answer_before", "final_turn"]).copy()
        plot_df["final_turn"] = plot_df["final_turn"].astype(int)
        plot_df["answer_before"] = plot_df["answer_before"].map(normalize_label)
        plot_df = plot_df[plot_df["answer_before"].isin(VERDICT_ORDER)]

        sns.boxplot(
            data=plot_df,
            x="answer_before",
            y="final_turn",
            order=VERDICT_ORDER,
            hue="answer_before",
            hue_order=VERDICT_ORDER,
            palette=VERDICT_PALETTE,
            ax=ax,
            legend=False,
            showfliers=False,
            width=0.62,
            linewidth=0.8,
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        sns.stripplot(
            data=plot_df,
            x="answer_before",
            y="final_turn",
            order=VERDICT_ORDER,
            color="#333333",
            alpha=0.28,
            size=1.6,
            jitter=0.22,
            ax=ax,
            zorder=1,
        )
        ax.set_title(f"{label}\nflip rate {flip_rate:.1f}% · n={n}", fontsize=11, pad=3)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-0.6, 20.2)
        ax.set_yticks([0, 5, 10, 15])
        ax.set_xticks(range(len(VERDICT_ORDER)))
        ax.set_xticklabels(VERDICT_TICKS, fontsize=10)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", labelsize=10, pad=1)

        for i, verdict in enumerate(VERDICT_ORDER):
            n_group = int((plot_df["answer_before"] == verdict).sum())
            ax.text(
                i,
                16.4,
                f"n={n_group}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )

    fig.suptitle("FEVER: dialogue length by initial verdict (turn 0)", fontsize=11)
    fig.supylabel("Dialogue length (last turn reached)", fontsize=10)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    plot_all_models()
