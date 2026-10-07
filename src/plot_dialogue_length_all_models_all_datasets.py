"""
Dialogue-length boxplots across all target models and datasets.

Drawn at ACL single-column width with Times at 9–11 pt, so the type stays
close to the paper body when the file is included at \\columnwidth.
Datasets are stacked: a side-by-side row cannot fit those labels in one column.

Example:
  python3 plot_dialogue_length_all_models_all_datasets.py
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

from analyze_persuasion_confidence import build_dialogue_confidence  # noqa: E402


ROOT = _PROJECT_ROOT
OUT_PNG = ROOT / "output_wood" / "persuasion" / "png" / "dialogue_length_all_models_all_datasets.png"
OUT_PDF = ROOT / "output_wood" / "persuasion" / "png" / "dialogue_length_all_models_all_datasets.pdf"
FEVER480 = (
    ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "fever"
    / "csv"
    / "fever480_160x3_complexity_wood_v1_lr_40_resplit.csv"
)

MODEL_ORDER = ["GPT-4o", "DeepSeek", "Gemma", "Qwen"]
MODEL_DISPLAY = {"GPT-4o": "GPT", "DeepSeek": "DeepSeek", "Gemma": "Gemma", "Qwen": "Qwen"}
MODEL_COLORS = {
    "GPT-4o": "#4C78A8",
    "DeepSeek": "#F58518",
    "Gemma": "#54A24B",
    "Qwen": "#E45756",
}
DATASETS = [("fever", "FEVER"), ("popqa", "PopQA"), ("debateqa", "DebateQA")]

EXPL_PATHS = {
    ("GPT-4o", "fever"): ROOT / "output_wood/persuasion/GPT-4o/fever/rollout/csv/expl.csv",
    ("DeepSeek", "fever"): ROOT / "output_wood/persuasion/DeepSeek/fever/csv/expl.csv",
    ("Gemma", "fever"): ROOT / "output_wood/persuasion/Gemma/fever/rollout/csv/expl.csv",
    ("Qwen", "fever"): ROOT / "output_wood/persuasion/Qwen/fever/rollout/csv/expl.csv",
    ("GPT-4o", "popqa"): ROOT / "output_wood/persuasion/GPT-4o/popqa/rollout/csv/expl.csv",
    ("DeepSeek", "popqa"): ROOT / "output_wood/persuasion/DeepSeek/popqa/rollout/csv/expl.csv",
    ("Gemma", "popqa"): ROOT / "output_wood/persuasion/Gemma/popqa/rollout/csv/expl.csv",
    ("Qwen", "popqa"): ROOT / "output_wood/persuasion/Qwen/popqa/rollout/csv/expl.csv",
    ("GPT-4o", "debateqa"): ROOT / "output_wood/persuasion/GPT-4o/debateqa/rollout/csv/expl.csv",
    ("DeepSeek", "debateqa"): ROOT / "output_wood/persuasion/DeepSeek/debateqa/rollout/csv/expl.csv",
    ("Gemma", "debateqa"): ROOT / "output_wood/persuasion/Gemma/debateqa/rollout/csv/expl.csv",
    ("Qwen", "debateqa"): ROOT / "output_wood/persuasion/Qwen/debateqa/rollout/csv/expl.csv",
}


def load_dialogues(model: str, dataset_key: str, fever480_ids: set[int]) -> pd.DataFrame:
    df = pd.read_csv(EXPL_PATHS[(model, dataset_key)])
    if dataset_key == "fever" and "original_index" in df.columns:
        df = df[df["original_index"].astype(int).isin(fever480_ids)]
    return build_dialogue_confidence(df)


def collect(fever480_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_rows: list[dict] = []
    stats_rows: list[dict] = []
    for dataset_key, dataset_label in DATASETS:
        for model in MODEL_ORDER:
            dlg = load_dialogues(model, dataset_key, fever480_ids)
            turns = pd.to_numeric(dlg["final_turn"], errors="coerce").dropna()
            for turn in turns:
                plot_rows.append(
                    {
                        "model": model,
                        "model_display": MODEL_DISPLAY[model],
                        "dataset": dataset_label,
                        "final_turn": int(turn),
                    }
                )
            n = int(len(dlg))
            flip_rate = float(dlg["flip_final"].mean() * 100) if n else float("nan")
            stats_rows.append(
                {
                    "model": MODEL_DISPLAY[model],
                    "dataset": dataset_label,
                    "n": n,
                    "flip_rate": flip_rate,
                }
            )
    return pd.DataFrame(plot_rows), pd.DataFrame(stats_rows)


def plot_dialogue_length(plot_df: pd.DataFrame, stats_df: pd.DataFrame, out_png: Path = OUT_PNG) -> None:
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
    fig, axes = plt.subplots(3, 1, figsize=(3.42, 6.15), sharex=True, sharey=True)
    display_order = [MODEL_DISPLAY[m] for m in MODEL_ORDER]
    palette = {MODEL_DISPLAY[m]: MODEL_COLORS[m] for m in MODEL_ORDER}

    for ax, (_, dataset_label) in zip(axes, DATASETS):
        sub = plot_df[plot_df["dataset"] == dataset_label]
        sns.boxplot(
            data=sub,
            x="model_display",
            y="final_turn",
            order=display_order,
            hue="model_display",
            hue_order=display_order,
            palette=palette,
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
            data=sub,
            x="model_display",
            y="final_turn",
            order=display_order,
            color="#333333",
            alpha=0.28,
            size=1.6,
            jitter=0.22,
            ax=ax,
            zorder=1,
        )
        ax.set_title(dataset_label, fontsize=11, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-0.6, 22.8)
        ax.set_yticks([0, 5, 10, 15])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", labelsize=10, pad=1)

        stats_sub = stats_df[stats_df["dataset"] == dataset_label].set_index("model")
        for i, name in enumerate(display_order):
            n = int(stats_sub.loc[name, "n"])
            rate = float(stats_sub.loc[name, "flip_rate"])
            ax.text(
                i,
                16.6,
                f"n={n}\nflip {rate:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
                linespacing=1.05,
            )

    fig.suptitle("Dialogue length across target models and datasets", fontsize=11)
    fig.supylabel("Dialogue length (last turn reached)", fontsize=10)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {OUT_PDF}")


def main() -> None:
    fever480_ids = set(pd.read_csv(FEVER480)["original_index"].astype(int))
    plot_df, stats_df = collect(fever480_ids)
    print(stats_df.to_string(index=False))
    plot_dialogue_length(plot_df, stats_df)


if __name__ == "__main__":
    main()
