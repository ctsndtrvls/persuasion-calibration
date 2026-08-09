"""
Flip-turn boxplots across all target models and datasets (flipped dialogues only).

Complements dialogue_length_all_models_all_datasets.png: that figure includes
no-flip dialogues at the turn-15 cap; this one shows *when* flips occur among
dialogues that do flip.

Example:
  python3 plot_flip_turn_all_models_all_datasets.py
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
OUT_PNG = ROOT / "output_wood" / "persuasion" / "png" / "flip_turn_all_models_all_datasets.png"
OUT_CSV = ROOT / "output_wood" / "persuasion" / "csv" / "flip_turn_distribution_stats.csv"
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


def collect_flip_rows(fever480_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_rows: list[dict] = []
    stats_rows: list[dict] = []
    for dataset_key, dataset_label in DATASETS:
        for model in MODEL_ORDER:
            dlg = load_dialogues(model, dataset_key, fever480_ids)
            flipped = dlg[dlg["flip_final"] == 1].copy()
            ft = pd.to_numeric(flipped["flip_turn"], errors="coerce").dropna()
            for turn in ft:
                plot_rows.append(
                    {
                        "model": model,
                        "model_display": MODEL_DISPLAY[model],
                        "dataset": dataset_label,
                        "flip_turn": int(turn),
                    }
                )
            n = len(ft)
            stats_rows.append(
                {
                    "model": MODEL_DISPLAY[model],
                    "dataset": dataset_label,
                    "n_total": len(dlg),
                    "n_flip": int(n),
                    "flip_rate": round(float(dlg["flip_final"].mean() * 100), 1),
                    "median": float(ft.median()) if n else float("nan"),
                    "q1": float(ft.quantile(0.25)) if n else float("nan"),
                    "q3": float(ft.quantile(0.75)) if n else float("nan"),
                    "mean": round(float(ft.mean()), 2) if n else float("nan"),
                    "pct_leq1": round(float((ft <= 1).mean() * 100), 1) if n else float("nan"),
                    "pct_leq2": round(float((ft <= 2).mean() * 100), 1) if n else float("nan"),
                    "pct_leq3": round(float((ft <= 3).mean() * 100), 1) if n else float("nan"),
                    "pct_ge10": round(float((ft >= 10).mean() * 100), 1) if n else float("nan"),
                }
            )
    return pd.DataFrame(plot_rows), pd.DataFrame(stats_rows)


def plot_flip_turns(plot_df: pd.DataFrame, stats_df: pd.DataFrame, out_png: Path = OUT_PNG) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    fig.suptitle(
        "Turn of first verdict change across models and datasets",
        fontsize=13,
        weight="bold",
        y=1.02,
    )
    display_order = [MODEL_DISPLAY[m] for m in MODEL_ORDER]
    palette = {MODEL_DISPLAY[m]: MODEL_COLORS[m] for m in MODEL_ORDER}

    for ax, (_, dataset_label) in zip(axes, DATASETS):
        sub = plot_df[plot_df["dataset"] == dataset_label]
        sns.boxplot(
            data=sub,
            x="model_display",
            y="flip_turn",
            order=display_order,
            hue="model_display",
            hue_order=display_order,
            palette=palette,
            ax=ax,
            legend=False,
            showfliers=False,
            width=0.55,
        )
        sns.stripplot(
            data=sub,
            x="model_display",
            y="flip_turn",
            order=display_order,
            color="#555555",
            alpha=0.18,
            size=2.2,
            jitter=0.25,
            ax=ax,
            zorder=1,
        )
        ax.set_title(dataset_label, fontsize=12, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Flip turn (flipped dialogues only)" if ax is axes[0] else "")
        ax.set_ylim(0, 16)
        ax.set_yticks(range(0, 16, 5))

        stats_sub = stats_df[stats_df["dataset"] == dataset_label].set_index("model")
        for i, name in enumerate(display_order):
            n_flip = int(stats_sub.loc[name, "n_flip"])
            ax.text(i, 15.4, f"n={n_flip}", ha="center", va="bottom", fontsize=8, color="#333333")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    fever480_ids = set(pd.read_csv(FEVER480)["original_index"].astype(int))
    plot_df, stats_df = collect_flip_rows(fever480_ids)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    plot_flip_turns(plot_df, stats_df)


if __name__ == "__main__":
    main()
