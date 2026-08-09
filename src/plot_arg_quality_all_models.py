"""
Cross-model argument-quality summary figures (FEVER / PopQA / DebateQA).

1) Dimension means (horizontal grouped bars) — one PNG per dataset.
2) Flip-turn quality by dimension (vertical grouped bars, style of
   arg_quality_flip middle panel) — one PNG per dataset.

Judge inputs are the existing top-3 LLM-judge CSVs. Most conditions only
score the flip turn of flipped dialogues; DeepSeek FEVER all-turns is
reduced to flip turns for fair comparison. Model order: GPT-4o, DeepSeek,
Gemma, Qwen.

Example:
  python3 plot_arg_quality_all_models.py
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
import numpy as np
import pandas as pd

from analyze_arg_quality_flip import dialogue_flip_meta, summarize_flip_turn_quality
from persuasion_arg_quality import TOP_LEVEL_QUALITY_DIMENSIONS

ROOT = _PROJECT_ROOT
OUT_DIR = ROOT / "output_wood" / "persuasion" / "arg_quality_compare"
MODEL_ORDER = ["GPT-4o", "DeepSeek", "Gemma", "Qwen"]
MODEL_DISPLAY = {
    "GPT-4o": "GPT",
    "DeepSeek": "DeepSeek",
    "Gemma": "Gemma",
    "Qwen": "Qwen",
}
MODEL_COLORS = {
    "GPT-4o": "#4C78A8",
    "DeepSeek": "#F58518",
    "Gemma": "#54A24B",
    "Qwen": "#E45756",
}
DATASETS = {
    "fever": "FEVER",
    "popqa": "PopQA",
    "debateqa": "DebateQA",
}

JUDGE_CSV: dict[tuple[str, str], Path] = {
    ("GPT-4o", "fever"): ROOT
    / "output_wood/persuasion/GPT-4o/fever/arg_quality/arg_quality_long_fever_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
    ("GPT-4o", "popqa"): ROOT
    / "output_wood/persuasion/GPT-4o/popqa/arg_quality/arg_quality_long_popqa_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
    ("GPT-4o", "debateqa"): ROOT
    / "output_wood/persuasion/GPT-4o/debateqa/arg_quality/arg_quality_long_debateqa_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
    ("DeepSeek", "fever"): ROOT
    / "output_wood/persuasion/DeepSeek/fever/arg_quality/arg_quality_long_fever214_all_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
    ("DeepSeek", "popqa"): ROOT
    / "output_wood/persuasion/DeepSeek/popqa/arg_quality/arg_quality_long_popqa_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
    ("DeepSeek", "debateqa"): ROOT
    / "output_wood/persuasion/DeepSeek/debateqa/arg_quality/arg_quality_long_debateqa_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
    ("Gemma", "fever"): ROOT
    / "output_wood/persuasion/Gemma/fever/arg_quality/arg_quality_long_fever_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
    ("Gemma", "popqa"): ROOT
    / "output_wood/persuasion/Gemma/popqa/arg_quality/arg_quality_long_popqa_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
    ("Gemma", "debateqa"): ROOT
    / "output_wood/persuasion/Gemma/debateqa/arg_quality/arg_quality_long_debateqa_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
    ("Qwen", "fever"): ROOT
    / "output_wood/persuasion/Qwen/fever/arg_quality/arg_quality_long_fever_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
    ("Qwen", "popqa"): ROOT
    / "output_wood/persuasion/Qwen/popqa/arg_quality/arg_quality_long_popqa_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
    ("Qwen", "debateqa"): ROOT
    / "output_wood/persuasion/Qwen/debateqa/arg_quality/arg_quality_long_debateqa_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
}

EXPL_CSV: dict[tuple[str, str], Path] = {
    ("GPT-4o", "fever"): ROOT / "output_wood/persuasion/GPT-4o/fever/rollout/csv/expl.csv",
    ("GPT-4o", "popqa"): ROOT / "output_wood/persuasion/GPT-4o/popqa/rollout/csv/expl.csv",
    ("GPT-4o", "debateqa"): ROOT / "output_wood/persuasion/GPT-4o/debateqa/rollout/csv/expl.csv",
    ("DeepSeek", "fever"): ROOT / "output_wood/persuasion/DeepSeek/fever/csv/expl.csv",
    ("DeepSeek", "popqa"): ROOT / "output_wood/persuasion/DeepSeek/popqa/rollout/csv/expl.csv",
    ("DeepSeek", "debateqa"): ROOT / "output_wood/persuasion/DeepSeek/debateqa/rollout/csv/expl.csv",
    ("Gemma", "fever"): ROOT / "output_wood/persuasion/Gemma/fever/rollout/csv/expl.csv",
    ("Gemma", "popqa"): ROOT / "output_wood/persuasion/Gemma/popqa/rollout/csv/expl.csv",
    ("Gemma", "debateqa"): ROOT / "output_wood/persuasion/Gemma/debateqa/rollout/csv/expl.csv",
    ("Qwen", "fever"): ROOT / "output_wood/persuasion/Qwen/fever/rollout/csv/expl.csv",
    ("Qwen", "popqa"): ROOT / "output_wood/persuasion/Qwen/popqa/rollout/csv/expl.csv",
    ("Qwen", "debateqa"): ROOT / "output_wood/persuasion/Qwen/debateqa/rollout/csv/expl.csv",
}


def load_flip_turn_judge(model: str, dataset: str) -> pd.DataFrame:
    """Load judge rows restricted to flip turns of flipped dialogues."""
    path = JUDGE_CSV[(model, dataset)]
    judge = pd.read_csv(path)
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if d in judge.columns]
    if "mean_score" not in judge.columns:
        judge["mean_score"] = judge[dims].mean(axis=1)

    # DeepSeek FEVER is all-turns; restrict to flip turns for cross-model parity.
    if model == "DeepSeek" and dataset == "fever":
        expl = pd.read_csv(EXPL_CSV[(model, dataset)])
        flip_meta = dialogue_flip_meta(expl)
        flipped = flip_meta[flip_meta["flip_final"] == 1][["dialogue_id", "flip_turn"]]
        judge = judge.merge(flipped, on="dialogue_id", how="inner")
        judge = judge[judge["turn"] == judge["flip_turn"].astype(int)].copy()
        judge = judge.drop(columns=["flip_turn"])

    judge = judge.drop_duplicates(subset=["dialogue_id", "turn"], keep="first")
    judge["model"] = model
    judge["dataset"] = dataset
    return judge


def dimension_means_table(judge: pd.DataFrame) -> pd.DataFrame:
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if d in judge.columns]
    row = {d: float(judge[d].mean()) for d in dims}
    row["mean_overall"] = float(judge["mean_score"].mean()) if "mean_score" in judge.columns else float(np.mean(list(row.values())))
    row["n"] = int(len(judge))
    return pd.DataFrame([row])


def plot_dimension_means_dataset(dataset: str, means_df: pd.DataFrame, out_png: Path) -> None:
    dims = list(TOP_LEVEL_QUALITY_DIMENSIONS)
    models = [m for m in MODEL_ORDER if m in set(means_df["model"])]
    y = np.arange(len(dims))
    n_models = len(models)
    height = 0.18
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * height

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for offset, model in zip(offsets, models):
        sub = means_df[means_df["model"] == model].iloc[0]
        vals = [sub[d] for d in dims]
        bars = ax.barh(
            y + offset,
            vals,
            height=height * 0.92,
            color=MODEL_COLORS[model],
            label=f"{MODEL_DISPLAY[model]} (n={int(sub['n'])})",
            alpha=0.92,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                min(val + 0.04, 2.95),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                ha="left",
                fontsize=8,
            )

    overall = means_df["mean_overall"].mean()
    ax.axvline(overall, color="gray", linestyle="--", linewidth=1, label=f"grand mean={overall:.2f}")
    ax.set_yticks(y)
    ax.set_yticklabels(dims)
    ax.set_xlim(0, 3.05)
    ax.set_xlabel("Mean judge score (0–3)")
    ax.set_title(
        f"Argument quality by dimension at flip turn — {DATASETS[dataset]}\n"
        "Flipped dialogues only · LLM judge (gpt-5.4-mini)",
        pad=10,
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.invert_yaxis()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_flip_turn_by_dimension_dataset(dataset: str, means_df: pd.DataFrame, out_png: Path) -> None:
    """Vertical grouped bars by dimension (style of arg_quality_flip middle panel)."""
    dims = list(TOP_LEVEL_QUALITY_DIMENSIONS)
    models = [m for m in MODEL_ORDER if m in set(means_df["model"])]
    x = np.arange(len(dims))
    n_models = len(models)
    width = 0.18
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for offset, model in zip(offsets, models):
        sub = means_df[means_df["model"] == model].iloc[0]
        vals = [sub[d] for d in dims]
        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.92,
            color=MODEL_COLORS[model],
            label=f"{MODEL_DISPLAY[model]} (n={int(sub['n'])})",
            alpha=0.92,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dims])
    ax.set_ylim(0, 3.15)
    ax.set_ylabel("Mean judge score (0–3)")
    ax.set_title(
        f"Argument quality at flip turn by dimension — {DATASETS[dataset]}\n"
        "Flipped dialogues only · LLM judge (gpt-5.4-mini)",
        pad=10,
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_deepseek_fever_flip_vs_other(out_png: Path) -> pd.DataFrame | None:
    """Only condition with all-turns judge scores: DeepSeek FEVER pilot."""
    judge_path = JUDGE_CSV[("DeepSeek", "fever")]
    expl_path = EXPL_CSV[("DeepSeek", "fever")]
    if not judge_path.exists() or not expl_path.exists():
        return None
    judge = pd.read_csv(judge_path)
    expl = pd.read_csv(expl_path)
    # Keep only dialogues present in the 214 pilot judge set.
    ids = set(judge["dialogue_id"])
    expl = expl[expl["dialogue_id"].isin(ids)].copy()
    flip_meta = dialogue_flip_meta(expl)
    from analyze_arg_quality_flip import build_dialogue_quality

    dlg = build_dialogue_quality(judge, flip_meta)
    summary = summarize_flip_turn_quality(judge, dlg)
    summary.to_csv(OUT_DIR / "csv" / "deepseek_fever_flip_vs_other_summary.csv", index=False)

    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if f"mean_{d}" in summary.columns]
    x = np.arange(len(dims))
    width = 0.34
    at_flip = summary[summary["turn_group"] == "At flip turn"].iloc[0]
    other = summary[summary["turn_group"] == "Other turns"].iloc[0]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for group_row, offset, label, color in [
        (at_flip, -width / 2, "At flip turn", "#E45756"),
        (other, width / 2, "Other turns", "#4C78A8"),
    ]:
        vals = [group_row[f"mean_{d}"] for d in dims]
        bars = ax.bar(x + offset, vals, width=width, label=label, color=color, alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dims])
    ax.set_ylim(0, 3.1)
    ax.set_ylabel("Mean judge score (0–3)")
    ax.set_title(
        "Argument quality at flip turn vs other turns — FEVER / DeepSeek\n"
        f"(at flip turn n={int(at_flip['n_counterarguments'])}; "
        f"other turns n={int(other['n_counterarguments'])}; all-turns pilot)",
        pad=8,
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "png").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "csv").mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in DATASETS:
        means_rows = []
        for model in MODEL_ORDER:
            judge = load_flip_turn_judge(model, dataset)
            means = dimension_means_table(judge)
            means["model"] = model
            means["dataset"] = dataset
            means_rows.append(means)
            rows.append(means.iloc[0].to_dict())
        means_df = pd.concat(means_rows, ignore_index=True)
        plot_dimension_means_dataset(
            dataset,
            means_df,
            OUT_DIR / "png" / f"arg_quality_dimension_means_{dataset}_all_models.png",
        )
        plot_flip_turn_by_dimension_dataset(
            dataset,
            means_df,
            OUT_DIR / "png" / f"arg_quality_at_flip_turn_{dataset}_all_models.png",
        )
        print(f"Wrote dimension means + flip-turn bars for {dataset}")

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "csv" / "arg_quality_dimension_means_all_models.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    pilot_png = OUT_DIR / "png" / "arg_quality_flip_vs_other_deepseek_fever.png"
    plot_deepseek_fever_flip_vs_other(pilot_png)
    print(f"Wrote pilot flip-vs-other: {pilot_png}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
