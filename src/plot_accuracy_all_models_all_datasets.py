"""
Accuracy figures across all target models and datasets.

1) accuracy_by_turn_all_models_all_datasets.png
   Mean accuracy among dialogues still active at each turn (until_flip).

2) accuracy_t0_vs_final_all_models_all_datasets.png
   Accuracy at turn 0 vs final turn for all dialogues (persuasion damage).

Example:
  python3 plot_accuracy_all_models_all_datasets.py
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from persuasion_rollout_view import (  # noqa: E402
    answer_matches_gold,
    normalize_fever_label,
    prepare_dialogue_view,
    task_type_from_df,
)

ROOT = _PROJECT_ROOT
OUT_DIR = ROOT / "output_wood" / "persuasion" / "png"
CSV_DIR = ROOT / "output_wood" / "persuasion" / "csv"
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
STAGE_ORDER = ["Turn 0", "Final"]
STAGE_PALETTE = {"Turn 0": "#4C78A8", "Final": "#E45756"}

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


def load_expl(model: str, dataset_key: str, fever480_ids: set[int]) -> pd.DataFrame:
    df = pd.read_csv(EXPL_PATHS[(model, dataset_key)])
    if dataset_key == "fever" and "original_index" in df.columns:
        df = df[df["original_index"].astype(int).isin(fever480_ids)].copy()
    df["turn"] = pd.to_numeric(df["turn"], errors="coerce")
    return df


def first_turn_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_index().drop_duplicates(subset=["dialogue_id", "turn"], keep="first").copy()


def row_correct(answer: object, gold: object, *, task_type: str) -> bool:
    if task_type == "fever":
        return normalize_fever_label(answer) == normalize_fever_label(gold)
    return answer_matches_gold(answer, gold)


def accuracy_by_turn(snap: pd.DataFrame, task_type: str) -> pd.DataFrame:
    # gold per dialogue from turn 0
    gold = (
        snap[snap["turn"] == 0][["dialogue_id", "gold_label"]]
        .drop_duplicates("dialogue_id", keep="first")
    )
    m = snap.merge(gold, on="dialogue_id", how="left", suffixes=("", "_t0"))
    gold_col = "gold_label_t0" if "gold_label_t0" in m.columns else "gold_label"
    m["correct"] = [
        row_correct(a, g, task_type=task_type) for a, g in zip(m["answer"], m[gold_col])
    ]
    return (
        m.groupby("turn", as_index=False)
        .agg(mean_accuracy=("correct", "mean"), n_dialogues=("dialogue_id", "count"))
        .sort_values("turn")
    )


def collect(fever480_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_turn_rows: list[dict] = []
    stats_rows: list[dict] = []

    for dataset_key, dataset_label in DATASETS:
        for model in MODEL_ORDER:
            df = load_expl(model, dataset_key, fever480_ids)
            task_type = task_type_from_df(df)
            snap = first_turn_snapshot(df)
            view = prepare_dialogue_view(df)
            display = MODEL_DISPLAY[model]

            g = accuracy_by_turn(snap, task_type)
            for _, r in g.iterrows():
                by_turn_rows.append(
                    {
                        "model": model,
                        "model_display": display,
                        "dataset": dataset_label,
                        "turn": int(r["turn"]),
                        "mean_accuracy": float(r["mean_accuracy"]) * 100.0,
                        "n_dialogues": int(r["n_dialogues"]),
                    }
                )

            acc_t0 = float(view["correct_t0"].mean()) * 100.0
            acc_final = float(view["correct_final"].mean()) * 100.0
            delta = acc_final - acc_t0
            n = len(view)
            flip_rate = float(view["flip"].mean()) * 100.0

            flipped = view[view["flip"] == True]  # noqa: E712
            if len(flipped):
                acc_t0_flip = float(flipped["correct_t0"].mean()) * 100.0
                acc_final_flip = float(flipped["correct_final"].mean()) * 100.0
                became_wrong = int((flipped["correct_t0"] & ~flipped["correct_final"]).sum())
                became_right = int((~flipped["correct_t0"] & flipped["correct_final"]).sum())
            else:
                acc_t0_flip = acc_final_flip = float("nan")
                became_wrong = became_right = 0

            stats_rows.append(
                {
                    "model": display,
                    "dataset": dataset_label,
                    "n_total": n,
                    "acc_t0_pct": round(acc_t0, 2),
                    "acc_final_pct": round(acc_final, 2),
                    "delta_pp": round(delta, 2),
                    "flip_rate_pct": round(flip_rate, 1),
                    "n_flip": int(len(flipped)),
                    "acc_t0_flipped_pct": round(acc_t0_flip, 2),
                    "acc_final_flipped_pct": round(acc_final_flip, 2),
                    "n_right_to_wrong": became_wrong,
                    "n_wrong_to_right": became_right,
                }
            )

            print(
                f"done {display} {dataset_label}: "
                f"acc {acc_t0:.1f}% → {acc_final:.1f}% (Δ={delta:+.1f} pp)",
                flush=True,
            )

    return pd.DataFrame(by_turn_rows), pd.DataFrame(stats_rows)


def plot_accuracy_by_turn(by_turn: pd.DataFrame, out_png: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), sharey=True)
    fig.suptitle(
        "Mean accuracy by turn across models and datasets",
        fontsize=13,
        weight="bold",
        y=1.02,
    )
    for ax, (_, dataset_label) in zip(axes, DATASETS):
        sub = by_turn[by_turn["dataset"] == dataset_label]
        for model in MODEL_ORDER:
            msub = sub[sub["model"] == model].sort_values("turn")
            if msub.empty:
                continue
            sizes = 20 + 40 * (msub["n_dialogues"] / msub["n_dialogues"].max())
            ax.plot(
                msub["turn"],
                msub["mean_accuracy"],
                color=MODEL_COLORS[model],
                label=MODEL_DISPLAY[model],
                linewidth=2.0,
                zorder=2,
            )
            ax.scatter(
                msub["turn"],
                msub["mean_accuracy"],
                s=sizes,
                color=MODEL_COLORS[model],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        ax.set_title(dataset_label, fontsize=12, pad=8)
        ax.set_xlabel("Turn")
        ax.set_xlim(-0.5, 15.5)
        ax.set_xticks(range(0, 16, 3))
        if ax is axes[0]:
            ax.set_ylabel("Mean accuracy (%)")
        ax.set_ylim(-2, 105)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.text(
        0.5,
        -0.08,
        "Marker size scales with the number of dialogues still active at that turn (until_flip).",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def plot_t0_vs_final(stats_df: pd.DataFrame, out_png: Path) -> None:
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
    # when the image is placed at \columnwidth. Datasets are stacked because
    # a 1×3 row cannot fit body-sized model names and Δ labels in one column.
    fig, axes = plt.subplots(3, 1, figsize=(3.42, 6.35), sharex=True, sharey=True)
    display_order = [MODEL_DISPLAY[m] for m in MODEL_ORDER]
    x = np.arange(len(display_order))
    width = 0.36

    for ax, (_, dataset_label) in zip(axes, DATASETS):
        sub = stats_df[stats_df["dataset"] == dataset_label].set_index("model").loc[display_order]
        bars0 = ax.bar(
            x - width / 2,
            sub["acc_t0_pct"],
            width,
            label="Turn 0",
            color=STAGE_PALETTE["Turn 0"],
            edgecolor="white",
            linewidth=0.6,
        )
        bars1 = ax.bar(
            x + width / 2,
            sub["acc_final_pct"],
            width,
            label="Final",
            color=STAGE_PALETTE["Final"],
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_title(dataset_label, fontsize=11, pad=2)
        ax.set_xticks(x)
        ax.set_xticklabels(display_order, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(0, 152)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", labelsize=10, pad=1)

        # Short bars cannot hold a 9 pt numeral, so the value sits above them.
        # Deltas are two lines so neighbouring "Δ=… pp" labels do not collide.
        pair_tops = [0.0] * len(display_order)
        for bars in (bars0, bars1):
            for bar in bars:
                h = float(bar.get_height())
                cx = bar.get_x() + bar.get_width() / 2
                group = int(round(cx))
                if h >= 22:
                    ax.text(
                        cx,
                        h / 2,
                        f"{h:.0f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                        fontweight="bold",
                    )
                    pair_tops[group] = max(pair_tops[group], h + 14.0)
                else:
                    ax.text(
                        cx,
                        h + 1.2,
                        f"{h:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="#222222",
                    )
                    pair_tops[group] = max(pair_tops[group], h + 16.0)

        for i, name in enumerate(display_order):
            delta = float(sub.loc[name, "delta_pp"])
            ax.text(
                i,
                pair_tops[i] + 1.8,
                f"Δ={delta:+.1f}\npp",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
                linespacing=0.9,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.58, 0.975),
    )
    fig.suptitle(
        "Accuracy at turn 0 vs final turn across models and datasets",
        fontsize=10,
    )
    fig.supylabel("Accuracy (%)", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def main() -> None:
    fever480_ids = set(pd.read_csv(FEVER480)["original_index"].astype(int))
    by_turn, stats = collect(fever480_ids)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    by_turn.to_csv(CSV_DIR / "accuracy_by_turn_all_models.csv", index=False)
    stats.to_csv(CSV_DIR / "accuracy_t0_final_stats.csv", index=False)
    print(f"Wrote {CSV_DIR / 'accuracy_by_turn_all_models.csv'}")
    print(f"Wrote {CSV_DIR / 'accuracy_t0_final_stats.csv'}")
    print(stats.to_string(index=False))

    plot_accuracy_by_turn(by_turn, OUT_DIR / "accuracy_by_turn_all_models_all_datasets.png")
    plot_t0_vs_final(stats, OUT_DIR / "accuracy_t0_vs_final_all_models_all_datasets.png")


if __name__ == "__main__":
    main()
