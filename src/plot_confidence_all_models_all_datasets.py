"""
Confidence figures across all target models and datasets.

1) mean_confidence_by_turn_all_models_all_datasets.png
   Mean self-reported confidence by turn (active dialogues; n shrinks under until_flip).

2) confidence_c0_vs_cflip_all_models_all_datasets.png
   Turn-0 confidence (C0) vs confidence at the flip turn (C_flip), flipped dialogues only.
   These are the confidence signals that enter U^pers / U^hybrid.

Example:
  python3 plot_confidence_all_models_all_datasets.py
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
STAGE_ORDER = ["Turn 0 (C0)", "At flip (C_flip)"]
STAGE_PALETTE = {"Turn 0 (C0)": "#4C78A8", "At flip (C_flip)": "#E45756"}

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
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["flipped_from_initial"] = (
        pd.to_numeric(df.get("flipped_from_initial"), errors="coerce").fillna(0).astype(int)
    )
    df["flip_turn"] = pd.to_numeric(df.get("flip_turn"), errors="coerce")
    return df


def first_turn_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """First observation per (dialogue, turn)."""
    return (
        df.sort_index()
        .drop_duplicates(subset=["dialogue_id", "turn"], keep="first")
        .copy()
    )


def dialogue_flip_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Flip flag / flip_turn from the final row of each dialogue (full file order)."""
    last = df.sort_index().groupby("dialogue_id", sort=False).tail(1)
    out = last[["dialogue_id", "flipped_from_initial", "flip_turn"]].copy()
    out = out.rename(columns={"flipped_from_initial": "flip_final"})
    out["flip_final"] = out["flip_final"].fillna(0).astype(int)
    return out


def collect(fever480_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_turn_rows: list[dict] = []
    paired_rows: list[dict] = []
    stats_rows: list[dict] = []

    for dataset_key, dataset_label in DATASETS:
        for model in MODEL_ORDER:
            df = load_expl(model, dataset_key, fever480_ids)
            snap = first_turn_snapshot(df)
            meta = dialogue_flip_meta(df)
            display = MODEL_DISPLAY[model]

            # Mean confidence by turn (all active dialogues at that turn)
            g = (
                snap.groupby("turn", as_index=False)["confidence"]
                .agg(mean_confidence="mean", n_dialogues="count")
                .sort_values("turn")
            )
            for _, r in g.iterrows():
                by_turn_rows.append(
                    {
                        "model": model,
                        "model_display": display,
                        "dataset": dataset_label,
                        "turn": int(r["turn"]),
                        "mean_confidence": float(r["mean_confidence"]),
                        "n_dialogues": int(r["n_dialogues"]),
                    }
                )

            # C0 for all dialogues
            t0 = snap[snap["turn"] == 0][["dialogue_id", "confidence"]].rename(
                columns={"confidence": "conf_before"}
            )
            merged = meta.merge(t0, on="dialogue_id", how="left")

            # C_flip for flipped dialogues
            flipped = merged[merged["flip_final"] == 1].copy()
            flip_turns = flipped[["dialogue_id", "flip_turn"]].rename(
                columns={"flip_turn": "flip_turn_meta"}
            )
            at_flip = snap.merge(flip_turns, on="dialogue_id", how="inner")
            at_flip = at_flip[at_flip["turn"] == at_flip["flip_turn_meta"]][
                ["dialogue_id", "confidence"]
            ].rename(columns={"confidence": "conf_at_flip"})
            paired = flipped.merge(at_flip, on="dialogue_id", how="inner")
            paired = paired.dropna(subset=["conf_before", "conf_at_flip"])

            for rec in paired[
                ["dialogue_id", "conf_before", "conf_at_flip"]
            ].itertuples(index=False):
                paired_rows.append(
                    {
                        "model": model,
                        "model_display": display,
                        "dataset": dataset_label,
                        "dialogue_id": rec.dialogue_id,
                        "conf_before": float(rec.conf_before),
                        "conf_at_flip": float(rec.conf_at_flip),
                        "delta": float(rec.conf_at_flip - rec.conf_before),
                    }
                )

            mean_c0_all = float(merged["conf_before"].mean())
            mean_c0_flip = float(paired["conf_before"].mean()) if len(paired) else float("nan")
            mean_cflip = float(paired["conf_at_flip"].mean()) if len(paired) else float("nan")
            mean_delta = (
                float(paired["conf_at_flip"].mean() - paired["conf_before"].mean())
                if len(paired)
                else float("nan")
            )
            pct_drop = (
                float((paired["conf_at_flip"] < paired["conf_before"]).mean() * 100)
                if len(paired)
                else float("nan")
            )
            pct_rise = (
                float((paired["conf_at_flip"] > paired["conf_before"]).mean() * 100)
                if len(paired)
                else float("nan")
            )
            pct_same = (
                float((paired["conf_at_flip"] == paired["conf_before"]).mean() * 100)
                if len(paired)
                else float("nan")
            )

            stats_rows.append(
                {
                    "model": display,
                    "dataset": dataset_label,
                    "n_total": int(len(merged)),
                    "n_flip": int(len(paired)),
                    "mean_C0_all": round(mean_c0_all, 3),
                    "mean_C0_flipped": round(mean_c0_flip, 3),
                    "mean_C_flip": round(mean_cflip, 3),
                    "mean_delta_Cflip_minus_C0": round(mean_delta, 3),
                    "pct_drop_at_flip": round(pct_drop, 1),
                    "pct_rise_at_flip": round(pct_rise, 1),
                    "pct_unchanged_at_flip": round(pct_same, 1),
                    "mean_conf_turn0_curve": round(
                        float(g.loc[g["turn"] == 0, "mean_confidence"].iloc[0]), 3
                    )
                    if (g["turn"] == 0).any()
                    else float("nan"),
                    "mean_conf_turn1_curve": round(
                        float(g.loc[g["turn"] == 1, "mean_confidence"].iloc[0]), 3
                    )
                    if (g["turn"] == 1).any()
                    else float("nan"),
                }
            )
            print(
                f"done {display} {dataset_label}: n_flip={len(paired)} Δ={mean_delta:+.2f}",
                flush=True,
            )

    wide_df = pd.DataFrame(paired_rows)
    long_df = wide_df.melt(
        id_vars=["model", "model_display", "dataset", "dialogue_id", "delta"],
        value_vars=["conf_before", "conf_at_flip"],
        var_name="stage_key",
        value_name="confidence",
    )
    long_df["stage"] = long_df["stage_key"].map(
        {"conf_before": "Turn 0 (C0)", "conf_at_flip": "At flip (C_flip)"}
    )

    by_turn_df = pd.DataFrame(by_turn_rows)
    stats_df = pd.DataFrame(stats_rows)
    return by_turn_df, long_df, stats_df


def plot_mean_by_turn(by_turn: pd.DataFrame, out_png: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), sharey=True)
    fig.suptitle(
        "Mean self-reported confidence by turn across models and datasets",
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
            # marker size ~ remaining dialogues
            sizes = 20 + 40 * (msub["n_dialogues"] / msub["n_dialogues"].max())
            ax.plot(
                msub["turn"],
                msub["mean_confidence"],
                color=MODEL_COLORS[model],
                label=MODEL_DISPLAY[model],
                linewidth=2.0,
                zorder=2,
            )
            ax.scatter(
                msub["turn"],
                msub["mean_confidence"],
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
            ax.set_ylabel("Mean self-reported confidence (1–10)")
        ax.set_ylim(5.5, 10.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.02))
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


def plot_c0_vs_cflip(long_df: pd.DataFrame, stats_df: pd.DataFrame, out_png: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), sharey=True)
    fig.suptitle(
        "Initial confidence (C0) vs confidence at flip (C_flip)",
        fontsize=13,
        weight="bold",
        y=1.02,
    )
    display_order = [MODEL_DISPLAY[m] for m in MODEL_ORDER]

    for ax, (_, dataset_label) in zip(axes, DATASETS):
        sub = long_df[long_df["dataset"] == dataset_label]
        sns.boxplot(
            data=sub,
            x="model_display",
            y="confidence",
            hue="stage",
            order=display_order,
            hue_order=STAGE_ORDER,
            palette=STAGE_PALETTE,
            ax=ax,
            showfliers=False,
            width=0.7,
            linewidth=1.0,
        )
        # light strip
        sns.stripplot(
            data=sub,
            x="model_display",
            y="confidence",
            hue="stage",
            order=display_order,
            hue_order=STAGE_ORDER,
            ax=ax,
            dodge=True,
            alpha=0.12,
            size=1.8,
            palette=STAGE_PALETTE,
            legend=False,
            jitter=0.2,
        )
        ax.set_title(dataset_label, fontsize=12, pad=8)
        ax.set_xlabel("")
        ax.set_ylim(0.5, 10.8)
        if ax is axes[0]:
            ax.set_ylabel("Self-reported confidence (1–10)")
        else:
            ax.set_ylabel("")

        # annotate mean Δ above each model
        stats_sub = stats_df[stats_df["dataset"] == dataset_label].set_index("model")
        for i, name in enumerate(display_order):
            delta = float(stats_sub.loc[name, "mean_delta_Cflip_minus_C0"])
            n_flip = int(stats_sub.loc[name, "n_flip"])
            ax.text(
                i,
                10.55,
                f"Δ={delta:+.2f}\nn={n_flip}",
                ha="center",
                va="top",
                fontsize=7.5,
                color="#333333",
            )

        # one legend only
        if ax is not axes[-1]:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        else:
            ax.legend(title="", loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    fever480_ids = set(pd.read_csv(FEVER480)["original_index"].astype(int))
    by_turn, long_df, stats_df = collect(fever480_ids)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    by_turn.to_csv(CSV_DIR / "mean_confidence_by_turn_all_models.csv", index=False)
    stats_df.to_csv(CSV_DIR / "confidence_c0_cflip_stats.csv", index=False)
    print(f"Wrote {CSV_DIR / 'mean_confidence_by_turn_all_models.csv'}")
    print(f"Wrote {CSV_DIR / 'confidence_c0_cflip_stats.csv'}")
    print(stats_df.to_string(index=False))

    plot_mean_by_turn(by_turn, OUT_DIR / "mean_confidence_by_turn_all_models_all_datasets.png")
    plot_c0_vs_cflip(long_df, stats_df, OUT_DIR / "confidence_c0_vs_cflip_all_models_all_datasets.png")


if __name__ == "__main__":
    main()
