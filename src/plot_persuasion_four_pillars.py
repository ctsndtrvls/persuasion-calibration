"""
Overview across thesis directions: calibration, confidence, argument quality,
and persuasion dynamics (accuracy before/after persuasion, flip rate, turns).

Example:
  cd src && python3 plot_persuasion_four_pillars.py
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
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

from persuasion_arg_quality import TOP_LEVEL_QUALITY_DIMENSIONS
from plot_ece_calibration import bin_edges, compute_bin_stats, ece_from_bins
from plot_persuasion_fever_summary import prepare_dialogue_view

FEVER_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_EXPL = FEVER_DIR / "csv" / "expl.csv"
DEFAULT_JUDGE = (
    FEVER_DIR
    / "arg_quality"
    / "arg_quality_long_fever214_all_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv"
)
MAX_TURN = 15

CORE_KEYS = ["calibration", "confidence", *TOP_LEVEL_QUALITY_DIMENSIONS]
CORE_BAR_LABELS = [
    "Calibration",
    "Confidence",
    "Cogency",
    "Effectiveness",
    "Reasonableness",
]
STAGE_PALETTE = {"Turn 0": "#4C78A8", "Final": "#E45756"}


def ece_for_rows(conf_prob: np.ndarray, correct: np.ndarray, edges: np.ndarray) -> float:
    _, acc_pct, mean_conf, counts = compute_bin_stats(conf_prob, correct, edges)
    return float(ece_from_bins(mean_conf, acc_pct, counts))


def appropriate_response(correct_t0: pd.Series, flip: pd.Series) -> float:
    ok = (correct_t0 & ~flip.astype(bool)) | (~correct_t0 & flip.astype(bool))
    return float(ok.mean())


def composite_score(row: pd.Series) -> float:
    """Unweighted mean of calibration, confidence, and top-3 arg-quality dims (all 0–1)."""
    return float(np.mean([row[k] for k in CORE_KEYS]))


def mean_quality_dims(judge: pd.DataFrame, *, turn: int | None = None) -> dict[str, float]:
    subset = judge if turn is None else judge[judge["turn"] == turn]
    if subset.empty:
        return {d: float("nan") for d in TOP_LEVEL_QUALITY_DIMENSIONS}
    return {d: float(subset[d].mean() / 3.0) for d in TOP_LEVEL_QUALITY_DIMENSIONS}


def build_scores(view: pd.DataFrame, judge: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    edges = bin_edges()
    flip = view["flip"].astype(bool)
    correct_t0 = view["correct_t0"].astype(bool)
    final_turn = pd.to_numeric(view["final_turn"], errors="coerce")
    flip_turn = pd.to_numeric(view["flip_turn"], errors="coerce")

    conf_t0 = (pd.to_numeric(view["conf_t0"], errors="coerce") / 10.0).to_numpy(dtype=float)
    conf_final = (pd.to_numeric(view["conf_final"], errors="coerce") / 10.0).to_numpy(dtype=float)
    cor_t0 = view["correct_t0"].to_numpy(dtype=float)
    cor_final = view["correct_final"].to_numpy(dtype=float)

    ece_t0 = ece_for_rows(conf_t0, cor_t0, edges)
    ece_final = ece_for_rows(conf_final, cor_final, edges)

    accuracy_t0 = float(correct_t0.mean())
    appropriate = appropriate_response(correct_t0, flip)
    flip_rate = float(flip.mean())
    mean_turns = float(final_turn.mean())
    median_flip_turn = float(flip_turn[flip].median()) if flip.any() else float("nan")

    core = pd.DataFrame(
        [
            {
                "stage": "Turn 0",
                "calibration": 1.0 - ece_t0,
                "confidence": float(pd.to_numeric(view["conf_t0"], errors="coerce").mean() / 10.0),
                **mean_quality_dims(judge, turn=1),
                "ece": ece_t0,
                "n_dialogues": len(view),
            },
            {
                "stage": "Final",
                "calibration": 1.0 - ece_final,
                "confidence": float(pd.to_numeric(view["conf_final"], errors="coerce").mean() / 10.0),
                **mean_quality_dims(judge),
                "ece": ece_final,
                "n_dialogues": len(view),
            },
        ]
    )
    core["composite_score"] = core.apply(composite_score, axis=1)

    persuasion = {
        "accuracy_t0": accuracy_t0,
        "appropriate_response": appropriate,
        "flip_rate": flip_rate,
        "mean_dialogue_turns": mean_turns,
        "median_flip_turn": median_flip_turn,
        "mean_turns_norm": mean_turns / MAX_TURN,
        "median_flip_turn_norm": median_flip_turn / MAX_TURN if not np.isnan(median_flip_turn) else float("nan"),
    }
    return core, persuasion


def _draw_group_bracket(ax: plt.Axes, x_start: float, x_end: float, y: float, label: str) -> None:
    """Light bracket + label spanning a group of bars."""
    mid = (x_start + x_end) / 2
    tick_h = 0.012
    ax.plot([x_start, x_start, x_end, x_end], [y, y + tick_h, y + tick_h, y], color="#3D5A40", linewidth=1.0, clip_on=False)
    ax.text(mid, y + tick_h + 0.015, label, ha="center", va="bottom", fontsize=9.5, color="#3D5A40", fontweight="medium")


def plot_overview(
    core: pd.DataFrame,
    persuasion: dict[str, float],
    dataset: str,
    target: str,
    out_png: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 0.95], hspace=0.38)

    values_by_stage = {row["stage"]: [row[k] for k in CORE_KEYS] for _, row in core.iterrows()}

    ax_c = fig.add_subplot(gs[0, 0])
    x = np.arange(len(CORE_KEYS))
    width = 0.34

    # Visual group: argument quality dimensions (cogency, effectiveness, reasonableness)
    aq_x0, aq_x1 = x[2] - 0.52, x[4] + 0.52
    ax_c.add_patch(
        Rectangle(
            (aq_x0, 0),
            aq_x1 - aq_x0,
            1.08,
            facecolor="#EEF6EE",
            edgecolor="#B7D4BC",
            linewidth=1.0,
            zorder=0,
        )
    )

    for i, (stage, color) in enumerate(STAGE_PALETTE.items()):
        vals = values_by_stage[stage]
        bars = ax_c.bar(x + (i - 0.5) * width, vals, width, label=stage, color=color, edgecolor="white", zorder=2)
        for bar, val in zip(bars, vals):
            ax_c.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=3,
            )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(CORE_BAR_LABELS, fontsize=9)
    ax_c.set_ylim(0, 1.08)
    ax_c.set_ylabel("Score (0–1)")
    _draw_group_bracket(ax_c, aq_x0 + 0.08, aq_x1 - 0.08, 1.045, "Argument quality")
    ax_c.legend(fontsize=9, loc="upper left")

    ax_p = fig.add_subplot(gs[1, 0])
    pers_labels = [
        "Accuracy\n(before persuasion,\nturn 0)",
        "Accuracy\n(after persuasion)",
        "Flip rate",
        "Mean dialogue\nlength (turns)",
        "Median flip\nturn (flipped only)",
    ]
    pers_values = [
        persuasion["accuracy_t0"],
        persuasion["appropriate_response"],
        persuasion["flip_rate"],
        persuasion["mean_turns_norm"],
        persuasion["median_flip_turn_norm"],
    ]
    pers_annotations = [
        f"{persuasion['accuracy_t0'] * 100:.0f}%",
        f"{persuasion['appropriate_response'] * 100:.0f}%",
        f"{persuasion['flip_rate'] * 100:.0f}%",
        f"{persuasion['mean_dialogue_turns']:.1f} turns",
        f"{persuasion['median_flip_turn']:.0f} turns",
    ]
    colors = ["#4C78A8", "#E45756", "#F58518", "#72B7B2", "#B279A2"]
    x_p = np.arange(len(pers_labels))
    bars = ax_p.bar(x_p, pers_values, color=colors, edgecolor="white", width=0.62)
    for bar, val, ann in zip(bars, pers_values, pers_annotations):
        ax_p.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            ann,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="medium",
        )
    ax_p.set_xticks(x_p)
    ax_p.set_xticklabels(pers_labels, fontsize=9)
    ax_p.set_ylim(0, 1.08)
    ax_p.set_ylabel("Scale 0–1 (turns shown as turns / 15)")
    ax_p.set_title("Persuasion dynamics (end of experiment)", fontsize=11, pad=10)

    n = int(core["n_dialogues"].iloc[0])
    fig.suptitle(f"Final overview — {dataset} · {target} (n={n})", fontsize=13, y=0.98)
    fig.subplots_adjust(top=0.9, bottom=0.1)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_composite_score(
    core: pd.DataFrame,
    dataset: str,
    target: str,
    out_png: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    stages = list(core["stage"])
    values = core["composite_score"].to_numpy(dtype=float)
    colors = [STAGE_PALETTE[s] for s in stages]
    x = np.arange(len(stages))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="medium",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Composite score (0–1)")
    n = int(core["n_dialogues"].iloc[0])
    ax.set_title(
        f"Composite score — {dataset} · {target} (n={n})\n"
        "Unweighted mean of calibration, confidence, cogency, effectiveness, reasonableness",
        fontsize=11,
        pad=12,
    )

    component_lines = []
    for _, row in core.iterrows():
        parts = ", ".join(f"{label} {row[key]:.2f}" for label, key in zip(CORE_BAR_LABELS, CORE_KEYS))
        component_lines.append(f"{row['stage']}: {parts}")
    fig.text(0.5, 0.01, "\n".join(component_lines), ha="center", va="bottom", fontsize=8.5, color="#444444")

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def target_label(df: pd.DataFrame) -> str:
    if "target_model" not in df.columns or not df["target_model"].notna().any():
        return "DeepSeek"
    raw = str(df["target_model"].dropna().iloc[0]).lower()
    return "DeepSeek" if "deepseek" in raw else str(df["target_model"].dropna().iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot thesis overview (pillars + persuasion dynamics).")
    ap.add_argument("--expl", type=Path, default=DEFAULT_EXPL)
    ap.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--out-dir", type=Path, default=FEVER_DIR)
    args = ap.parse_args()

    expl = pd.read_csv(args.expl)
    judge = pd.read_csv(args.judge)
    view = prepare_dialogue_view(expl)
    core, persuasion = build_scores(view, judge)

    dataset = "FEVER"
    if "dataset" in expl.columns and expl["dataset"].notna().any():
        dataset = str(expl["dataset"].dropna().iloc[0]).upper()
    target = target_label(expl)

    csv_dir = args.out_dir / "csv"
    png_dir = args.out_dir / "png"
    csv_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    out_core = csv_dir / "four_pillars_scores.csv"
    out_pers = csv_dir / "persuasion_dynamics_scores.csv"
    out_composite = csv_dir / "composite_score.csv"
    out_png = png_dir / "final_overview.png"
    out_composite_png = png_dir / "composite_score.png"

    core.to_csv(out_core, index=False)
    core[["stage", "composite_score", *CORE_KEYS, "n_dialogues"]].to_csv(out_composite, index=False)
    pd.DataFrame([persuasion]).to_csv(out_pers, index=False)
    plot_overview(core, persuasion, dataset, target, out_png)
    plot_composite_score(core, dataset, target, out_composite_png)

    print(f"Saved core pillars: {out_core}")
    print(f"Saved composite score: {out_composite}")
    print(f"Saved persuasion dynamics: {out_pers}")
    print(f"Saved figure: {out_png}")
    print(f"Saved composite figure: {out_composite_png}")
    print("\nCore pillars:")
    print(core.to_string(index=False))
    print("\nComposite score:")
    print(core[["stage", "composite_score"]].to_string(index=False))
    print("\nPersuasion dynamics:")
    for k, v in persuasion.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
