"""
Single overview figure: confidence + epistemic markers + before/after comparison.

FEVER persuasion / DeepSeek target (expl.csv, 214 dialogues).

Output: output_wood/persuasion/DeepSeek/fever/png/uncertainty_overview.png
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
import pandas as pd
import seaborn as sns

from analyze_persuasion_confidence import build_dialogue_confidence
from analyze_persuasion_epistemic_calibration import (
    MARKER_COUNT_COL,
    MARKER_LIST_COL,
    add_epistemic_marker_columns,
    build_dialogue_epistemic,
    summarize_top_markers,
)
from linguistic_confidence import UNCERTAINTY_MARKERS, extract_epistemic_markers

FEVER_DEEPSEEK_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_INPUT = FEVER_DEEPSEEK_DIR / "csv" / "expl.csv"
HIGH_CONF_THRESHOLD = 8


def build_dialogue_epistemic_at_t0(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_epistemic_marker_columns(df)
    snap = (
        enriched[enriched["turn"] == 0]
        .drop_duplicates(subset="dialogue_id", keep="first")
        .copy()
    )
    snap["high_conf"] = pd.to_numeric(snap["confidence"], errors="coerce") >= HIGH_CONF_THRESHOLD
    snap["has_uncertainty_marker"] = snap[MARKER_LIST_COL].map(
        lambda s: bool(set(str(s).split("|")) & UNCERTAINTY_MARKERS) if str(s).strip() else False
    )
    return snap[["dialogue_id", "high_conf", "has_uncertainty_marker", MARKER_COUNT_COL]]


def plot_overview(df: pd.DataFrame, out_png: Path) -> None:
    dlg_conf = build_dialogue_confidence(df)
    dlg_epi = build_dialogue_epistemic_at_t0(df)
    dlg = dlg_conf.merge(dlg_epi, on="dialogue_id", how="left")

    paired = dlg.dropna(subset=["conf_after_turn1"])
    delta = paired["conf_delta_turn1"]
    n_drop = int((delta < 0).sum())
    n_same = int((delta == 0).sum())
    n_rise = int((delta > 0).sum())
    enriched = add_epistemic_marker_columns(df)
    n_rows = len(enriched)
    n_marker_rows = int((enriched[MARKER_COUNT_COL] > 0).sum())
    pct_marker_rows = 100.0 * n_marker_rows / n_rows if n_rows else 0.0
    dlg_epi_all = build_dialogue_epistemic(enriched)
    n_marker_dlg = int(dlg_epi_all["has_any_marker"].sum())
    n_dlg = len(dlg_epi_all)
    pct_marker_dlg = 100.0 * n_marker_dlg / n_dlg if n_dlg else 0.0
    top = summarize_top_markers(enriched, top_n=3)
    top_marker_names = (
        ", ".join(top["marker"].tolist()) if not top.empty else "—"
    )

    high_conf = dlg["high_conf"].fillna(False)
    n_miscal_t1 = int((high_conf & (dlg["flip_after_turn1"] == 1)).sum())
    n_miscal_final = int((high_conf & (dlg["flip_final"] == 1)).sum())
    n_mismatch = int((high_conf & dlg["has_uncertainty_marker"].fillna(False)).sum())

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.15, 1.1], hspace=0.58, wspace=0.3)

    # ── (1) Self-reported confidence ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    conf_t0 = dlg["conf_before"].dropna()
    sns.histplot(conf_t0, bins=range(1, 12), ax=ax1, color="#4C78A8", stat="count")
    ax1.set_title(
        "(1) Self-reported confidence (column: confidence, scale 1–10)",
        pad=10,
    )
    ax1.set_xlabel("Confidence at turn 0 (before persuasion)", labelpad=8)
    ax1.set_ylabel("Number of claims")
    ax1.axvline(conf_t0.mean(), color="#E45756", linestyle="--", label=f"mean={conf_t0.mean():.2f}")
    ax1.legend(fontsize=8)

    # ── (2) Epistemic markers + calibration ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    cal_df = pd.DataFrame(
        {
            "pattern": [
                "High conf @ start\n+ flip after 1st counterarg",
                "High conf @ start\n+ flip by end of dialogue",
                "High conf @ start\n+ hedge in explanation",
            ],
            "count": [n_miscal_t1, n_miscal_final, n_mismatch],
        }
    )
    sns.barplot(
        data=cal_df,
        x="count",
        y="pattern",
        hue="pattern",
        ax=ax2,
        palette=["#E45756", "#F58518", "#9ECAE9"],
        legend=False,
        orient="h",
    )
    ax2.set_title("(2) Epistemic markers in decision_explanation", fontsize=11, pad=14)
    ax2.text(
        0.01,
        0.97,
        f"Log rows with markers: {n_marker_rows}/{n_rows} ({pct_marker_rows:.1f}%)\n"
        f"Dialogues with markers: {n_marker_dlg}/{n_dlg} ({pct_marker_dlg:.1f}%)\n"
        f"Top epistemic markers: {top_marker_names}",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax2.set_xlabel("Number of claims", labelpad=6)
    ax2.set_ylabel("")

    # ── (3) Before vs after 1st counterargument ─────────────────────────
    ax3a = fig.add_subplot(gs[2, 0])
    ax3b = fig.add_subplot(gs[2, 1])
    counts_df = pd.DataFrame(
        {
            "outcome": ["Decreased", "Unchanged", "Increased"],
            "count": [n_drop, n_same, n_rise],
        }
    )
    palette = {"Decreased": "#E45756", "Unchanged": "#BAB0AC", "Increased": "#54A24B"}
    sns.barplot(
        data=counts_df,
        x="outcome",
        y="count",
        hue="outcome",
        ax=ax3a,
        palette=palette,
        legend=False,
    )
    ax3a.set_title("(3) Confidence before vs after 1st counterargument")
    ax3a.set_xlabel("")
    ax3a.set_ylabel("Number of claims")
    for i, row in counts_df.iterrows():
        ax3a.text(i, row["count"] + 2, str(int(row["count"])), ha="center", fontsize=11)

    long = paired.melt(
        id_vars="dialogue_id",
        value_vars=["conf_before", "conf_after_turn1"],
        var_name="stage",
        value_name="confidence",
    )
    long["stage"] = long["stage"].map(
        {
            "conf_before": "Before",
            "conf_after_turn1": "After 1st counterarg",
        }
    )
    order = ["Before", "After 1st counterarg"]
    sns.boxplot(
        data=long,
        x="stage",
        y="confidence",
        hue="stage",
        order=order,
        hue_order=order,
        ax=ax3b,
        palette=["#72B7B2", "#E45756"],
        legend=False,
    )
    ax3b.set_title("Distribution: before vs after")
    ax3b.set_ylim(1, 10)
    ax3b.set_xlabel("")

    fig.subplots_adjust(top=0.97, bottom=0.06, hspace=0.58)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Plot persuasion uncertainty overview.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=FEVER_DEEPSEEK_DIR)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    png_dir = args.out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    out_png = png_dir / "uncertainty_overview.png"
    plot_overview(df, out_png)
    print(f"Saved overview figure: {out_png}")


if __name__ == "__main__":
    main()
