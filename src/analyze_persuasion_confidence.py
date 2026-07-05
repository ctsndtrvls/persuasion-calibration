"""
Compare target self-reported confidence before vs after persuader counterarguments.

Uses column `confidence` (1–10) from expl.csv.
Per dialogue (214): first turn-0 = before persuasion; first turn-1 = after 1st counterargument;
last row = final state.

Example:
  python3 analyze_persuasion_confidence.py
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

PROJECT_ROOT = _PROJECT_ROOT
FEVER_DEEPSEEK_DIR = PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_INPUT = FEVER_DEEPSEEK_DIR / "csv" / "expl.csv"
DEFAULT_OUT_DIR = FEVER_DEEPSEEK_DIR
MAX_TURN = 15
CHANGE_PALETTE = {"Decreased": "#E45756", "Unchanged": "#BAB0AC", "Increased": "#54A24B"}
VERDICT_PALETTE = {
    "SUPPORTS": "#4C78A8",
    "REFUTES": "#E45756",
    "NOT ENOUGH INFO": "#F58518",
}
VERDICT_ORDER = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
ACCURACY_ORDER = ["Correct at t0", "Incorrect at t0"]
ACCURACY_PALETTE = {"Correct at t0": "#4C78A8", "Incorrect at t0": "#F58518"}
LABEL_CODES = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
LABEL_NAMES = {v: k for k, v in LABEL_CODES.items()}
ANSWER_CMAP = ListedColormap(list(VERDICT_PALETTE.values()))


def confidence_change_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=CHANGE_PALETTE["Decreased"], edgecolor="white", label="Decreased (final < turn 0)"),
        Patch(facecolor=CHANGE_PALETTE["Unchanged"], edgecolor="white", label="Unchanged (final = turn 0)"),
        Patch(facecolor=CHANGE_PALETTE["Increased"], edgecolor="white", label="Increased (final > turn 0)"),
    ]


def target_model_label(df: pd.DataFrame) -> str:
    if "target_model" not in df.columns or not df["target_model"].notna().any():
        return "DeepSeek"
    raw = str(df["target_model"].dropna().iloc[0]).lower()
    if "deepseek" in raw:
        return "DeepSeek"
    return str(df["target_model"].dropna().iloc[0])


def persuasion_setup_labels(df: pd.DataFrame) -> tuple[str, str, str]:
    persuader = "unknown persuader"
    if "persuader_model" in df.columns and df["persuader_model"].notna().any():
        persuader = str(df["persuader_model"].dropna().iloc[0]).replace("openai/", "")
    dataset = "FEVER"
    if "dataset" in df.columns and df["dataset"].notna().any():
        dataset = str(df["dataset"].dropna().iloc[0]).upper()
    target = target_model_label(df)
    return f"Persuader: {persuader} · Dataset: {dataset} · Target: {target}", dataset, target


def confidence_change_label(delta: float) -> str:
    if pd.isna(delta):
        return "Unchanged"
    if delta < 0:
        return "Decreased"
    if delta > 0:
        return "Increased"
    return "Unchanged"


def normalize_label(x: object) -> str:
    s = str(x or "").strip().upper()
    if "NOT ENOUGH" in s or s == "NEI":
        return "NOT ENOUGH INFO"
    if "REFUTE" in s:
        return "REFUTES"
    if "SUPPORT" in s:
        return "SUPPORTS"
    return s


def first_turn_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """First occurrence of each turn per dialogue (file order = timeline)."""
    rows = []
    for did, g in df.groupby("dialogue_id", sort=False):
        seen: set[int] = set()
        for _, r in g.iterrows():
            t = int(r["turn"])
            if t in seen:
                continue
            seen.add(t)
            rows.append(
                {
                    "dialogue_id": did,
                    "turn": t,
                    "confidence": pd.to_numeric(r["confidence"], errors="coerce"),
                    "answer": normalize_label(r["answer"]),
                }
            )
    return pd.DataFrame(rows)


def build_turn_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """First snapshot per turn per dialogue: confidence and answer."""
    rows = []
    for did, g in df.groupby("dialogue_id", sort=False):
        seen: set[int] = set()
        prev_answer: str | None = None
        for _, r in g.iterrows():
            t = int(r["turn"])
            if t in seen:
                continue
            seen.add(t)
            answer = normalize_label(r["answer"])
            rows.append(
                {
                    "dialogue_id": did,
                    "original_index": int(r["original_index"]) if pd.notna(r.get("original_index")) else -1,
                    "turn": t,
                    "confidence": pd.to_numeric(r["confidence"], errors="coerce"),
                    "answer": answer,
                    "label_changed": prev_answer is not None and answer != prev_answer,
                }
            )
            prev_answer = answer
    return pd.DataFrame(rows)


def mean_confidence_by_turn(traj: pd.DataFrame) -> pd.DataFrame:
    return (
        traj.groupby("turn", as_index=False)["confidence"]
        .agg(mean_confidence="mean", n_dialogues="count")
        .sort_values("turn")
    )


def dialogue_gold_labels(df: pd.DataFrame) -> pd.DataFrame:
    """One gold label per dialogue from turn-0 rows."""
    t0_rows = df[df["turn"] == 0][["dialogue_id", "gold_label"]].drop_duplicates(
        subset="dialogue_id", keep="first"
    )
    t0_rows["gold_label"] = t0_rows["gold_label"].map(normalize_label)
    return t0_rows


def build_dialogue_confidence(df: pd.DataFrame) -> pd.DataFrame:
    snap = first_turn_snapshot(df)
    t0 = snap[snap["turn"] == 0].rename(
        columns={"confidence": "conf_before", "answer": "answer_before"}
    )[["dialogue_id", "conf_before", "answer_before"]]
    t1 = snap[snap["turn"] == 1].rename(
        columns={"confidence": "conf_after_turn1", "answer": "answer_after_turn1"}
    )[["dialogue_id", "conf_after_turn1", "answer_after_turn1"]]

    last = (
        df.sort_index()
        .groupby("dialogue_id", sort=False)
        .tail(1)[
            ["dialogue_id", "turn", "confidence", "answer", "flipped_from_initial", "flip_turn"]
        ]
        .rename(
            columns={
                "turn": "final_turn",
                "confidence": "conf_final",
                "answer": "answer_final",
            }
        )
    )
    last["conf_final"] = pd.to_numeric(last["conf_final"], errors="coerce")
    last["flipped_from_initial"] = (
        pd.to_numeric(last["flipped_from_initial"], errors="coerce").fillna(0).astype(int)
    )
    last["answer_final"] = last["answer_final"].map(normalize_label)

    out = t0.merge(t1, on="dialogue_id", how="left")
    out = out.merge(last, on="dialogue_id", how="left")
    out = out.merge(dialogue_gold_labels(df), on="dialogue_id", how="left")

    out["answer_before"] = out["answer_before"].map(normalize_label)
    out["correct_t0"] = out["answer_before"] == out["gold_label"]
    out["accuracy_t0"] = np.where(out["correct_t0"], "Correct at t0", "Incorrect at t0")

    out["conf_delta_turn1"] = out["conf_after_turn1"] - out["conf_before"]
    out["conf_delta_final"] = out["conf_final"] - out["conf_before"]
    out["flip_after_turn1"] = (
        out["answer_after_turn1"].notna() & (out["answer_before"] != out["answer_after_turn1"])
    ).astype(int)
    out["flip_final"] = out["flipped_from_initial"]
    return out


def summarize_overall(dlg: pd.DataFrame) -> pd.DataFrame:
    n = len(dlg)
    has_t1 = dlg["conf_after_turn1"].notna()
    d1 = dlg.loc[has_t1, "conf_delta_turn1"]
    dfinal = dlg["conf_delta_final"]
    n_drop = int((d1 < 0).sum()) if len(d1) else 0
    n_rise = int((d1 > 0).sum()) if len(d1) else 0
    n_same = int((d1 == 0).sum()) if len(d1) else 0
    rows = [
        ("n_dialogues", float(n)),
        ("dialogues_with_turn1", float(has_t1.sum())),
        ("n_confidence_decreased_after_1st_counterarg", float(n_drop)),
        ("n_confidence_increased_after_1st_counterarg", float(n_rise)),
        ("n_confidence_unchanged_after_1st_counterarg", float(n_same)),
        ("mean_conf_before", float(dlg["conf_before"].mean())),
        ("mean_conf_after_turn1", float(dlg.loc[has_t1, "conf_after_turn1"].mean())),
        ("mean_conf_final", float(dlg["conf_final"].mean())),
        ("mean_conf_delta_turn1", float(d1.mean()) if len(d1) else float("nan")),
        ("median_conf_delta_turn1", float(d1.median()) if len(d1) else float("nan")),
        ("pct_conf_drop_turn1", float((d1 < 0).mean() * 100) if len(d1) else float("nan")),
        ("pct_conf_rise_turn1", float((d1 > 0).mean() * 100) if len(d1) else float("nan")),
        ("pct_conf_unchanged_turn1", float((d1 == 0).mean() * 100) if len(d1) else float("nan")),
        ("mean_conf_delta_final", float(dfinal.mean())),
        ("median_conf_delta_final", float(dfinal.median())),
        ("pct_conf_drop_final", float((dfinal < 0).mean() * 100)),
        ("flip_after_turn1_pct", float(dlg.loc[has_t1, "flip_after_turn1"].mean() * 100)),
        ("flip_final_pct", float(dlg["flip_final"].mean() * 100)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def summarize_verdict_accuracy(dlg: pd.DataFrame) -> pd.DataFrame:
    """Flip rate and dialogue length by initial verdict × gold accuracy."""
    rows: list[dict[str, object]] = []
    for verdict in VERDICT_ORDER:
        for correct in (True, False):
            sub = dlg[(dlg["answer_before"] == verdict) & (dlg["correct_t0"] == correct)]
            if sub.empty:
                continue
            flipped = sub[sub["flip_final"] == 1]
            flip_turn = pd.to_numeric(flipped["flip_turn"], errors="coerce")
            rows.append(
                {
                    "initial_verdict": verdict,
                    "correct_t0": int(correct),
                    "accuracy_t0": "Correct at t0" if correct else "Incorrect at t0",
                    "n": len(sub),
                    "flip_rate_pct": round(sub["flip_final"].mean() * 100, 1),
                    "mean_final_turn": round(sub["final_turn"].mean(), 2),
                    "median_final_turn": float(sub["final_turn"].median()),
                    "median_flip_turn": float(flip_turn.median()) if len(flipped) else float("nan"),
                }
            )
    for correct in (True, False):
        sub = dlg[dlg["correct_t0"] == correct]
        flipped = sub[sub["flip_final"] == 1]
        flip_turn = pd.to_numeric(flipped["flip_turn"], errors="coerce")
        rows.append(
            {
                "initial_verdict": "ALL",
                "correct_t0": int(correct),
                "accuracy_t0": "Correct at t0" if correct else "Incorrect at t0",
                "n": len(sub),
                "flip_rate_pct": round(sub["flip_final"].mean() * 100, 1),
                "mean_final_turn": round(sub["final_turn"].mean(), 2),
                "median_final_turn": float(sub["final_turn"].median()),
                "median_flip_turn": float(flip_turn.median()) if len(flipped) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


CHANGE_ORDER = ["Decreased", "Unchanged", "Increased"]


def build_confidence_histogram_frame(
    dlg: pd.DataFrame, traj: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-turn confidence with change_dir (final − turn 0) for stacked histograms."""
    paired = dlg.dropna(subset=["conf_before", "conf_final"]).copy()
    paired["change_dir"] = paired["conf_delta_final"].map(confidence_change_label)
    traj_m = traj.merge(paired[["dialogue_id", "change_dir"]], on="dialogue_id", how="inner")
    traj_m = traj_m.dropna(subset=["confidence"]).copy()
    traj_m["confidence"] = pd.to_numeric(traj_m["confidence"], errors="coerce")
    return traj_m, paired


def _plot_confidence_hist_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    title: str,
    show_ylabel: bool = True,
    show_legend: bool = False,
) -> None:
    sns.histplot(
        data=data,
        x="confidence",
        hue="change_dir",
        hue_order=CHANGE_ORDER,
        palette=CHANGE_PALETTE,
        bins=range(1, 12),
        stat="count",
        multiple="stack",
        ax=ax,
        edgecolor="white",
        linewidth=0.35,
        legend=show_legend,
    )
    mean_val = data["confidence"].mean()
    ax.axvline(
        mean_val,
        color="#333333",
        linestyle="--",
        linewidth=1.1,
        label=f"mean={mean_val:.2f}",
    )
    ax.set_title(title, pad=8, fontsize=10)
    ax.set_xlabel("Self-reported confidence (1–10)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Number of claims", fontsize=9)
    else:
        ax.set_ylabel("")
    ax.set_xlim(0.5, 10.5)


def plot_confidence_summary_pack_histogram(
    dlg: pd.DataFrame, traj: pd.DataFrame, df: pd.DataFrame, out_png: Path
) -> None:
    """
    Histogram pack (style of uncertainty_overview panel 1):
    turn 0 full width, turn 1+ with colors = Δ confidence (final − turn 0),
    plus a pooled histogram across all turns.
    """
    sns.set_theme(style="whitegrid")
    traj_m, paired = build_confidence_histogram_frame(dlg, traj)
    _, dataset, target = persuasion_setup_labels(df)
    n = len(paired)

    bottom_turns = [1, 2, 3]
    n_cols = len(bottom_turns) + 1
    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(3, n_cols, height_ratios=[1.15, 1, 1], hspace=0.42, wspace=0.32)

    ax0 = fig.add_subplot(gs[0, :])
    t0 = traj_m[traj_m["turn"] == 0]
    _plot_confidence_hist_panel(
        ax0,
        t0,
        title="Turn 0 — before persuasion",
        show_legend=False,
    )
    if ax0.get_legend() is not None:
        ax0.get_legend().remove()

    for col, turn in enumerate(bottom_turns):
        ax = fig.add_subplot(gs[1, col])
        sub = traj_m[traj_m["turn"] == turn]
        _plot_confidence_hist_panel(
            ax,
            sub,
            title=f"Turn {turn}",
            show_ylabel=(col == 0),
        )

    ax_final = fig.add_subplot(gs[1, len(bottom_turns)])
    final_df = paired[["conf_final", "change_dir"]].rename(columns={"conf_final": "confidence"})
    final_df = final_df.dropna(subset=["confidence"])
    _plot_confidence_hist_panel(
        ax_final,
        final_df,
        title="Final (last turn per claim)",
        show_ylabel=False,
    )

    ax_all = fig.add_subplot(gs[2, :])
    n_obs = len(traj_m)
    turn_min = int(traj_m["turn"].min())
    turn_max = int(traj_m["turn"].max())
    _plot_confidence_hist_panel(
        ax_all,
        traj_m,
        title=f"All turns in one (turns {turn_min}–{turn_max}, n={n_obs} observations)",
        show_ylabel=True,
    )
    if ax_all.get_legend() is not None:
        ax_all.get_legend().remove()

    fig.suptitle(
        f"Self-reported confidence before and after persuasion — {dataset} · {target} (n={n})",
        fontsize=12,
        y=0.99,
    )
    fig.legend(
        handles=confidence_change_legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        title="Bar colors: change in confidence from turn 0 to final turn",
        title_fontsize=9,
    )
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_change_boxplot(dlg: pd.DataFrame, df: pd.DataFrame, out_png: Path) -> None:
    """Boxplot of confidence by stage, colored by direction of change (final vs turn 0)."""
    sns.set_theme(style="whitegrid")
    paired = dlg.dropna(subset=["conf_before", "conf_final"]).copy()
    paired["change_dir"] = paired["conf_delta_final"].map(confidence_change_label)
    stage_map = {
        "conf_before": "Turn 0\n(before)",
        "conf_after_turn1": "Turn 1\n(1st counterarg)",
        "conf_final": "Final",
    }
    long = paired.melt(
        id_vars=["dialogue_id", "change_dir"],
        value_vars=["conf_before", "conf_after_turn1", "conf_final"],
        var_name="stage_key",
        value_name="confidence",
    )
    long = long.dropna(subset=["confidence"])
    long["stage"] = long["stage_key"].map(stage_map)
    stage_order = list(stage_map.values())

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=long,
        x="stage",
        y="confidence",
        hue="change_dir",
        order=stage_order,
        hue_order=CHANGE_ORDER,
        ax=ax,
        palette=CHANGE_PALETTE,
        linewidth=1.0,
        fliersize=2,
    )
    sns.stripplot(
        data=long,
        x="stage",
        y="confidence",
        hue="change_dir",
        order=stage_order,
        hue_order=CHANGE_ORDER,
        ax=ax,
        palette=CHANGE_PALETTE,
        dodge=True,
        alpha=0.35,
        size=2.5,
        jitter=0.18,
        legend=False,
    )
    ax.set_ylim(1, 10.5)
    ax.set_xlabel("Persuasion stage")
    ax.set_ylabel("Self-reported confidence (1–10)")
    ax.legend(title="Δ confidence\n(final − turn 0)", loc="lower right", fontsize=9)
    _, dataset, target = persuasion_setup_labels(df)
    ax.set_title(
        f"Self-reported confidence by persuasion stage — {dataset} · {target}",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _annotate_verdict_turn_boxplot(ax: plt.Axes, data: pd.DataFrame, verdict_col: str) -> None:
    """Boxplot of dialogue length (turns) by verdict with mean/median labels."""
    plot_df = data.dropna(subset=[verdict_col, "final_turn"]).copy()
    plot_df["final_turn"] = plot_df["final_turn"].astype(int)
    plot_df[verdict_col] = plot_df[verdict_col].map(normalize_label)
    sns.boxplot(
        data=plot_df,
        x=verdict_col,
        y="final_turn",
        order=VERDICT_ORDER,
        hue=verdict_col,
        palette=VERDICT_PALETTE,
        ax=ax,
        linewidth=1.0,
        fliersize=2,
        legend=False,
    )
    sns.stripplot(
        data=plot_df,
        x=verdict_col,
        y="final_turn",
        order=VERDICT_ORDER,
        ax=ax,
        color="#333333",
        alpha=0.25,
        size=2.5,
        jitter=0.22,
    )
    ymax = max(plot_df["final_turn"].max(), 1)
    for i, verdict in enumerate(VERDICT_ORDER):
        sub = plot_df[plot_df[verdict_col] == verdict]["final_turn"]
        if sub.empty:
            continue
        ax.text(
            i,
            ymax + 0.35,
            f"n={len(sub)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
    ax.set_ylim(-0.5, ymax + 1.2)
    ax.set_xticks(range(len(VERDICT_ORDER)))
    ax.set_xticklabels(["SUPPORTS", "REFUTES", "NEI"], fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Dialogue length (last turn reached)")


def plot_flip_turn_dynamics(dlg: pd.DataFrame, traj: pd.DataFrame, df: pd.DataFrame, out_png: Path) -> None:
    """When the target flips verdict + how many turns each verdict label required."""
    del traj  # not used in this figure
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.38)

    # ── Top: bar chart of flip turn (flipped claims only) ─────────────────
    ax1 = fig.add_subplot(gs[0, :])
    flipped = dlg[dlg["flip_final"] == 1].copy()
    flip_turn = pd.to_numeric(flipped["flip_turn"], errors="coerce").dropna().astype(int)
    flip_counts = flip_turn.value_counts().sort_index()
    bars = ax1.bar(
        flip_counts.index.astype(int),
        flip_counts.values,
        color="#E45756",
        edgecolor="white",
        linewidth=0.6,
    )
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8, str(int(h)), ha="center", fontsize=8)
    n_never = int((dlg["flip_final"] == 0).sum())
    ax1.set_xlim(1.5, MAX_TURN + 0.5)
    ax1.set_xticks(range(2, MAX_TURN + 1))
    ax1.set_xlabel("Turn when verdict first changed")
    ax1.set_ylabel("Number of claims")
    ax1.set_title(
        f"Verdict flip by persuasion turn  "
        f"({len(flipped)} flipped, {n_never} never flipped by end)",
        fontsize=11,
    )

    # ── Bottom: dialogue length by initial verdict (turn 0) ─────────────
    view = dlg.copy()
    view["final_turn"] = pd.to_numeric(view["final_turn"], errors="coerce")

    ax2 = fig.add_subplot(gs[1, 0])
    _annotate_verdict_turn_boxplot(ax2, view, "answer_before")
    ax2.set_title(
        "By initial verdict (turn 0): how many persuasion turns were needed?",
        fontsize=10,
        pad=8,
    )

    _, dataset, target = persuasion_setup_labels(df)
    fig.suptitle(
        f"When does the target change its verdict? — {dataset} · {target}",
        fontsize=13,
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Top: turn when the verdict first changed (flipped claims only). "
        "Bottom: dialogue length by starting verdict (turn 0). "
        "n = number of claims per group.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(bottom=0.09)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _verdict_accuracy_label(row: pd.Series) -> str:
    short = {"SUPPORTS": "SUP", "REFUTES": "REF", "NOT ENOUGH INFO": "NEI"}
    acc = "ok" if row["correct_t0"] else "err"
    return f"{short[row['answer_before']]} {acc}"


def plot_flip_by_verdict_and_accuracy(dlg: pd.DataFrame, df: pd.DataFrame, out_png: Path) -> None:
    """Synthesis: initial verdict × gold accuracy → flip rate and dialogue length."""
    sns.set_theme(style="whitegrid")
    view = dlg.dropna(subset=["answer_before", "final_turn", "gold_label"]).copy()
    view["final_turn"] = pd.to_numeric(view["final_turn"], errors="coerce")
    view["group_label"] = view.apply(_verdict_accuracy_label, axis=1)

    group_order = [
        _verdict_accuracy_label(pd.Series({"answer_before": v, "correct_t0": c}))
        for v in VERDICT_ORDER
        for c in (True, False)
        if not view[(view["answer_before"] == v) & (view["correct_t0"] == c)].empty
    ]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.15], hspace=0.38)

    # ── Top: flip rate by verdict × accuracy ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    cell_stats = summarize_verdict_accuracy(view)
    cell_stats = cell_stats[cell_stats["initial_verdict"] != "ALL"].copy()
    cell_stats["group_label"] = cell_stats.apply(
        lambda r: _verdict_accuracy_label(
            pd.Series({"answer_before": r["initial_verdict"], "correct_t0": bool(r["correct_t0"])})
        ),
        axis=1,
    )
    cell_stats = cell_stats.set_index("group_label").reindex(group_order).reset_index()
    x_pos = np.arange(len(group_order))
    colors = [
        VERDICT_PALETTE[row["initial_verdict"]]
        if row["accuracy_t0"] == "Correct at t0"
        else "#BAB0AC"
        for _, row in cell_stats.iterrows()
    ]
    bars = ax1.bar(x_pos, cell_stats["flip_rate_pct"], color=colors, edgecolor="white", linewidth=0.6)
    for bar, (_, row) in zip(bars, cell_stats.iterrows()):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{row['flip_rate_pct']:.0f}%\nn={int(row['n'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(group_order, fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Flip rate (%)")
    ax1.set_title(
        "By initial verdict × gold accuracy  (ok = correct at t0, err = incorrect; solid = correct, gray = incorrect)",
        fontsize=10,
        pad=8,
    )

    # ── Bottom: dialogue length by verdict × accuracy ───────────────────
    ax2 = fig.add_subplot(gs[1, :])
    plot_df = view.copy()
    plot_df["answer_before"] = pd.Categorical(plot_df["answer_before"], categories=VERDICT_ORDER, ordered=True)
    sns.boxplot(
        data=plot_df,
        x="answer_before",
        y="final_turn",
        hue="accuracy_t0",
        hue_order=ACCURACY_ORDER,
        order=VERDICT_ORDER,
        palette=ACCURACY_PALETTE,
        ax=ax2,
        linewidth=1.0,
        fliersize=2,
    )
    sns.stripplot(
        data=plot_df,
        x="answer_before",
        y="final_turn",
        hue="accuracy_t0",
        hue_order=ACCURACY_ORDER,
        order=VERDICT_ORDER,
        ax=ax2,
        dodge=True,
        palette=["#333333", "#333333"],
        alpha=0.2,
        size=2.2,
        jitter=0.18,
        legend=False,
    )
    handles, labels = ax2.get_legend_handles_labels()
    legend_labels = ["Correct at turn 0", "Incorrect at turn 0"]
    ax2.legend(handles[:2], legend_labels[: len(handles[:2])], loc="upper right", fontsize=9)
    ax2.set_xlabel("")
    ax2.set_ylabel("Dialogue length (last turn reached)")
    ymax = max(plot_df["final_turn"].max(), 1)
    ax2.set_ylim(-0.5, ymax + 1.2)
    ax2.set_title(
        "Dialogue length by initial verdict × gold accuracy",
        fontsize=10,
        pad=8,
    )

    _, dataset, target = persuasion_setup_labels(df)
    fig.suptitle(
        f"Verdict stance × accuracy — {dataset} · {target}",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mean_confidence_by_turn(traj: pd.DataFrame, df: pd.DataFrame, out_png: Path) -> None:
    """Mean self-reported confidence at each turn (points + line)."""
    sns.set_theme(style="whitegrid")
    by_turn = mean_confidence_by_turn(traj)
    by_turn = by_turn[by_turn["turn"] <= MAX_TURN]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        by_turn["turn"],
        by_turn["mean_confidence"],
        color="#4C78A8",
        linewidth=1.5,
        zorder=1,
    )
    ax.scatter(
        by_turn["turn"],
        by_turn["mean_confidence"],
        s=by_turn["n_dialogues"] * 1.8,
        c="#4C78A8",
        edgecolors="white",
        linewidths=0.6,
        zorder=2,
    )
    for _, r in by_turn.iterrows():
        ax.annotate(
            f"{r['mean_confidence']:.2f}",
            (r["turn"], r["mean_confidence"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            color="#333333",
        )
    ax.set_xlim(-0.3, MAX_TURN + 0.3)
    ax.set_xticks(range(0, MAX_TURN + 1))
    ax.set_ylim(7.5, 9.6)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean self-reported confidence")
    _, dataset, target = persuasion_setup_labels(df)
    ax.set_title(
        f"Mean confidence across claims by turn — {dataset} · {target}",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_pack(
    dlg: pd.DataFrame,
    df: pd.DataFrame,
    traj: pd.DataFrame,
    out_png: Path,
) -> None:
    """Write confidence_summary_pack.png + auxiliary confidence/flip figures."""
    plot_confidence_change_boxplot(dlg, df, out_png.parent / "confidence_change_boxplot.png")
    plot_flip_turn_dynamics(dlg, traj, df, out_png.parent / "flip_turn_by_dialogue.png")
    plot_flip_by_verdict_and_accuracy(dlg, df, out_png.parent / "flip_by_verdict_and_accuracy.png")
    plot_mean_confidence_by_turn(traj, df, out_png.parent / "mean_confidence_by_turn.png")
    plot_confidence_summary_pack_histogram(dlg, traj, df, out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Confidence before/after persuasion analysis.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    traj = build_turn_trajectories(df)
    dlg = build_dialogue_confidence(df)
    summary = summarize_overall(dlg)
    verdict_accuracy = summarize_verdict_accuracy(dlg)

    csv_dir = args.out_dir / "csv"
    png_dir = args.out_dir / "png"
    csv_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    out_dlg = csv_dir / "confidence_dialogues.csv"
    out_summary = csv_dir / "confidence_summary.csv"
    out_by_turn = csv_dir / "confidence_by_turn.csv"
    out_verdict_accuracy = csv_dir / "flip_verdict_accuracy_summary.csv"
    out_png = png_dir / "confidence_summary_pack.png"
    out_box = png_dir / "confidence_change_boxplot.png"
    out_flip = png_dir / "flip_turn_by_dialogue.png"
    out_flip_accuracy = png_dir / "flip_by_verdict_and_accuracy.png"
    out_mean = png_dir / "mean_confidence_by_turn.png"

    dlg.to_csv(out_dlg, index=False)
    summary.to_csv(out_summary, index=False)
    verdict_accuracy.to_csv(out_verdict_accuracy, index=False)
    mean_confidence_by_turn(traj).to_csv(out_by_turn, index=False)
    plot_confidence_pack(dlg, df, traj, out_png)

    print(f"Saved dialogue table: {out_dlg}")
    print(f"Saved summary: {out_summary}")
    print(f"Saved verdict×accuracy summary: {out_verdict_accuracy}")
    print(f"Saved by-turn means: {out_by_turn}")
    print(f"Saved figure: {out_png}")
    print(f"Saved figure: {out_box}")
    print(f"Saved figure: {out_flip}")
    print(f"Saved figure: {out_flip_accuracy}")
    print(f"Saved figure: {out_mean}")
    print("\nKey metrics:")
    for _, r in summary.iterrows():
        print(f"  {r['metric']}: {r['value']}")


if __name__ == "__main__":
    main()
