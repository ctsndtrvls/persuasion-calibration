"""
Joint analysis: target self-reported confidence vs persuader argument quality.

Example:
  python3 analyze_confidence_arg_quality_joint.py
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
from scipy import stats

from persuasion_arg_quality import TOP_LEVEL_QUALITY_DIMENSIONS
from plot_ece_calibration import bin_edges, ece_from_bins, compute_bin_stats

FEVER_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_EXPL = FEVER_DIR / "csv" / "expl.csv"
DEFAULT_JUDGE = (
    FEVER_DIR
    / "arg_quality"
    / "arg_quality_long_fever214_all_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv"
)
MAX_TURN = 15
DIM_COLORS = {
    "cogency": "#E45756",
    "effectiveness": "#F58518",
    "reasonableness": "#54A24B",
}


def normalize_label(x: object) -> str:
    s = str(x or "").strip().upper()
    if "NOT ENOUGH" in s or s == "NEI":
        return "NOT ENOUGH INFO"
    if "REFUTE" in s:
        return "REFUTES"
    if "SUPPORT" in s:
        return "SUPPORTS"
    return s


def first_turn_snapshots(expl: pd.DataFrame) -> pd.DataFrame:
    return (
        expl.sort_index()
        .drop_duplicates(subset=["dialogue_id", "turn"], keep="first")
        .copy()
    )


def build_turn_pairs(expl: pd.DataFrame, judge: pd.DataFrame) -> pd.DataFrame:
    snap = first_turn_snapshots(expl)
    snap["confidence"] = pd.to_numeric(snap["confidence"], errors="coerce")
    snap["correct"] = (
        snap["answer"].map(normalize_label) == snap["gold_label"].map(normalize_label)
    ).astype(int)

    pairs = judge.merge(
        snap[["dialogue_id", "turn", "confidence", "correct", "answer", "gold_label"]],
        on=["dialogue_id", "turn"],
        how="inner",
    )
    pairs["conf_prob"] = pairs["confidence"] / 10.0
    return pairs


def summarize_by_turn(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for turn, g in pairs.groupby("turn"):
        row = {
            "turn": int(turn),
            "n": len(g),
            "mean_confidence": float(g["confidence"].mean()),
        }
        for d in TOP_LEVEL_QUALITY_DIMENSIONS:
            if d in g.columns:
                row[f"mean_{d}"] = float(g[d].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("turn")


def build_ece_frame(expl: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    snap = first_turn_snapshots(expl)
    t0 = snap[snap["turn"] == 0].copy()
    t0["confidence"] = pd.to_numeric(t0["confidence"], errors="coerce")
    t0["correct"] = (
        t0["answer"].map(normalize_label) == t0["gold_label"].map(normalize_label)
    ).astype(int)
    t0["conf_prob"] = t0["confidence"] / 10.0
    t0["ece_group"] = "Turn 0 (before persuasion)"

    during = pairs[pairs["turn"] >= 1].copy()
    during["ece_group"] = "Turns 1+ (after counterargument)"

    final = (
        snap.sort_values(["dialogue_id", "turn"])
        .groupby("dialogue_id", as_index=False)
        .tail(1)
    )
    final["confidence"] = pd.to_numeric(final["confidence"], errors="coerce")
    final["correct"] = (
        final["answer"].map(normalize_label) == final["gold_label"].map(normalize_label)
    ).astype(int)
    final["conf_prob"] = final["confidence"] / 10.0
    final["ece_group"] = "Final turn"

    cols = ["dialogue_id", "turn", "conf_prob", "correct", "ece_group"]
    return pd.concat(
        [
            t0[cols + ["confidence"]].rename(columns={"confidence": "confidence_raw"}),
            during[cols + ["confidence"]].rename(columns={"confidence": "confidence_raw"}),
            final[cols + ["confidence"]].rename(columns={"confidence": "confidence_raw"}),
        ],
        ignore_index=True,
    )


def summarize_ece(ece_df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    rows = []
    for group, g in ece_df.groupby("ece_group"):
        conf = g["conf_prob"].to_numpy(dtype=float)
        cor = g["correct"].to_numpy(dtype=float)
        _, acc_pct, mean_conf, counts = compute_bin_stats(conf, cor, edges)
        rows.append(
            {
                "ece_group": group,
                "n": len(g),
                "accuracy": float(cor.mean()),
                "mean_confidence": float(g["confidence_raw"].mean()),
                "mean_conf_prob": float(conf.mean()),
                "ece": float(ece_from_bins(mean_conf, acc_pct, counts)),
            }
        )
    return pd.DataFrame(rows)


def plot_confidence_arg_quality(by_turn: pd.DataFrame, out_png: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    turns = by_turn["turn"].to_numpy()
    ax1.plot(
        turns,
        by_turn["mean_confidence"],
        marker="o",
        color="#4C78A8",
        linewidth=2.0,
        label="Self-reported confidence (target)",
    )
    ax1.set_ylabel("Mean confidence (1–10)", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.set_ylim(7.8, 9.5)
    ax1.set_xlim(0.5, MAX_TURN + 0.5)
    ax1.set_xticks(range(1, MAX_TURN + 1))

    ax1b = ax1.twinx()
    for d in TOP_LEVEL_QUALITY_DIMENSIONS:
        col = f"mean_{d}"
        if col not in by_turn.columns:
            continue
        ax1b.plot(
            turns,
            by_turn[col],
            marker="s",
            linewidth=1.6,
            linestyle="--",
            label=d.capitalize(),
            color=DIM_COLORS[d],
        )
    ax1b.set_ylabel("Mean judge score (0–3)", color="#333333")
    ax1b.tick_params(axis="y", labelcolor="#333333")
    ax1b.set_ylim(1.7, 2.75)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=9,
        frameon=True,
    )
    ax1.set_xlabel("Persuasion turn")
    ax1.set_title(
        "Mean target confidence and mean argument-quality dimensions by turn",
        pad=10,
        fontsize=11,
    )

    fig.suptitle(
        "Self-reported confidence × argument quality — FEVER / DeepSeek target / GPT persuader",
        fontsize=12,
        y=1.02,
    )
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def confidence_dimension_stats(pairs: pd.DataFrame, dim: str) -> dict[str, float]:
    x = pd.to_numeric(pairs["confidence"], errors="coerce")
    y = pd.to_numeric(pairs[dim], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"n": int(mask.sum()), "pearson_r": float("nan"), "spearman_rho": float("nan")}
    r, p = stats.pearsonr(x, y)
    rho, p_s = stats.spearmanr(x, y)
    z = np.arctanh(r)
    se_z = 1 / np.sqrt(len(x) - 3)
    return {
        "dimension": dim,
        "n": int(mask.sum()),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "pearson_r_ci95_lo": float(np.tanh(z - 1.96 * se_z)),
        "pearson_r_ci95_hi": float(np.tanh(z + 1.96 * se_z)),
        "spearman_rho": float(rho),
        "spearman_p": float(p_s),
    }


def export_confidence_quality_correlation(
    pairs: pd.DataFrame,
    by_turn: pd.DataFrame,
    out_csv: Path,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if d in pairs.columns]

    for label, frame in [
        ("all_counterarguments", pairs),
        ("turn_1_only", pairs[pairs["turn"] == 1]),
    ]:
        for dim in dims:
            row = confidence_dimension_stats(frame, dim)
            row["analysis_level"] = label
            rows.append(row)

    p_dm = pairs.copy()
    p_dm["confidence_dm"] = p_dm.groupby("dialogue_id")["confidence"].transform(lambda s: s - s.mean())
    for dim in dims:
        p_dm[f"{dim}_dm"] = p_dm.groupby("dialogue_id")[dim].transform(lambda s: s - s.mean())
        dm = pd.DataFrame({"confidence": p_dm["confidence_dm"], dim: p_dm[f"{dim}_dm"]})
        row = confidence_dimension_stats(dm, dim)
        row["analysis_level"] = "within_dialogue_demeaned"
        rows.append(row)

    bt = by_turn.copy()
    for dim in dims:
        col = f"mean_{dim}"
        if col not in bt.columns:
            continue
        frame = bt.rename(columns={"mean_confidence": "confidence", col: dim})[["confidence", dim]]
        row = confidence_dimension_stats(frame, dim)
        row["analysis_level"] = "mean_scores_by_turn"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def plot_confidence_quality_correlation(pairs: pd.DataFrame, out_png: Path) -> None:
    """Scatter: target confidence vs each quality dimension (all counterarguments)."""
    sns.set_theme(style="whitegrid")
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if d in pairs.columns]
    fig, axes = plt.subplots(1, len(dims), figsize=(4.2 * len(dims), 4.5), sharey=True)
    if len(dims) == 1:
        axes = [axes]

    work = pairs.copy()
    work["confidence"] = pd.to_numeric(work["confidence"], errors="coerce")

    for ax, dim in zip(axes, dims):
        sub = work.dropna(subset=["confidence", dim])
        x = sub["confidence"].to_numpy(dtype=float)
        y = sub[dim].to_numpy(dtype=float)
        ax.scatter(x, y, alpha=0.18, s=14, color=DIM_COLORS.get(dim, "#666666"), edgecolors="none")
        if len(x) >= 3:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, intercept + slope * xs, color="#333333", linewidth=1.8)
            st = confidence_dimension_stats(sub, dim)
            ax.text(
                0.04,
                0.96,
                f"ρ = {st['spearman_rho']:.2f}\nr = {st['pearson_r']:.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            )
        ax.set_xlabel("Target confidence (1–10)")
        if ax is axes[0]:
            ax.set_ylabel("Judge score (0–3)")
        ax.set_title(dim.capitalize(), fontsize=11, color=DIM_COLORS.get(dim, "#333333"))
        ax.set_xlim(4.5, 10.5)
        ax.set_ylim(0.5, 3.2)

    fig.suptitle(
        "Does argument quality co-vary with target confidence?\n"
        "FEVER / DeepSeek target / GPT persuader (one point = one counterargument)",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def cogency_reasonableness_stats(pairs: pd.DataFrame) -> dict[str, float]:
    x = pd.to_numeric(pairs["cogency"], errors="coerce")
    y = pd.to_numeric(pairs["reasonableness"], errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    r, p = stats.pearsonr(x, y)
    rho, p_s = stats.spearmanr(x, y)
    slope, intercept, _, p_lr, _ = stats.linregress(x, y)
    n = int(mask.sum())
    z = np.arctanh(r)
    se_z = 1 / np.sqrt(n - 3)
    return {
        "n": n,
        "pearson_r": float(r),
        "pearson_p": float(p),
        "pearson_r_ci95_lo": float(np.tanh(z - 1.96 * se_z)),
        "pearson_r_ci95_hi": float(np.tanh(z + 1.96 * se_z)),
        "spearman_rho": float(rho),
        "spearman_p": float(p_s),
        "r_squared": float(r**2),
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
        "regression_p": float(p_lr),
        "exact_match_pct": float((x == y).mean() * 100),
    }


def export_cogency_reasonableness_analysis(pairs: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    rows = [cogency_reasonableness_stats(pairs)]
    rows[0]["analysis_level"] = "all_counterarguments"

    p1 = pairs[pairs["turn"] == 1]
    s1 = cogency_reasonableness_stats(p1)
    s1["analysis_level"] = "turn_1_only"
    rows.append(s1)

    p_dm = pairs.copy()
    p_dm["cogency_dm"] = p_dm.groupby("dialogue_id")["cogency"].transform(lambda s: s - s.mean())
    p_dm["reasonableness_dm"] = p_dm.groupby("dialogue_id")["reasonableness"].transform(
        lambda s: s - s.mean()
    )
    dm = pd.DataFrame(
        {
            "cogency": p_dm["cogency_dm"],
            "reasonableness": p_dm["reasonableness_dm"],
        }
    )
    s_dm = cogency_reasonableness_stats(dm)
    s_dm["analysis_level"] = "within_dialogue_demeaned"
    rows.append(s_dm)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def build_cogency_reasonableness_correlation_rows(
    pairs: pd.DataFrame,
    by_turn: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(label: str, frame: pd.DataFrame) -> None:
        st = cogency_reasonableness_stats(frame)
        st["label"] = label
        rows.append(st)

    add("All counterarguments (turns 1–15)", pairs)
    bt = by_turn.rename(
        columns={"mean_cogency": "cogency", "mean_reasonableness": "reasonableness"}
    )
    add("Mean scores by turn", bt[["cogency", "reasonableness"]])
    return rows


def _format_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def plot_cogency_reasonableness_correlation(
    pairs: pd.DataFrame,
    by_turn: pd.DataFrame,
    out_png: Path,
) -> None:
    """Correlation summary with prominent r / R² and analysis table."""
    sns.set_theme(style="whitegrid")
    rows = build_cogency_reasonableness_correlation_rows(pairs, by_turn)
    main = rows[0]

    fig = plt.figure(figsize=(11, 5.8))
    ax_top = fig.add_subplot(1, 1, 1)
    ax_top.axis("off")

    fig.text(
        0.5,
        0.97,
        "Cogency × Reasonableness",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.92,
        "FEVER · DeepSeek target · GPT persuader",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )

    hero_y = 0.86
    metrics = [
        ("Pearson r", f"{main['pearson_r']:.3f}"),
        ("R²", f"{main['r_squared']:.3f}"),
        ("Spearman ρ", f"{main['spearman_rho']:.3f}"),
        ("n", f"{int(main['n'])}"),
    ]
    for i, (label, value) in enumerate(metrics):
        x = 0.04 + i * 0.23
        ax_top.text(
            x, hero_y, value, transform=ax_top.transAxes,
            fontsize=28, fontweight="bold", ha="left", va="center",
        )
        ax_top.text(
            x, hero_y - 0.12, label, transform=ax_top.transAxes,
            fontsize=11, ha="left", va="center", color="#444444",
        )

    table_data = [
        [
            r["label"],
            str(int(r["n"])),
            f"{r['pearson_r']:.3f}",
            f"{r['r_squared']:.3f}",
            f"{r['spearman_rho']:.3f}",
            _format_p(r["pearson_p"]),
        ]
        for r in rows
    ]
    col_labels = ["Analysis level", "n", "Pearson r", "R²", "Spearman ρ", "p"]
    table = ax_top.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.02, 1.0, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E8EEF7")
            cell.set_text_props(fontweight="bold")
        elif col == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02

    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Confidence × argument quality + ECE.")
    ap.add_argument("--expl", type=Path, default=DEFAULT_EXPL)
    ap.add_argument("--judge-csv", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--out-dir", type=Path, default=FEVER_DIR)
    args = ap.parse_args()

    expl = pd.read_csv(args.expl)
    judge = pd.read_csv(args.judge_csv)
    pairs = build_turn_pairs(expl, judge)
    by_turn = summarize_by_turn(pairs)
    ece_df = build_ece_frame(expl, pairs)
    ece_summary = summarize_ece(
        ece_df[ece_df["ece_group"].isin(["Turn 0 (before persuasion)", "Turns 1+ (after counterargument)", "Final turn"])],
        bin_edges(),
    )

    out_dir = args.out_dir / "arg_quality"
    png_dir = args.out_dir / "png"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    pairs.to_csv(out_dir / "confidence_arg_quality_pairs.csv", index=False)
    by_turn.to_csv(out_dir / "confidence_arg_quality_by_turn.csv", index=False)
    ece_df.to_csv(out_dir / "confidence_arg_quality_ece_rows.csv", index=False)
    ece_summary.to_csv(out_dir / "confidence_arg_quality_ece_summary.csv", index=False)

    out_png = png_dir / "confidence_arg_quality.png"
    plot_confidence_arg_quality(by_turn, out_png)

    corr_csv = out_dir / "cogency_reasonableness_correlation_analysis.csv"
    export_cogency_reasonableness_analysis(pairs, corr_csv)

    corr_png = png_dir / "cogency_reasonableness_correlation.png"
    plot_cogency_reasonableness_correlation(pairs, by_turn, corr_png)

    conf_qual_csv = out_dir / "confidence_quality_correlation.csv"
    conf_qual_summary = export_confidence_quality_correlation(pairs, by_turn, conf_qual_csv)
    conf_qual_png = png_dir / "confidence_quality_correlation.png"
    plot_confidence_quality_correlation(pairs, conf_qual_png)

    old_png = png_dir / "confidence_arg_quality_joint.png"
    if old_png.exists():
        old_png.unlink()

    print(f"Paired turn rows: {len(pairs)}")
    print(f"Saved figure: {out_png}")
    print(f"Saved figure: {corr_png}")
    print(f"Saved figure: {conf_qual_png}")
    print(f"Saved correlations: {corr_csv}")
    print(f"Saved confidence×quality correlations: {conf_qual_csv}")
    print("\nConfidence × quality (all counterarguments, Spearman ρ):")
    main = conf_qual_summary[conf_qual_summary["analysis_level"] == "all_counterarguments"]
    for _, r in main.iterrows():
        print(f"  {r['dimension']:14s}  ρ={r['spearman_rho']:.3f}  r={r['pearson_r']:.3f}  n={int(r['n'])}")
    print("\nECE summary:")
    for _, r in ece_summary.iterrows():
        print(
            f"  {r['ece_group']}: n={int(r['n'])}  "
            f"acc={r['accuracy']:.3f}  mean_conf={r['mean_confidence']:.2f}  ECE={r['ece']:.3f}"
        )


if __name__ == "__main__":
    main()
