"""
Render a single-page overview table for persuasion results (English).

Columns: Turn 0 (before persuasion), Final, Change (Δ).
Rows: Accuracy, Mean confidence, Flip rate, ECE.

Example:
  python3 plot_persuasion_overview_table.py
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

from plot_ece_calibration import bin_edges, compute_bin_stats, ece_from_bins
from persuasion_rollout_view import prepare_dialogue_view

FEVER_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_INPUT = FEVER_DIR / "csv" / "expl.csv"
DEFAULT_OUT_DIR = FEVER_DIR


def persuasion_setup_labels(df: pd.DataFrame) -> tuple[str, str, str]:
    persuader = "unknown persuader"
    if "persuader_model" in df.columns and df["persuader_model"].notna().any():
        persuader = str(df["persuader_model"].dropna().iloc[0]).replace("openai/", "")
    dataset = "FEVER"
    if "dataset" in df.columns and df["dataset"].notna().any():
        dataset = str(df["dataset"].dropna().iloc[0]).upper()
    target = "DeepSeek"
    if "target_model" in df.columns and df["target_model"].notna().any():
        raw = str(df["target_model"].dropna().iloc[0]).lower()
        if "deepseek" in raw:
            target = "DeepSeek"
        else:
            target = str(df["target_model"].dropna().iloc[0])
    return persuader, dataset, target


def ece_for_rows(conf_prob: np.ndarray, correct: np.ndarray, edges: np.ndarray) -> float:
    _, acc_pct, mean_conf, counts = compute_bin_stats(conf_prob, correct, edges)
    return float(ece_from_bins(mean_conf, acc_pct, counts))


def build_overview_metrics(view: pd.DataFrame) -> dict[str, float | int]:
    edges = bin_edges()
    conf_t0 = (pd.to_numeric(view["conf_t0"], errors="coerce") / 10.0).to_numpy(dtype=float)
    conf_final = (pd.to_numeric(view["conf_final"], errors="coerce") / 10.0).to_numpy(dtype=float)
    cor_t0 = view["correct_t0"].to_numpy(dtype=float)
    cor_final = view["correct_final"].to_numpy(dtype=float)

    acc_t0 = float(cor_t0.mean())
    acc_final = float(cor_final.mean())
    mean_conf_t0 = float(pd.to_numeric(view["conf_t0"], errors="coerce").mean())
    mean_conf_final = float(pd.to_numeric(view["conf_final"], errors="coerce").mean())
    flip_rate = float(view["flip"].mean())
    ece_t0 = ece_for_rows(conf_t0, cor_t0, edges)
    ece_final = ece_for_rows(conf_final, cor_final, edges)

    return {
        "n_dialogues": int(len(view)),
        "acc_t0": acc_t0,
        "acc_final": acc_final,
        "acc_delta_pp": (acc_final - acc_t0) * 100.0,
        "mean_conf_t0": mean_conf_t0,
        "mean_conf_final": mean_conf_final,
        "mean_conf_delta": mean_conf_final - mean_conf_t0,
        "flip_rate": flip_rate,
        "ece_t0": ece_t0,
        "ece_final": ece_final,
        "ece_delta": ece_final - ece_t0,
    }


def format_delta(value: float, *, suffix: str = "", signed: bool = True) -> str:
    if np.isnan(value):
        return "—"
    if signed:
        return f"{value:+.2f}{suffix}"
    return f"{value:.2f}{suffix}"


def build_table_rows(metrics: dict[str, float | int]) -> list[list[str]]:
    return [
        [
            "Accuracy",
            f"{metrics['acc_t0'] * 100:.1f}%",
            f"{metrics['acc_final'] * 100:.1f}%",
            format_delta(metrics["acc_delta_pp"], suffix=" pp"),
        ],
        [
            "Mean confidence (1-10)",
            f"{metrics['mean_conf_t0']:.2f}",
            f"{metrics['mean_conf_final']:.2f}",
            format_delta(metrics["mean_conf_delta"]),
        ],
        [
            "Flip rate",
            "—",
            f"{metrics['flip_rate'] * 100:.1f}%",
            "—",
        ],
        [
            "ECE",
            f"{metrics['ece_t0']:.3f}",
            f"{metrics['ece_final']:.3f}",
            format_delta(metrics["ece_delta"]),
        ],
    ]


def plot_overview_table(
    table_rows: list[list[str]],
    *,
    title: str,
    subtitle: str,
    out_png: Path,
) -> None:
    col_labels = ["Metric", "Turn 0 (before)", "Final", "Change (Δ)"]
    cell_text = table_rows

    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.18, 0.96, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    header_color = "#4C78A8"
    row_colors = ["#F7F7F7", "#FFFFFF", "#F7F7F7", "#FFFFFF"]
    n_cols = len(col_labels)

    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", weight="bold")

    delta_colors = {
        "Accuracy": {"+": "#55A868", "-": "#C44E52"},
        "Mean confidence (1-10)": {"+": "#C44E52", "-": "#55A868"},
        "ECE": {"+": "#C44E52", "-": "#55A868"},
    }

    for i, row_color in enumerate(row_colors, start=1):
        metric_name = cell_text[i - 1][0]
        for j in range(n_cols):
            cell = table[(i, j)]
            cell.set_facecolor(row_color)
            if j == 0:
                cell.set_text_props(weight="bold", ha="center")
            if j == 3:
                val = cell_text[i - 1][3]
                palette = delta_colors.get(metric_name, {})
                if val.startswith("+") and "+" in palette:
                    cell.set_text_props(color=palette["+"])
                elif val.startswith("-") and "-" in palette:
                    cell.set_text_props(color=palette["-"])

    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)
    ax.text(
        0.5,
        0.06,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )

    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_overview_csv(metrics: dict[str, float | int], out_csv: Path) -> None:
    rows = [
        ("n_dialogues", metrics["n_dialogues"]),
        ("accuracy_turn0", metrics["acc_t0"]),
        ("accuracy_final", metrics["acc_final"]),
        ("accuracy_delta_pp", metrics["acc_delta_pp"]),
        ("mean_confidence_turn0", metrics["mean_conf_t0"]),
        ("mean_confidence_final", metrics["mean_conf_final"]),
        ("mean_confidence_delta", metrics["mean_conf_delta"]),
        ("flip_rate", metrics["flip_rate"]),
        ("ece_turn0", metrics["ece_t0"]),
        ("ece_final", metrics["ece_final"]),
        ("ece_delta", metrics["ece_delta"]),
    ]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(out_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot persuasion overview table (English).")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    expl = pd.read_csv(args.input)
    view = prepare_dialogue_view(expl)
    metrics = build_overview_metrics(view)
    table_rows = build_table_rows(metrics)

    persuader, dataset, target = persuasion_setup_labels(expl)
    title = f"Persuasion overview — {dataset} · Target: {target}"
    subtitle = (
        f"Persuader: {persuader} · n = {metrics['n_dialogues']} dialogues · "
        "Turn 0 = before persuasion; Final = last turn per dialogue"
    )

    png_dir = args.out_dir / "png"
    csv_dir = args.out_dir / "csv"
    png_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    out_png = png_dir / "persuasion_overview_table.png"
    out_csv = csv_dir / "persuasion_overview_table.csv"

    plot_overview_table(table_rows, title=title, subtitle=subtitle, out_png=out_png)
    write_overview_csv(metrics, out_csv)

    print(f"Saved figure: {out_png}")
    print(f"Saved CSV: {out_csv}")
    print("\nOverview:")
    for row in table_rows:
        print("  " + " | ".join(row))


if __name__ == "__main__":
    main()
