"""
Render inter-annotator agreement summary table (PNG).

Example:
  python3 plot_human_annotator_agreement_table.py
  python3 plot_human_annotator_agreement_table.py --updates ../output_wood/.../human_annotator_cases_updated.numbers
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
import pandas as pd
from numbers_parser import Document

from analyze_human_annotator_agreement import (
    ALPHA_THRESHOLD,
    KAPPA_THRESHOLD,
    agreement_report,
    discover_default_inputs,
)
from analyze_human_annotator_disagreements import ANNOTATOR_NAMES, load_merged_annotators


def load_updates(path: Path) -> pd.DataFrame:
    doc = Document(path)
    table = doc.sheets[0].tables[0]
    headers = [table.cell(0, c).value for c in range(table.num_cols)]
    rows: list[dict[str, object]] = []
    for r in range(1, table.num_rows):
        rows.append({headers[c]: table.cell(r, c).value for c in range(table.num_cols)})
    out = pd.DataFrame(rows)
    out["item_id"] = out["item_id"].astype(int)
    # Normalize column naming across different Numbers exports.
    # Some sheets use A1/A2/A3 instead of annotator_1/annotator_2/annotator_3.
    rename = {}
    if "annotator_1" not in out.columns and "A1" in out.columns:
        rename.update({"A1": "annotator_1", "A2": "annotator_2", "A3": "annotator_3"})
    if rename:
        out = out.rename(columns=rename)
    return out


def apply_updates(base: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    merged = base.copy()
    for _, row in updates.iterrows():
        idx = merged.index[merged["item_id"] == int(row["item_id"])][0]
        dim = str(row["dimension"])
        for i, name in enumerate(ANNOTATOR_NAMES, start=1):
            col = f"{name}__{dim}"
            merged.at[idx, col] = int(row[f"annotator_{i}"])
    return merged


def merged_to_annotator_csvs(merged: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    context_cols = ["item_id", "dataset", "claim", "target_answer_before", "counterargument", "target_answer_after"]
    for i, name in enumerate(ANNOTATOR_NAMES, start=1):
        cols = [c for c in context_cols if c in merged.columns]
        df = merged[cols].copy()
        for dim in ("cogency", "effectiveness", "reasonableness"):
            df[dim] = merged[f"{name}__{dim}"].astype(int)
        path = out_dir / f"human_annotation_annotator_{i}_updated.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def report_table_df(report: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for dim in ("cogency", "effectiveness", "reasonableness"):
        sub = report[report["dimension"] == dim]
        alpha = sub[sub["metric"] == "krippendorff_alpha"].iloc[0]
        k12 = sub[(sub["metric"] == "quadratic_kappa") & (sub["annotator_a"] == "annotator_1")].iloc[0]
        k13 = sub[(sub["metric"] == "quadratic_kappa") & (sub["annotator_b"] == "annotator_3") & (sub["annotator_a"] == "annotator_1")].iloc[0]
        k23 = sub[(sub["metric"] == "quadratic_kappa") & (sub["annotator_a"] == "annotator_2")].iloc[0]
        rows.append(
            {
                "dimension": dim.capitalize(),
                "alpha": f"{alpha['value']:.3f}",
                "alpha_threshold": f"≥ {ALPHA_THRESHOLD:.2f}",
                "kappa_12": f"{k12['value']:.3f}",
                "kappa_13": f"{k13['value']:.3f}",
                "kappa_23": f"{k23['value']:.3f}",
                "kappa_threshold": f"≥ {KAPPA_THRESHOLD:.2f}",
                "alpha_pass": bool(alpha["passes"]),
                "kappa_12_pass": bool(k12["passes"]),
                "kappa_13_pass": bool(k13["passes"]),
                "kappa_23_pass": bool(k23["passes"]),
            }
        )
    return pd.DataFrame(rows)


def plot_agreement_table(report: pd.DataFrame, out_path: Path, *, title_suffix: str = "") -> bool:
    table = report_table_df(report)
    all_pass = (
        table["alpha_pass"].all()
        and table["kappa_12_pass"].all()
        and table["kappa_13_pass"].all()
        and table["kappa_23_pass"].all()
    )

    col_labels = [
        "Dimension",
        "Krippendorff's α\n(overall)",
        "α threshold",
        "Cohen's κ\n(annotator 1 ↔ 2)",
        "Cohen's κ\n(annotator 1 ↔ 3)",
        "Cohen's κ\n(annotator 2 ↔ 3)",
        "κ threshold",
    ]
    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []

    for _, row in table.iterrows():
        cells = [
            row["dimension"],
            f"{row['alpha']}" + ("" if row["alpha_pass"] else " (fail)"),
            row["alpha_threshold"],
            f"{row['kappa_12']}" + ("" if row["kappa_12_pass"] else " (fail)"),
            f"{row['kappa_13']}" + ("" if row["kappa_13_pass"] else " (fail)"),
            f"{row['kappa_23']}" + ("" if row["kappa_23_pass"] else " (fail)"),
            row["kappa_threshold"],
        ]
        colors = [
            "#ffffff",
            "#d8f3dc" if row["alpha_pass"] else "#ffd6d6",
            "#ffffff",
            "#d8f3dc" if row["kappa_12_pass"] else "#ffd6d6",
            "#d8f3dc" if row["kappa_13_pass"] else "#ffd6d6",
            "#d8f3dc" if row["kappa_23_pass"] else "#ffd6d6",
            "#ffffff",
        ]
        cell_text.append(cells)
        cell_colors.append(colors)

    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")
    title = "Human annotator agreement (n = 100 items, 3 annotators)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    table_art = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#e9ecef"] * len(col_labels),
        cellLoc="center",
        loc="center",
    )
    table_art.auto_set_font_size(False)
    table_art.set_fontsize(10)
    table_art.scale(1, 1.8)

    for (r, c), cell in table_art.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
        if c > 0 and r > 0:
            val = cell_text[r - 1][c]
            if "(fail)" in val:
                cell.set_text_props(color="#b00020")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return bool(all_pass)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot human annotator agreement table.")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=_PROJECT_ROOT / "output_wood" / "persuasion" / "human_annotators",
    )
    ap.add_argument(
        "--updates",
        type=Path,
        default=None,
        help="Optional .numbers file with updated scores for selected item_id × dimension rows.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path.",
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Output agreement CSV path.",
    )
    args = ap.parse_args()

    input_dir = args.input_dir
    if args.updates:
        merged = load_merged_annotators(input_dir)
        updates = load_updates(args.updates)
        merged = apply_updates(merged, updates)
        paths = merged_to_annotator_csvs(merged, input_dir)
        report = agreement_report(paths)
        out_png = args.out or (input_dir / "human_annotator_agreement_table_updated.png")
        report_csv = args.report_out or (input_dir / "human_annotator_agreement_updated.csv")
        suffix = "After calibration meeting updates (15 cases)"
    else:
        paths = discover_default_inputs(input_dir)
        if all(p.suffix == ".csv" for p in paths):
            report = agreement_report(paths)
        else:
            from analyze_human_annotator_agreement import convert_submissions

            paths = convert_submissions(paths, input_dir)
            report = agreement_report(paths)
        out_png = args.out or (input_dir / "human_annotator_agreement_table.png")
        report_csv = args.report_out or (input_dir / "human_annotator_agreement.csv")
        suffix = ""

    report.to_csv(report_csv, index=False)
    all_pass = plot_agreement_table(report, out_png, title_suffix=suffix)
    print(f"Saved table -> {out_png}")
    print(f"Saved report -> {report_csv}")
    print(f"All thresholds met: {all_pass}")


if __name__ == "__main__":
    main()
