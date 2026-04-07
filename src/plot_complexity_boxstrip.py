"""
Box plots of Wood complexity per dataset, for Low / Middle / High tertiles
(complexity_level 1/2/3).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "output_wood"

DEFAULT_CSVS = [
    ("fever480_160x3_complexity_wood_v1_lr_40_resplit.csv", "fever"),
    ("conflictqa_popqa480_160x3_wood_v2_resplit.csv", "conflictqa_popqa"),
    ("conflictqa_strategyqa480_160x3_wood_v2_resplit.csv", "conflictqa_strategyqa"),
]


def dataset_display_name(tag: str) -> str:
    t = tag.lower()
    if t == "fever":
        return "FEVER"
    if t == "conflictqa_popqa":
        return "ConflictQA — popQA"
    if t == "conflictqa_strategyqa":
        return "ConflictQA — strategyQA"
    return tag


def load_and_label(paths_tags: list[tuple[Path, str]], y_col: str) -> pd.DataFrame:
    frames = []
    for path, tag in paths_tags:
        df = pd.read_csv(path)
        df["_dataset"] = dataset_display_name(tag)
        if y_col not in df.columns:
            raise KeyError(f"{path}: missing column {y_col!r}")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        if "complexity_level" not in df.columns:
            raise KeyError(f"{path}: missing complexity_level")
        df["complexity_level"] = pd.to_numeric(df["complexity_level"], errors="coerce").astype("Int64")
        lvl_map = {1: "Low", 2: "Middle", 3: "High"}
        df["complexity"] = df["complexity_level"].map(lvl_map)
        df = df.dropna(subset=[y_col, "complexity"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--y-col",
        default="wood_total_mean",
        help="Numeric column for y-axis (default: wood_total_mean)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR / "complexity_boxstrip_by_dataset.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help="Optional: pairs are not supported; pass 0 args to use defaults in output_wood/",
    )
    args = parser.parse_args()

    if args.csv_paths:
        paths_tags = [(Path(p), Path(p).stem) for p in args.csv_paths]
    else:
        paths_tags = [
            (DEFAULT_OUT_DIR / name, tag) for name, tag in DEFAULT_CSVS
        ]

    missing = [p for p, _ in paths_tags if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CSV(s): " + ", ".join(str(p) for p in missing))

    df = load_and_label(paths_tags, args.y_col)
    order_ds: list[str] = []
    for _, tag in paths_tags:
        lab = dataset_display_name(tag)
        if lab not in order_ds:
            order_ds.append(lab)
    order_ds = [d for d in order_ds if d in set(df["_dataset"])]
    order_cx = ["Low", "Middle", "High"]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 6))

    sns.boxplot(
        data=df,
        x="_dataset",
        y=args.y_col,
        hue="complexity",
        order=order_ds,
        hue_order=order_cx,
        width=0.65,
        fliersize=0,
        ax=ax,
    )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Wood total (mean across judges)" if args.y_col == "wood_total_mean" else args.y_col)
    ax.set_title("Complexity by dataset (Low / Middle / High)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        title="Complexity",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("Saved:", args.out.resolve())


if __name__ == "__main__":
    main()
