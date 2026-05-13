"""
Compute and plot self-reported confidence by model.

Input CSVs are expected to contain at least:
  - model
  - confidence (in [0, 1])
Optional:
  - dataset

Examples:
  python3 src/plot_self_reported_confidence.py \
    --inputs output_wood/temperature_experiments/conflictqa/csv/conflictqa_ece_*_conflictqa_temp06_scale10.csv \
             output_wood/temperature_experiments/fever/csv/conflictqa_ece_*_fever_temp06_scale10.csv \
    --out-dir output_wood/self_reported_confidence \
    --prefix temp06_scale10

Summaries are written to out-dir/csv/, plots to out-dir/png/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "output_wood" / "self_reported_confidence"

MODEL_LABELS: dict[str, str] = {
    "openai/gpt-4o-2024-11-20": "GPT-4o (Nov 2024)",
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "deepseek/deepseek-chat-v2.5": "DeepSeek Chat v2.5",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "qwen/qwen3-14b": "Qwen3-14B",
    "google/gemma-4-26b-a4b-it": "Gemma 4 26B (A4B IT)",
    # Legacy IDs for older files.
    "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet (legacy)",
    "google/gemini-flash-1.5-8b": "Gemini Flash 1.5-8B (legacy)",
    "google/gemini-2.0-flash": "Gemini 2.0 Flash (legacy)",
}


def model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "dataset" not in df.columns:
        df = df.copy()
        df["dataset"] = "all"
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["confidence"])
    df = df[(df["confidence"] >= 0.0) & (df["confidence"] <= 1.0)]
    if df.empty:
        return df
    df = df.copy()
    df["source_file"] = path.name
    return df


def load_inputs(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(_load_csv(path))
    if not frames:
        raise ValueError("No input CSV files provided")
    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise ValueError("No valid confidence rows found in input files")
    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["model", "dataset"], as_index=False)["confidence"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    summary = summary.rename(
        columns={
            "count": "n",
            "mean": "mean_confidence",
            "median": "median_confidence",
            "std": "std_confidence",
        }
    )
    summary["model_label"] = summary["model"].map(model_label)
    return summary.sort_values(["dataset", "mean_confidence"], ascending=[True, False]).reset_index(drop=True)


def save_plot_overall(df: pd.DataFrame, out_path: Path, dataset_label: str = "") -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    overall = (
        df.groupby("model", as_index=False)["confidence"]
        .mean()
        .rename(columns={"confidence": "mean_confidence"})
        .sort_values("mean_confidence", ascending=False)
    )
    overall["model_label"] = overall["model"].map(model_label)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    sns.barplot(
        data=overall,
        x="model_label",
        y="mean_confidence",
        hue="model_label",
        legend=False,
        palette="viridis",
        ax=ax,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean self-reported confidence")
    ds_label = dataset_label.strip()
    if not ds_label:
        ds_values = sorted(df["dataset"].dropna().astype(str).unique().tolist())
        if len(ds_values) == 1:
            ds_label = ds_values[0]
    if ds_label:
        ax.set_title(f"Self-reported confidence by model ({ds_label})")
    else:
        ax.set_title("Self-reported confidence by model (overall)")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_plot_by_dataset(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    dataset_count = df["dataset"].nunique()
    if dataset_count < 2:
        return

    by_ds = (
        df.groupby(["model", "dataset"], as_index=False)["confidence"]
        .mean()
        .rename(columns={"confidence": "mean_confidence"})
    )
    by_ds["model_label"] = by_ds["model"].map(model_label)

    model_order = (
        by_ds.groupby("model_label", as_index=False)["mean_confidence"]
        .mean()
        .sort_values("mean_confidence", ascending=False)["model_label"]
        .tolist()
    )
    ds_order = sorted(by_ds["dataset"].unique().tolist())

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    sns.barplot(
        data=by_ds,
        x="model_label",
        y="mean_confidence",
        hue="dataset",
        order=model_order,
        hue_order=ds_order,
        palette="Set2",
        ax=ax,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean self-reported confidence")
    ax.set_title("Self-reported confidence by model and dataset")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input CSV files with model/confidence columns",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Root directory; writes summaries under out-dir/csv/ and plots under out-dir/png/",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional suffix/prefix tag in output filenames (e.g. temp06_scale10)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only write summary CSV (useful on environments without matplotlib backend)",
    )
    parser.add_argument(
        "--dataset-label",
        default="",
        help="Optional dataset label shown in the overall plot title (e.g. DebateQA).",
    )
    args = parser.parse_args()

    tag = args.prefix.strip()
    tag = f"_{tag}" if tag else ""

    df = load_inputs(args.inputs)
    summary = build_summary(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.out_dir / "csv"
    png_dir = args.out_dir / "png"
    csv_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    summary_path = csv_dir / f"self_reported_confidence_summary{tag}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    if not args.skip_plots:
        overall_plot_path = png_dir / f"self_reported_confidence_by_model{tag}.png"
        save_plot_overall(df, overall_plot_path, dataset_label=args.dataset_label)
        print(f"Saved: {overall_plot_path}")

        by_ds_plot_path = png_dir / f"self_reported_confidence_by_model_dataset{tag}.png"
        save_plot_by_dataset(df, by_ds_plot_path)
        if by_ds_plot_path.exists():
            print(f"Saved: {by_ds_plot_path}")


if __name__ == "__main__":
    main()
