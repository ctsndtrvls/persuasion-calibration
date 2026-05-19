"""
Epistemic-marker experiment for the new model cohorts (temp 0.6, verbal scale 1-10):

  - Qwen3-14B and Gemma 4 26B on FEVER, DebateQA and PopQA (conflictqa_popqa tag).
  - GPT-4o and DeepSeek Chat v2.5 on DebateQA only.

Reuses the ACL'25 metric helpers from `run_epistemic_marker_experiment.py` and the
deterministic marker extractor from `linguistic_confidence.py`. The existing
`build_epistemic_model_dataset_plots.py` script (used for the four-model
temp 0.5 cohort) is left untouched.

Outputs everything under
  output_wood/epistemic_markers/qwen_gemma_openai_deepseek_temp06_scale10/
analogous to the existing four-model layout:

  - acl25_epistemic_marker_summary.csv
  - acl25_marker_coverage.csv
  - model_dataset_metrics.csv
  - full_epistemic_marker_rollout.csv
  - overall_metrics.png
  - overall_marker_coverage.png
  - metrics_<model_slug>.png  (per-model bar chart of ACL'25 metrics)
  - by_model/<model_slug>/{metrics_by_dataset.csv,png, top_markers.png}
  - by_dataset/<dataset>/{metrics_by_model.csv, metrics_by_model.png}
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Sequence

_PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT_BOOTSTRAP / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from linguistic_confidence import NO_MARKER, primary_epistemic_marker
from run_epistemic_marker_experiment import (
    evaluate_model,
    format_float,
    load_rows,
    split_train_test,
    write_marker_coverage,
    write_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TE = PROJECT_ROOT / "output_wood" / "temperature_experiments"

DEFAULT_QWEN_GEMMA_CSV = (
    TE / "qwen_gemma_fever_popqa_debateqa" / "csv" / "ece_qwen_gemma_fever_popqa_debateqa_temp0p6_scale10.csv"
)
DEFAULT_OPENAI_DEEPSEEK_DEBATEQA_CSV = (
    TE / "debateqa" / "csv" / "debateqa_ece_openai_deepseek_temp0p6_scale10.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "output_wood" / "epistemic_markers" / "qwen_gemma_openai_deepseek_temp06_scale10"

MODEL_ORDER: list[str] = [
    "qwen/qwen3-14b",
    "google/gemma-4-26b-a4b-it",
    "openai/gpt-4o-2024-11-20",
    "deepseek/deepseek-chat-v2.5",
]

SHORT_MODEL_LABELS: dict[str, str] = {
    "qwen/qwen3-14b": "Qwen3-14B",
    "google/gemma-4-26b-a4b-it": "Gemma 4 26B",
    "openai/gpt-4o-2024-11-20": "GPT-4o",
    "deepseek/deepseek-chat-v2.5": "DeepSeek v2.5",
}

DATASET_ORDER = ["fever", "debateqa", "conflictqa_popqa"]

ACL25_METRICS = ["I-AvgECE", "C-AvgECE", "I-AvgCV", "C-AvgCV", "MAC", "MRC", "avg_test_accuracy"]
PER_DS_METRICS = [
    "marker_coverage",
    "accuracy",
    "mean_self_confidence",
    "mean_marker_confidence",
    "in_domain_marker_calibration_error",
]


def slugify_model(model_id: str) -> str:
    tail = model_id.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", tail)


def model_sort_key(model_id: str) -> tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model_id), model_id)
    except ValueError:
        return (len(MODEL_ORDER), model_id)


def dataset_sort_key(dataset: str) -> tuple[int, str]:
    try:
        return (DATASET_ORDER.index(dataset), dataset)
    except ValueError:
        return (len(DATASET_ORDER), dataset)


def short_label(model_id: str) -> str:
    return SHORT_MODEL_LABELS.get(model_id, model_id.split("/")[-1])


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_rollout_df(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["model", "dataset", "original_index"], keep="first").reset_index(drop=True)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).astype(float)
    df["marker"] = df["answer"].fillna("").astype(str).map(primary_epistemic_marker)
    df["has_marker"] = (df["marker"] != NO_MARKER).astype(int)
    return df


def compute_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, dataset), sub in df.groupby(["model", "dataset"], sort=False):
        prof: dict[str, float] = {}
        by_marker: dict[str, list[int]] = defaultdict(list)
        for _, r in sub.iterrows():
            by_marker[str(r["marker"])].append(int(r["correct"]))
        for marker, vals in by_marker.items():
            prof[marker] = mean(vals) if vals else float("nan")
        pred = [prof.get(m, 0.5) for m in sub["marker"].tolist()]
        ece_like = mean(abs(float(y) - float(p)) for y, p in zip(sub["correct"].tolist(), pred))
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "rows": int(len(sub)),
                "marker_coverage": float(sub["has_marker"].mean()),
                "accuracy": float(sub["correct"].mean()),
                "mean_self_confidence": float(sub["confidence"].mean()),
                "mean_marker_confidence": float(mean(pred)) if pred else float("nan"),
                "in_domain_marker_calibration_error": float(ece_like),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["model", "dataset"],
        key=lambda col: col.map(lambda v: model_sort_key(v)[0]) if col.name == "model" else col.map(lambda v: dataset_sort_key(v)[0]),
    ).reset_index(drop=True)
    return out


def _bar_with_zero_baseline(ax, xs: Sequence[str], ys: Sequence[float]) -> None:
    ax.bar(list(xs), list(ys))
    finite = [v for v in ys if isinstance(v, (int, float)) and v == v]  # filter NaN
    if finite:
        lo = min(min(finite), 0.0)
        hi = max(max(finite), 0.0)
        margin = 0.05 * (hi - lo if hi != lo else 1.0)
        ax.set_ylim(lo - margin, hi + margin)
    ax.axhline(0.0, color="black", linewidth=0.5)


def plot_overall_acl25_metrics(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    models_sorted = sorted(summary_df["model"].unique(), key=model_sort_key)
    labels = [short_label(m) for m in models_sorted]

    n = len(ACL25_METRICS)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows))
    axes = axes.ravel() if rows * cols > 1 else [axes]

    for i, metric in enumerate(ACL25_METRICS):
        ax = axes[i]
        ys = []
        for m in models_sorted:
            row = summary_df[summary_df["model"] == m]
            if not row.empty and pd.notna(row.iloc[0][metric]):
                ys.append(float(row.iloc[0][metric]))
            else:
                ys.append(float("nan"))
        _bar_with_zero_baseline(ax, labels, ys)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("ACL'25 Epistemic Marker Metrics (temp 0.6, scale 1-10)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_overall_marker_coverage(metrics_df: pd.DataFrame, out_path: Path) -> None:
    if metrics_df.empty:
        return
    datasets = sorted(metrics_df["dataset"].unique(), key=dataset_sort_key)
    models_sorted = sorted(metrics_df["model"].unique(), key=model_sort_key)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in models_sorted:
        sub = metrics_df[metrics_df["model"] == m].set_index("dataset")
        xs, ys = [], []
        for d in datasets:
            if d in sub.index:
                xs.append(d)
                ys.append(float(sub.loc[d, "marker_coverage"]))
        if xs:
            ax.plot(xs, ys, marker="o", label=short_label(m))
    ax.set_xticks(datasets)
    ax.set_ylabel("Marker coverage")
    ax.set_title("Epistemic Marker Coverage by Dataset and Model")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_per_model_acl25(summary_df: pd.DataFrame, model: str, out_path: Path) -> None:
    row = summary_df[summary_df["model"] == model]
    if row.empty:
        return
    vals = []
    for m in ACL25_METRICS:
        v = row.iloc[0][m]
        vals.append(float(v) if pd.notna(v) else float("nan"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _bar_with_zero_baseline(ax, ACL25_METRICS, vals)
    ax.set_title(f"ACL'25 Metrics: {model}")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_by_model_metrics_by_dataset(sub: pd.DataFrame, model: str, out_path: Path) -> None:
    if sub.empty:
        return
    sub = sub.sort_values(by="dataset", key=lambda s: s.map(lambda v: dataset_sort_key(v)[0]))
    fig, axes = plt.subplots(1, len(PER_DS_METRICS), figsize=(4.0 * len(PER_DS_METRICS), 3.6))
    if len(PER_DS_METRICS) == 1:
        axes = [axes]
    for i, metric in enumerate(PER_DS_METRICS):
        ax = axes[i]
        ax.bar(sub["dataset"], sub[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        if metric.endswith("error"):
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(0, 1.05)
    fig.suptitle(f"Metrics by Dataset: {model}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_top_markers(df_model: pd.DataFrame, model: str, out_path: Path, top_k: int = 15) -> None:
    if df_model.empty:
        return
    counts = df_model["marker"].value_counts().head(top_k).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(counts.index, counts.values)
    ax.set_title(f"Top Epistemic Markers: {model}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_summary_models_x_datasets(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Grid: rows = key per-(model,dataset) metrics, cols = datasets, bars = models."""
    if metrics_df.empty:
        return
    datasets = sorted(metrics_df["dataset"].unique(), key=dataset_sort_key)
    models_sorted = sorted(metrics_df["model"].unique(), key=model_sort_key)
    labels = [short_label(m) for m in models_sorted]

    n_rows = len(PER_DS_METRICS)
    n_cols = len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 2.8 * n_rows), squeeze=False)
    for i, metric in enumerate(PER_DS_METRICS):
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            ys = []
            for m in models_sorted:
                row = metrics_df[(metrics_df["model"] == m) & (metrics_df["dataset"] == ds)]
                if not row.empty and pd.notna(row.iloc[0][metric]):
                    ys.append(float(row.iloc[0][metric]))
                else:
                    ys.append(float("nan"))
            ax.bar(labels, ys)
            ax.tick_params(axis="x", rotation=25)
            if metric.endswith("error"):
                ax.set_ylim(bottom=0)
            else:
                ax.set_ylim(0, 1.05)
            if i == 0:
                ax.set_title(ds)
            if j == 0:
                ax.set_ylabel(metric)
    fig.suptitle("Epistemic-marker summary: models x datasets (temp 0.6, scale 1-10)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_by_dataset_metrics_by_model(sub: pd.DataFrame, dataset: str, out_path: Path) -> None:
    if sub.empty:
        return
    sub = sub.sort_values(by="model", key=lambda s: s.map(lambda v: model_sort_key(v)[0]))
    labels = [slugify_model(m) for m in sub["model"]]
    fig, axes = plt.subplots(1, len(PER_DS_METRICS), figsize=(4.0 * len(PER_DS_METRICS), 3.6))
    if len(PER_DS_METRICS) == 1:
        axes = [axes]
    for i, metric in enumerate(PER_DS_METRICS):
        ax = axes[i]
        ax.bar(labels, sub[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        if metric.endswith("error"):
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(0, 1.05)
    fig.suptitle(f"Metrics by Model: {dataset}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[str(DEFAULT_QWEN_GEMMA_CSV), str(DEFAULT_OPENAI_DEEPSEEK_DEBATEQA_CSV)],
        help="Rollout CSVs to combine.",
    )
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--min-occurrences", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)
    by_model_root = out_dir / "by_model"
    by_dataset_root = out_dir / "by_dataset"
    safe_mkdir(by_model_root)
    safe_mkdir(by_dataset_root)

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    df = load_rollout_df(input_paths)

    rollout_csv = out_dir / "full_epistemic_marker_rollout.csv"
    df.to_csv(rollout_csv, index=False)

    metrics_df = compute_metrics_table(df)
    metrics_csv = out_dir / "model_dataset_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    rows = load_rows(input_paths)
    train, test = split_train_test(rows, train_ratio=args.train_ratio, seed=args.seed)
    models = sorted({r.model for r in rows}, key=model_sort_key)
    summary_rows: list[dict[str, float]] = []
    for model in models:
        metrics = evaluate_model(model, train, test, min_occurrences=args.min_occurrences)
        if metrics:
            summary_rows.append(metrics)

    summary_csv = out_dir / "acl25_epistemic_marker_summary.csv"
    coverage_csv = out_dir / "acl25_marker_coverage.csv"
    write_summary(summary_csv, summary_rows)
    write_marker_coverage(coverage_csv, rows)

    summary_df = pd.DataFrame(summary_rows)
    for col in ACL25_METRICS:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    plot_overall_acl25_metrics(summary_df, out_dir / "overall_acl25_metrics.png")
    plot_overall_marker_coverage(metrics_df, out_dir / "overall_marker_coverage.png")
    plot_summary_models_x_datasets(metrics_df, out_dir / "summary_models_x_datasets.png")

    for model in models:
        slug = slugify_model(model)
        plot_per_model_acl25(summary_df, model, out_dir / f"metrics_{slug}.png")

        sub_metrics = metrics_df[metrics_df["model"] == model].copy()
        model_dir = by_model_root / slug
        safe_mkdir(model_dir)
        sub_metrics.to_csv(model_dir / "metrics_by_dataset.csv", index=False)
        plot_by_model_metrics_by_dataset(sub_metrics, model, model_dir / "metrics_by_dataset.png")
        plot_top_markers(df[df["model"] == model], model, model_dir / "top_markers.png")

    for dataset, sub in metrics_df.groupby("dataset", sort=False):
        ds_dir = by_dataset_root / str(dataset)
        safe_mkdir(ds_dir)
        sub.to_csv(ds_dir / "metrics_by_model.csv", index=False)
        plot_by_dataset_metrics_by_model(sub, str(dataset), ds_dir / "metrics_by_model.png")

    print("Saved:")
    print(f"  rollout:        {rollout_csv}")
    print(f"  metrics table:  {metrics_csv}")
    print(f"  ACL25 summary:  {summary_csv}")
    print(f"  marker cov:     {coverage_csv}")
    print(f"  overall plots:  {out_dir / 'overall_acl25_metrics.png'}")
    print(f"                  {out_dir / 'overall_marker_coverage.png'}")
    print(f"                  {out_dir / 'summary_models_x_datasets.png'}")
    print(f"  by_model:       {by_model_root}")
    print(f"  by_dataset:     {by_dataset_root}")
    print("Model results:")
    for r in summary_rows:
        print(
            f"- {r['model']}: "
            f"I-AvgECE={format_float(r['I-AvgECE'])}  "
            f"C-AvgECE={format_float(r['C-AvgECE'])}  "
            f"I-AvgCV={format_float(r['I-AvgCV'])}  "
            f"C-AvgCV={format_float(r['C-AvgCV'])}  "
            f"MAC={format_float(r['MAC'])}  "
            f"MRC={format_float(r['MRC'])}"
        )


if __name__ == "__main__":
    main()
