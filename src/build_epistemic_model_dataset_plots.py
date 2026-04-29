from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd

from linguistic_confidence import NO_MARKER, primary_epistemic_marker


def slugify_model(model_id: str) -> str:
    tail = model_id.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", tail)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def marker_profile_conf(rows: pd.DataFrame) -> dict[str, float]:
    by_marker: dict[str, list[int]] = defaultdict(list)
    for _, r in rows.iterrows():
        by_marker[str(r["marker"])].append(int(r["correct"]))
    out: dict[str, float] = {}
    for marker, vals in by_marker.items():
        out[marker] = mean(vals) if vals else float("nan")
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_root = root / "output_wood" / "epistemic_markers"
    rollout_path = out_root / "full_epistemic_marker_rollout_temp05_scale10.csv"
    if not rollout_path.exists():
        raise FileNotFoundError(f"Missing rollout file: {rollout_path}")

    df = pd.read_csv(rollout_path)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).astype(float)
    df["marker"] = df["answer"].fillna("").astype(str).map(primary_epistemic_marker)
    df["has_marker"] = (df["marker"] != NO_MARKER).astype(int)

    metric_rows: list[dict[str, object]] = []
    for (model, dataset), sub in df.groupby(["model", "dataset"], sort=True):
        prof = marker_profile_conf(sub)
        pred = [prof.get(m, 0.5) for m in sub["marker"].tolist()]
        ece_like = mean(abs(float(y) - float(p)) for y, p in zip(sub["correct"].tolist(), pred))
        metric_rows.append(
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

    metrics_df = pd.DataFrame(metric_rows).sort_values(["model", "dataset"]).reset_index(drop=True)
    metrics_csv = out_root / "model_dataset_metrics_temp05_scale10.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    by_model_root = out_root / "by_model"
    by_dataset_root = out_root / "by_dataset"
    safe_mkdir(by_model_root)
    safe_mkdir(by_dataset_root)

    key_metrics = [
        "marker_coverage",
        "accuracy",
        "mean_self_confidence",
        "mean_marker_confidence",
        "in_domain_marker_calibration_error",
    ]

    # Per-model folders and plots across datasets
    for model, sub in metrics_df.groupby("model", sort=True):
        model_slug = slugify_model(model)
        model_dir = by_model_root / model_slug
        safe_mkdir(model_dir)

        sub.to_csv(model_dir / "metrics_by_dataset.csv", index=False)

        fig, axes = plt.subplots(1, len(key_metrics), figsize=(20, 4))
        for i, metric in enumerate(key_metrics):
            ax = axes[i]
            ax.bar(sub["dataset"], sub[metric])
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=30)
            if metric.endswith("error"):
                ax.set_ylim(bottom=0)
            else:
                ax.set_ylim(0, 1.05)
        fig.suptitle(f"Metrics by Dataset: {model}", fontsize=13)
        fig.tight_layout()
        fig.savefig(model_dir / "metrics_by_dataset.png", dpi=170)
        plt.close(fig)

        # Marker usage frequency chart (top 15 markers)
        sub_rows = df[df["model"] == model]
        marker_counts = (
            sub_rows["marker"]
            .value_counts()
            .head(15)
            .sort_values(ascending=True)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(marker_counts.index, marker_counts.values)
        ax.set_title(f"Top Epistemic Markers: {model}")
        fig.tight_layout()
        fig.savefig(model_dir / "top_markers.png", dpi=170)
        plt.close(fig)

    # Per-dataset folders and plots across models
    for dataset, sub in metrics_df.groupby("dataset", sort=True):
        ds_dir = by_dataset_root / dataset
        safe_mkdir(ds_dir)
        sub.to_csv(ds_dir / "metrics_by_model.csv", index=False)

        fig, axes = plt.subplots(1, len(key_metrics), figsize=(20, 4))
        labels = [slugify_model(m) for m in sub["model"]]
        for i, metric in enumerate(key_metrics):
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
        fig.savefig(ds_dir / "metrics_by_model.png", dpi=170)
        plt.close(fig)

    print(f"Saved: {metrics_csv}")
    print(f"Saved folders: {by_model_root} and {by_dataset_root}")


if __name__ == "__main__":
    main()

