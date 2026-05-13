"""
Reliability diagrams (grouped bars) + ECE, similar to common calibration papers.

Expected CSV columns:
  - model   : model id (e.g. openai/gpt-4o-2024-11-20)
  - dataset : short name (e.g. fever, conflictqa_popqa, conflictqa_strategyqa)
  - confidence : predicted probability of the predicted class, in [0, 1]
  - correct : 1 if correct else 0

Outputs:
  1) ece_panels_3in1_models.png - three panels: pooled + two datasets
  2) ece_per_model_<id>.png - one figure per model (bars = datasets)

--input is required (use collect_conflictqa_ece.py to gather model outputs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "output_wood"

MODEL_IDS = [
    "openai/gpt-4o-2024-11-20",
    "anthropic/claude-sonnet-4-6",
    "deepseek/deepseek-chat-v2.5",
    "google/gemini-2.5-flash",
    "qwen/qwen3-14b",
    "google/gemma-4-26b-a4b-it",
    # Legacy IDs retained for older CSV files.
    "anthropic/claude-3.5-sonnet",
    "google/gemini-flash-1.5-8b",
    "google/gemini-2.0-flash",
]

MODEL_LABELS: dict[str, str] = {
    "openai/gpt-4o-2024-11-20": "GPT-4o (Nov 2024)",
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "deepseek/deepseek-chat-v2.5": "DeepSeek Chat v2.5",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "qwen/qwen3-14b": "Qwen3-14B",
    "google/gemma-4-26b-a4b-it": "Gemma 4 26B (A4B IT)",
    # legacy rows from older collect_conflictqa_ece runs
    "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet (deprecated API id)",
    "google/gemini-flash-1.5-8b": "Gemini Flash 1.5-8B (deprecated API id)",
    "google/gemini-2.0-flash": "Gemini 2.0 Flash (deprecated for new users)",
}


def _label(mid: str) -> str:
    return MODEL_LABELS.get(mid, mid)


def bin_edges() -> np.ndarray:
    return np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)


def assign_bin(conf: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index 0..len(edges)-2 for values in [edges[0], edges[-1]]."""
    c = np.clip(conf, edges[0], edges[-1] - 1e-9)
    return np.digitize(c, edges[1:-1], right=False)


def compute_bin_stats(conf: np.ndarray, correct: np.ndarray, edges: np.ndarray):
    """Returns mids, acc_pct, mean_conf, counts per bin."""
    b = assign_bin(conf, edges)
    n_bins = len(edges) - 1
    mids = 0.5 * (edges[:-1] + edges[1:])
    acc_pct = np.zeros(n_bins)
    mean_conf = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        m = b == i
        counts[i] = int(m.sum())
        if counts[i] == 0:
            acc_pct[i] = np.nan
            mean_conf[i] = np.nan
        else:
            acc_pct[i] = 100.0 * correct[m].mean()
            mean_conf[i] = 100.0 * conf[m].mean()
    return mids, acc_pct, mean_conf, counts


def ece_from_bins(mean_conf_pct: np.ndarray, acc_pct: np.ndarray, counts: np.ndarray) -> float:
    """Expected Calibration Error (fraction, not percent)."""
    n = counts.sum()
    if n == 0:
        return float("nan")
    total = 0.0
    for i in range(len(counts)):
        if counts[i] == 0:
            continue
        total += (counts[i] / n) * abs(mean_conf_pct[i] / 100.0 - acc_pct[i] / 100.0)
    return total


def weighted_ece_over_groups(df: pd.DataFrame, edges: np.ndarray) -> float:
    if df.empty:
        return float("nan")
    conf = df["confidence"].to_numpy(dtype=float)
    cor = df["correct"].to_numpy(dtype=float)
    _, _, _, counts = compute_bin_stats(conf, cor, edges)
    mids, acc_pct, mean_conf, _ = compute_bin_stats(conf, cor, edges)
    return ece_from_bins(mean_conf, acc_pct, counts)


def plot_grouped_calibration(
    ax: plt.Axes,
    df: pd.DataFrame,
    edges: np.ndarray,
    group_col: str,
    group_order: list[str],
    title: str,
    colors: list[str] | None = None,
) -> None:
    """Grouped bars: x = confidence bin, hue = group (e.g. model or dataset)."""
    if df.empty or not group_order:
        ax.set_title(title + " (no data)")
        ax.text(0.5, 0.5, "No rows", ha="center", va="center", transform=ax.transAxes)
        return

    n_bins = len(edges) - 1
    mids = 0.5 * (edges[:-1] + edges[1:])
    x = np.arange(n_bins, dtype=float)
    n_g = len(group_order)
    width = 0.8 / max(n_g, 1)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 0.9, n_g))

    legend_ece: dict[str, float] = {}
    for gi, gname in enumerate(group_order):
        sub = df[df[group_col] == gname]
        _, acc_pct, mean_conf, counts = compute_bin_stats(
            sub["confidence"].to_numpy(float),
            sub["correct"].to_numpy(float),
            edges,
        )
        ece = ece_from_bins(mean_conf, acc_pct, counts)
        legend_label = f"{gname} (ECE={ece:.3f})" if not np.isnan(ece) else gname
        legend_ece[gname] = ece
        offset = (gi - (n_g - 1) / 2) * width
        vals = np.where(np.isnan(acc_pct), 0.0, acc_pct)
        ax.bar(x + offset, vals, width=width * 0.92, label=legend_label, color=colors[gi % len(colors)])

    # Perfect calibration: staircase at bin mid conf -> accuracy should equal conf*100
    perfect_y = 100.0 * mids
    ax.plot(x, perfect_y, "k--", linewidth=1.5, label="Perfect calibration")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(n_bins)])
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    w_ece = weighted_ece_over_groups(df, edges)
    if not np.isnan(w_ece):
        ax.text(
            0.98,
            0.02,
            f"Weighted ECE (pooled in panel): {w_ece:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )


def load_input_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"model", "dataset", "confidence", "correct"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {missing}; need {need}")
    df = df.copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["confidence", "correct"])
    df["correct"] = df["correct"].astype(int)
    df = df[(df["confidence"] >= 0) & (df["confidence"] <= 1)]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with model,dataset,confidence,correct")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--only-datasets",
        nargs="+",
        default=None,
        help="Optional dataset filter, e.g. --only-datasets fever",
    )
    parser.add_argument(
        "--file-suffix",
        default="",
        help="Optional suffix for output files (e.g. fever).",
    )
    parser.add_argument(
        "--dataset-b",
        default="fever",
        help="Dataset name for middle panel (default: fever)",
    )
    parser.add_argument(
        "--dataset-c",
        default="conflictqa_popqa",
        help="Dataset name for right panel (default: conflictqa_popqa)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    edges = bin_edges()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    df = load_input_csv(args.input)
    if args.only_datasets:
        keep = set(args.only_datasets)
        df = df[df["dataset"].isin(keep)].copy()
        if df.empty:
            raise ValueError(f"No rows left after --only-datasets filter: {sorted(keep)}")
    print("Loaded:", args.input, df.shape)

    # Only known models in order (drop unknown or keep all — keep intersection with MODEL_IDS order)
    models_in_data = [m for m in MODEL_IDS if m in set(df["model"].unique())]
    if not models_in_data:
        models_in_data = sorted(df["model"].unique())
    model_labels_order = [_label(m) for m in models_in_data]

    # Map for plotting: temporary column with display name
    df = df.copy()
    df["_model_lab"] = df["model"].map(lambda x: _label(x))

    ds_order = ["fever", "debateqa", "conflictqa_popqa", "conflictqa_strategyqa"]
    ds_present = [d for d in ds_order if d in set(df["dataset"].unique())]
    if not ds_present:
        ds_present = sorted(df["dataset"].unique())

    suffix = args.file_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    if not suffix and len(ds_present) == 1:
        suffix = "_" + ds_present[0]

    # --- 1) Panels ---
    if len(ds_present) == 1:
        only_ds = ds_present[0]
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.8), constrained_layout=True)
        plot_grouped_calibration(
            ax,
            df[df["dataset"] == only_ds],
            edges,
            "_model_lab",
            model_labels_order,
            title=f"Dataset: {only_ds} — by model",
        )
        out3 = args.out_dir / f"ece_panels_1in1_models{suffix}.png"
        fig.savefig(out3, dpi=200)
        plt.close(fig)
        print("Saved:", out3)
    elif len(ds_present) == 3 and set(ds_present) == {"fever", "debateqa", "conflictqa_popqa"}:
        # One panel per dataset (Qwen/Gemma temp 0.6 rollouts: no StrategyQA in these CSVs).
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        triple = [
            ("(a) FEVER — by model", "fever"),
            ("(b) DebateQA — by model", "debateqa"),
            ("(c) ConflictQA PopQA — by model", "conflictqa_popqa"),
        ]
        for ax, (title, ds) in zip(axes, triple):
            plot_grouped_calibration(
                ax,
                df[df["dataset"] == ds],
                edges,
                "_model_lab",
                model_labels_order,
                title=title,
            )
        out3 = args.out_dir / f"ece_panels_fever_debateqa_popqa_by_model{suffix}.png"
        fig.savefig(out3, dpi=200)
        plt.close(fig)
        print("Saved:", out3)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

        plot_grouped_calibration(
            axes[0],
            df,
            edges,
            "_model_lab",
            model_labels_order,
            title="(a) All datasets pooled — by model",
        )

        df_b = df[df["dataset"] == args.dataset_b]
        plot_grouped_calibration(
            axes[1],
            df_b,
            edges,
            "_model_lab",
            model_labels_order,
            title=f"(b) Dataset: {args.dataset_b}",
        )

        df_c = df[df["dataset"] == args.dataset_c]
        plot_grouped_calibration(
            axes[2],
            df_c,
            edges,
            "_model_lab",
            model_labels_order,
            title=f"(c) Dataset: {args.dataset_c}",
        )

        out3 = args.out_dir / f"ece_panels_3in1_models{suffix}.png"
        fig.savefig(out3, dpi=200)
        plt.close(fig)
        print("Saved:", out3)

    # --- 2) One figure per model: bars = datasets ---
    ds_labels = {
        "fever": "FEVER",
        "debateqa": "DebateQA",
        "conflictqa_popqa": "ConflictQA PopQA",
        "conflictqa_strategyqa": "ConflictQA StrategyQA",
    }
    ds_legend_order = [ds_labels.get(d, d) for d in ds_present]
    df["_ds_lab"] = df["dataset"].map(lambda d: ds_labels.get(d, d))

    for m in models_in_data:
        sub = df[df["model"] == m]
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        plot_grouped_calibration(
            ax,
            sub,
            edges,
            "_ds_lab",
            ds_legend_order,
            title=f"Calibration — {_label(m)}",
        )
        safe = m.replace("/", "_").replace(".", "_")
        outp = args.out_dir / f"ece_per_model_{safe}{suffix}.png"
        fig.savefig(outp, dpi=200)
        plt.close(fig)
        print("Saved:", outp)


if __name__ == "__main__":
    main()
