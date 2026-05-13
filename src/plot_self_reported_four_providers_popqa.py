"""
One bar chart: mean self-reported confidence for OpenAI, DeepSeek, Qwen, Gemma
on the **ConflictQA PopQA** subset (480 rows per model), so counts match.

Data sources (defaults, same Wood resplits as other experiments):
  - OpenAI / DeepSeek: temperature_experiments … *_conflictqa_temp06_scale10.csv
    (confidence already in [0,1] in CSV).
  - Qwen / Gemma: temperature_experiments/qwen_gemma_fever_popqa_debateqa/csv/*_temp0p6_scale10_popqa.csv
    rows with dataset == conflictqa_popqa only.

Example:
  python3 src/plot_self_reported_four_providers_popqa.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OPENAI = (
    PROJECT_ROOT
    / "output_wood"
    / "temperature_experiments"
    / "conflictqa"
    / "csv"
    / "conflictqa_ece_openai_conflictqa_temp06_scale10.csv"
)
DEFAULT_DEEPSEEK = (
    PROJECT_ROOT
    / "output_wood"
    / "temperature_experiments"
    / "conflictqa"
    / "csv"
    / "conflictqa_ece_deepseek_conflictqa_temp06_scale10.csv"
)
DEFAULT_QWEN_GEMMA = (
    PROJECT_ROOT
    / "output_wood"
    / "temperature_experiments"
    / "qwen_gemma_fever_popqa_debateqa"
    / "csv"
    / "ece_qwen_gemma_fever_popqa_debateqa_temp0p6_scale10_popqa.csv"
)

# X-axis order (user request).
MODEL_ORDER: list[tuple[str, str]] = [
    ("openai/gpt-4o-2024-11-20", "GPT-4o"),
    ("deepseek/deepseek-chat-v2.5", "DeepSeek Chat v2.5"),
    ("qwen/qwen3-14b", "Qwen3-14B"),
    ("google/gemma-4-26b-a4b-it", "Gemma 4 26B (A4B IT)"),
]

DATASET_TAG = "conflictqa_popqa"


def _load_confidence(path: Path, model_id: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "dataset" not in df.columns or "confidence" not in df.columns or "model" not in df.columns:
        raise ValueError(f"{path}: need columns model, dataset, confidence")
    sub = df[df["dataset"].astype(str) == DATASET_TAG].copy()
    if model_id is not None:
        sub = sub[sub["model"].astype(str) == model_id]
    sub["confidence"] = pd.to_numeric(sub["confidence"], errors="coerce")
    sub = sub.dropna(subset=["confidence"])
    sub = sub[(sub["confidence"] >= 0.0) & (sub["confidence"] <= 1.0)]
    return sub


def build_frame(
    openai_csv: Path,
    deepseek_csv: Path,
    qwen_gemma_csv: Path,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    parts.append(_load_confidence(openai_csv, "openai/gpt-4o-2024-11-20"))
    parts.append(_load_confidence(deepseek_csv, "deepseek/deepseek-chat-v2.5"))
    qg = _load_confidence(qwen_gemma_csv)
    for mid, _ in MODEL_ORDER[2:]:
        parts.append(qg[qg["model"].astype(str) == mid])
    out = pd.concat(parts, ignore_index=True)
    label_map = dict(MODEL_ORDER)
    out["model_label"] = out["model"].map(lambda m: label_map.get(str(m), str(m)))
    return out


def save_plot(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    order_labels = [lbl for _, lbl in MODEL_ORDER]
    overall = (
        df.groupby("model_label", as_index=False)["confidence"]
        .mean()
        .rename(columns={"confidence": "mean_confidence"})
    )
    overall["model_label"] = pd.Categorical(overall["model_label"], categories=order_labels, ordered=True)
    overall = overall.sort_values("model_label")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    sns.barplot(
        data=overall,
        x="model_label",
        y="mean_confidence",
        hue="model_label",
        legend=False,
        palette="viridis",
        order=order_labels,
        ax=ax,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean self-reported confidence")
    ax.set_title(
        "Self-reported confidence: OpenAI, DeepSeek, Qwen, Gemma\n"
        f"(ConflictQA PopQA subset, N={len(df) // len(MODEL_ORDER)} per model)"
    )
    ax.tick_params(axis="x", rotation=12)
    fig.text(
        0.5,
        0.01,
        "OpenAI & DeepSeek: rollout temp 0.6, confidence scale 1–10 (stored as [0,1] in CSV). "
        "Qwen & Gemma: OpenRouter self-reported JSON in [0,1], temp 0.6, scale 1–10 (same rollouts as fever/debateqa/popqa plot).",
        ha="center",
        fontsize=9.5,
        color="0.35",
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openai-csv", type=Path, default=DEFAULT_OPENAI)
    parser.add_argument("--deepseek-csv", type=Path, default=DEFAULT_DEEPSEEK)
    parser.add_argument("--qwen-gemma-csv", type=Path, default=DEFAULT_QWEN_GEMMA)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "output_wood" / "self_reported_confidence",
    )
    parser.add_argument(
        "--out-name",
        default="self_reported_confidence_four_models_conflictqa_popqa.png",
        help="Output PNG filename",
    )
    args = parser.parse_args()

    df = build_frame(args.openai_csv, args.deepseek_csv, args.qwen_gemma_csv)
    for mid, lbl in MODEL_ORDER:
        n = len(df[df["model"].astype(str) == mid])
        if n == 0:
            raise ValueError(f"No rows for model {mid} ({lbl}) after filtering {DATASET_TAG}")

    summary = (
        df.groupby(["model", "model_label"], as_index=False)["confidence"]
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
    summary["dataset_slice"] = DATASET_TAG

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    csv_dir = out_dir / "csv"
    png_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / args.out_name
    csv_path = csv_dir / (Path(args.out_name).stem + "_summary.csv")

    save_plot(df, png_path)
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {png_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
