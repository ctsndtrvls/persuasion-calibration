"""
Mean self-reported confidence for four models (OpenAI, DeepSeek, Qwen, Gemma)
on three slices (temp 0.6, verbal scale 1–10 where applicable):

  - FEVER
  - DebateQA
  - ConflictQA — **PopQA only** (no StrategyQA)

OpenAI / DeepSeek: legacy rollout CSVs under output_wood/temperature_experiments/
  (fever + popqa split + DebateQA OpenAI+DeepSeek combined, filtered per model).

Qwen / Gemma: output_wood/temperature_experiments/qwen_gemma_fever_popqa_debateqa/…/
  ece_qwen_gemma_*_temp0p6_scale10.csv (same regime).

Writes:
  - png/self_reported_confidence_four_models_fever_debateqa_popqa_temp06_scale10.png
  - csv/self_reported_confidence_four_models_fever_debateqa_popqa_temp06_scale10_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "output_wood" / "self_reported_confidence"
TE = PROJECT_ROOT / "output_wood" / "temperature_experiments"

DEFAULT_OPENAI_FEVER = TE / "fever" / "csv" / "conflictqa_ece_openai_fever_temp06_scale10.csv"
DEFAULT_OPENAI_POPQA = TE / "conflictqa" / "csv" / "conflictqa_ece_openai_temp06_scale10_popqa.csv"
DEFAULT_DEEPSEEK_FEVER = TE / "fever" / "csv" / "conflictqa_ece_deepseek_fever_temp06_scale10.csv"
DEFAULT_DEEPSEEK_POPQA = TE / "conflictqa" / "csv" / "conflictqa_ece_deepseek_temp06_scale10_popqa.csv"
DEFAULT_DEBATEQA_OD = TE / "debateqa" / "csv" / "debateqa_ece_openai_deepseek_temp0p6_scale10.csv"
DEFAULT_QWEN_GEMMA = (
    TE / "qwen_gemma_fever_popqa_debateqa" / "csv" / "ece_qwen_gemma_fever_popqa_debateqa_temp0p6_scale10.csv"
)

MODEL_ORDER: list[tuple[str, str]] = [
    ("openai/gpt-4o-2024-11-20", "GPT-4o"),
    ("deepseek/deepseek-chat-v2.5", "DeepSeek Chat v2.5"),
    ("qwen/qwen3-14b", "Qwen3-14B"),
    ("google/gemma-4-26b-a4b-it", "Gemma 4 26B"),
]

# Canonical dataset tags in rollout CSVs
TAG_FEVER = "fever"
TAG_DEBATEQA = "debateqa"
TAG_POPQA = "conflictqa_popqa"

DATASET_HUE_ORDER = ["FEVER", "DebateQA", "PopQA"]


def _tag_to_hue(ds: str) -> str:
    s = str(ds).strip()
    if s == TAG_FEVER:
        return "FEVER"
    if s == TAG_DEBATEQA:
        return "DebateQA"
    if s == TAG_POPQA:
        return "PopQA"
    return s


def _load_cols(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"model", "dataset", "confidence"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{path}: missing {miss}")
    out = df[list(need)].copy()
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out = out.dropna(subset=["confidence"])
    out = out[(out["confidence"] >= 0.0) & (out["confidence"] <= 1.0)]
    return out


def _take_model(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    sub = df[df["model"].astype(str) == model_id].copy()
    sub["dataset_hue"] = sub["dataset"].map(_tag_to_hue)
    return sub


def build_frame(
    openai_fever: Path,
    openai_popqa: Path,
    deepseek_fever: Path,
    deepseek_popqa: Path,
    debateqa_od: Path,
    qwen_gemma: Path,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    of = _load_cols(openai_fever)
    op = _load_cols(openai_popqa)
    parts.append(_take_model(of, MODEL_ORDER[0][0]))
    parts.append(_take_model(op, MODEL_ORDER[0][0]))

    df_d = _load_cols(debateqa_od)
    parts.append(_take_model(df_d, MODEL_ORDER[0][0]))

    df_sf = _load_cols(deepseek_fever)
    df_sp = _load_cols(deepseek_popqa)
    parts.append(_take_model(df_sf, MODEL_ORDER[1][0]))
    parts.append(_take_model(df_sp, MODEL_ORDER[1][0]))
    parts.append(_take_model(df_d, MODEL_ORDER[1][0]))

    qg = _load_cols(qwen_gemma)
    qg = qg[qg["dataset"].astype(str).isin({TAG_FEVER, TAG_DEBATEQA, TAG_POPQA})].copy()
    qg["dataset_hue"] = qg["dataset"].map(_tag_to_hue)
    for mid, _ in MODEL_ORDER[2:]:
        parts.append(qg[qg["model"].astype(str) == mid].copy())

    out = pd.concat(parts, ignore_index=True)
    out["dataset_hue"] = pd.Categorical(out["dataset_hue"], categories=DATASET_HUE_ORDER, ordered=True)
    label_map = dict(MODEL_ORDER)
    out["model_label"] = out["model"].map(lambda m: label_map.get(str(m), str(m)))
    out["model_label"] = pd.Categorical(
        out["model_label"], categories=[lbl for _, lbl in MODEL_ORDER], ordered=True
    )
    return out


def save_plot(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    by = (
        df.groupby(["model_label", "dataset_hue"], as_index=False, observed=True)["confidence"]
        .mean()
        .rename(columns={"confidence": "mean_confidence"})
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    sns.barplot(
        data=by,
        x="model_label",
        y="mean_confidence",
        hue="dataset_hue",
        hue_order=DATASET_HUE_ORDER,
        palette="Set2",
        ax=ax,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean self-reported confidence")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--openai-fever", type=Path, default=DEFAULT_OPENAI_FEVER)
    ap.add_argument("--openai-popqa", type=Path, default=DEFAULT_OPENAI_POPQA)
    ap.add_argument("--deepseek-fever", type=Path, default=DEFAULT_DEEPSEEK_FEVER)
    ap.add_argument("--deepseek-popqa", type=Path, default=DEFAULT_DEEPSEEK_POPQA)
    ap.add_argument("--debateqa-openai-deepseek", type=Path, default=DEFAULT_DEBATEQA_OD)
    ap.add_argument("--qwen-gemma", type=Path, default=DEFAULT_QWEN_GEMMA)
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    df = build_frame(
        args.openai_fever,
        args.openai_popqa,
        args.deepseek_fever,
        args.deepseek_popqa,
        args.debateqa_openai_deepseek,
        args.qwen_gemma,
    )
    for mid, lbl in MODEL_ORDER:
        n = len(df[df["model"].astype(str) == mid])
        if n == 0:
            raise ValueError(f"No rows for {mid} ({lbl})")

    summary = (
        df.groupby(["model", "model_label", "dataset_hue"], as_index=False, observed=True)["confidence"]
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

    png_dir = args.out_dir / "png"
    csv_dir = args.out_dir / "csv"
    png_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    stem = "self_reported_confidence_four_models_fever_debateqa_popqa_temp06_scale10"
    png_path = png_dir / f"{stem}.png"
    csv_path = csv_dir / f"{stem}_summary.csv"

    save_plot(df, png_path)
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {png_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
