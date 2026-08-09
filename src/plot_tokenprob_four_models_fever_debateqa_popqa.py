#!/usr/bin/env python3
"""
Assemble four-model token-probability ECE figures for Stage 4:

  GPT-4o, DeepSeek, Gemma, Qwen × FEVER, DebateQA, ConflictQA PopQA

Uses existing CSVs under output_wood/token_prob_confidence/ (no new API calls).
Aligns rows to the canonical 480-item Wood subsets where original_index/row_key
is available; deduplicates Qwen/Gemma multi-rows to one per item.

Writes:
  output_wood/token_prob_confidence/summary/csv/
  output_wood/token_prob_confidence/summary/png/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import plot_ece_calibration as pec  # noqa: E402
from collect_conflictqa_ece import (  # noqa: E402
    _parse_ground_truth_cell,
    is_correct,
)

TP = PROJECT_ROOT / "output_wood" / "token_prob_confidence"
OUT_CSV = TP / "summary" / "csv"
OUT_PNG = TP / "summary" / "png"

CANON = {
    "fever": PROJECT_ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "fever"
    / "csv"
    / "fever480_160x3_complexity_wood_v1_lr_40_resplit.csv",
    "conflictqa_popqa": PROJECT_ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "conflictqa"
    / "csv"
    / "conflictqa_popqa480_160x3_wood_v2_resplit.csv",
    "debateqa": PROJECT_ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "debateqa"
    / "csv"
    / "debateqa_480_160x3.csv",
}

MODEL_CANON = {
    "openai/gpt-4o-2024-11-20": "openai/gpt-4o-2024-11-20",
    "deepseek/deepseek-chat-v2.5": "deepseek/deepseek-chat-v2.5",
    "deepseek-chat": "deepseek/deepseek-chat-v2.5",
    "deepseek/deepseek-chat": "deepseek/deepseek-chat-v2.5",
    "google/gemma-4-26b-a4b-it": "google/gemma-4-26b-a4b-it",
    "qwen/qwen3-14b": "qwen/qwen3-14b",
}

MODEL_ORDER = [
    "openai/gpt-4o-2024-11-20",
    "deepseek/deepseek-chat-v2.5",
    "google/gemma-4-26b-a4b-it",
    "qwen/qwen3-14b",
]

KEEP_DS = ["fever", "debateqa", "conflictqa_popqa"]
DS_LABELS = {
    "fever": "FEVER",
    "debateqa": "DebateQA",
    "conflictqa_popqa": "ConflictQA PopQA",
}


def _canon_gt(ds: str) -> pd.DataFrame:
    """original_index → ground_truth list for scoring."""
    df = pd.read_csv(CANON[ds])
    out = pd.DataFrame(
        {
            "original_index": pd.to_numeric(df["original_index"], errors="coerce").astype(int),
        }
    )
    if ds == "fever":
        # FEVER gold is the label column
        label_col = "label" if "label" in df.columns else "answer"
        out["ground_truth"] = df[label_col].fillna("").astype(str).map(lambda x: [x])
    else:
        out["ground_truth"] = df["ground_truth"].map(_parse_ground_truth_cell)
    return out


def _norm(df: pd.DataFrame, *, dataset: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if dataset is not None:
        out["dataset"] = dataset
    if "dataset" not in out.columns:
        raise ValueError("dataset column required")
    if "answer" not in out.columns:
        raise ValueError("answer column required for re-scoring")
    out["model"] = out["model"].astype(str).map(lambda m: MODEL_CANON.get(m, m))
    out["dataset"] = out["dataset"].astype(str)
    out["answer"] = out["answer"].fillna("").astype(str)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out = out.dropna(subset=["confidence"])
    out = out[(out["confidence"] >= 0) & (out["confidence"] <= 1)]
    if "original_index" not in out.columns and "row_key" in out.columns:
        out["original_index"] = pd.to_numeric(out["row_key"], errors="coerce")
    elif "original_index" in out.columns:
        out["original_index"] = pd.to_numeric(out["original_index"], errors="coerce")
    return out


def build_frame() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    # OpenAI FEVER / PopQA
    parts.append(_norm(pd.read_csv(TP / "openai/csv/fever_ece_openai_token_prob.csv")))
    parts.append(_norm(pd.read_csv(TP / "openai/csv/conflictqa_ece_openai_token_prob_popqa.csv")))

    # DeepSeek FEVER (480) + PopQA (600 → filter to 480)
    parts.append(
        _norm(
            pd.read_csv(
                TP
                / "deepseek/csv/deepseek_tokenprob_fever480_160x3_complexity_wood_v1_lr_40_resplit_20260504_162309.csv"
            )
        )
    )
    parts.append(
        _norm(
            pd.read_csv(
                TP
                / "deepseek/csv/deepseek_tokenprob_conflictqa_popqa600_200x3_wood_wood_conflictqa_v2_20260504_103903.csv"
            ),
            dataset="conflictqa_popqa",
        )
    )

    # Qwen + Gemma (FEVER / PopQA / StrategyQA in combined file)
    parts.append(_norm(pd.read_csv(TP / "qwen-gemma/csv/tokenprob_new_models_v2.csv")))

    # DebateQA four models (temp 0.0)
    parts.append(_norm(pd.read_csv(TP / "debateqa/csv/debateqa_tokenprob_four_models_temp0p0.csv")))

    raw = pd.concat(parts, ignore_index=True)
    raw = raw[raw["model"].isin(MODEL_ORDER) & raw["dataset"].isin(KEEP_DS)].copy()

    # Align to canonical 480 indices and re-score correctness vs gold
    aligned: list[pd.DataFrame] = []
    for ds in KEEP_DS:
        gt = _canon_gt(ds)
        ids = set(gt["original_index"].tolist())
        sub = raw[raw["dataset"] == ds].copy()
        sub = sub[sub["original_index"].isin(ids)].copy()
        sub = sub.sort_values(["model", "original_index"]).drop_duplicates(
            subset=["model", "original_index"], keep="first"
        )
        sub = sub.merge(gt, on="original_index", how="left")
        sub["correct"] = [
            is_correct(a, g if isinstance(g, list) else [])
            for a, g in zip(sub["answer"].tolist(), sub["ground_truth"].tolist())
        ]
        aligned.append(sub)

    out = pd.concat(aligned, ignore_index=True)
    return out[["model", "dataset", "original_index", "confidence", "correct"]]


def ece_summary(df: pd.DataFrame) -> pd.DataFrame:
    edges = pec.bin_edges()
    rows = []
    for mid in MODEL_ORDER:
        for ds in KEEP_DS:
            g = df[(df["model"] == mid) & (df["dataset"] == ds)]
            if g.empty:
                continue
            conf = g["confidence"].to_numpy(float)
            cor = g["correct"].to_numpy(float)
            _, acc, mean_c, counts = pec.compute_bin_stats(conf, cor, edges)
            ece = pec.ece_from_bins(mean_c, acc, counts)
            rows.append(
                {
                    "model": pec._label(mid),
                    "model_id": mid,
                    "dataset": ds,
                    "dataset_label": DS_LABELS[ds],
                    "n": len(g),
                    "ece": round(float(ece), 3),
                    "accuracy": round(float(cor.mean()), 3),
                    "mean_conf": round(float(conf.mean()), 3),
                }
            )
    return pd.DataFrame(rows)


def plot_panels(df: pd.DataFrame, out_path: Path) -> None:
    edges = pec.bin_edges()
    # Ensure DeepSeek label matches canonical id
    pec.MODEL_LABELS["deepseek/deepseek-chat-v2.5"] = "DeepSeek Chat v2.5"

    df = df.copy()
    df["_model_lab"] = df["model"].map(pec._label)
    labels = [pec._label(m) for m in MODEL_ORDER if m in set(df["model"])]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    panels = [
        ("(a) FEVER — by model (token prob)", "fever"),
        ("(b) DebateQA — by model (token prob)", "debateqa"),
        ("(c) ConflictQA PopQA — by model (token prob)", "conflictqa_popqa"),
    ]
    for ax, (title, ds) in zip(axes, panels):
        pec.plot_grouped_calibration(
            ax,
            df[df["dataset"] == ds],
            edges,
            "_model_lab",
            labels,
            title=title,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)


def plot_per_model(df: pd.DataFrame, out_dir: Path) -> None:
    edges = pec.bin_edges()
    ds_legend = [DS_LABELS[d] for d in KEEP_DS]
    df = df.copy()
    df["_ds_lab"] = df["dataset"].map(DS_LABELS)
    for mid in MODEL_ORDER:
        sub = df[df["model"] == mid]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        pec.plot_grouped_calibration(
            ax,
            sub,
            edges,
            "_ds_lab",
            ds_legend,
            title=f"Calibration — {pec._label(mid)} (token prob)",
        )
        safe = mid.replace("/", "_").replace(".", "_")
        outp = out_dir / f"ece_per_model_{safe}_token_prob_fever_debateqa_popqa.png"
        fig.savefig(outp, dpi=200)
        plt.close(fig)
        print("Saved:", outp)


def main() -> None:
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    OUT_PNG.mkdir(parents=True, exist_ok=True)

    df = build_frame()
    print(df.groupby(["model", "dataset"]).size().unstack(fill_value=0))

    combined = OUT_CSV / "tokenprob_four_models_fever_debateqa_popqa.csv"
    df.to_csv(combined, index=False)
    print("Wrote", combined, "n=", len(df))

    summary = ece_summary(df)
    summary_path = OUT_CSV / "tokenprob_four_models_fever_debateqa_popqa_ece_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print("Wrote", summary_path)

    plot_panels(
        df,
        OUT_PNG / "ece_panels_fever_debateqa_popqa_by_model_token_prob_four_models.png",
    )
    plot_per_model(df, OUT_PNG)


if __name__ == "__main__":
    main()
