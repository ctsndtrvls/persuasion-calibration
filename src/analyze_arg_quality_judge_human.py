"""
Compare LLM-judge scores with human annotations.

Pass one completed annotator CSV, or three separate files (human_annotation_annotator_*.csv).

Example:
  python3 analyze_arg_quality_judge_human.py \\
    --human ../output_wood/.../human_annotation_annotator_1.csv \\
           ../output_wood/.../human_annotation_annotator_2.csv \\
           ../output_wood/.../human_annotation_annotator_3.csv \\
    --judge-long ../output_wood/.../arg_quality_long_fever214_no_flip_t1_v1__openrouter__openai_gpt-5.4-mini.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from persuasion_arg_quality import (
    QUALITY_DIMENSIONS,
    TOP_LEVEL_QUALITY_DIMENSIONS,
)


def detect_score_dimensions(human: pd.DataFrame) -> tuple[str, ...]:
    if all(d in human.columns for d in TOP_LEVEL_QUALITY_DIMENSIONS):
        return TOP_LEVEL_QUALITY_DIMENSIONS
    if all(d in human.columns for d in QUALITY_DIMENSIONS):
        return QUALITY_DIMENSIONS
    legacy = [c.split("__", 1)[1] for c in human.columns if c.startswith("annotator_") and "__" in c]
    if legacy:
        return tuple(dict.fromkeys(legacy))
    raise SystemExit("Could not detect score columns in human CSV.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def human_majority_from_files(human_paths: list[Path], dim: str) -> pd.Series:
    scores_by_id: dict[str, list[float]] = {}
    for path in human_paths:
        h = pd.read_csv(path)
        if dim not in h.columns:
            continue
        for _, r in h.iterrows():
            eid = str(r["eval_id"])
            if pd.notna(r[dim]):
                scores_by_id.setdefault(eid, []).append(float(r[dim]))
    return pd.Series(
        {eid: float(pd.Series(v).mode().iloc[0]) for eid, v in scores_by_id.items() if v}
    )


def human_majority(row: pd.Series, dim: str) -> float:
    legacy = [f"annotator_{a}__{dim}" for a in ("a", "b", "c")]
    cols = [c for c in legacy if c in row.index] or ([dim] if dim in row.index else [])
    vals = [row[c] for c in cols if pd.notna(row[c])]
    if not vals:
        return np.nan
    return float(pd.Series(vals).mode().iloc[0])


def weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 5:
        return float("nan")
    return float(cohen_kappa_score(a[mask].astype(int), b[mask].astype(int), weights="quadratic"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--human",
        type=Path,
        nargs="+",
        help="One or more completed human CSVs (e.g. three annotator_* files).",
    )
    ap.add_argument("--judge-long", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    judge = pd.read_csv(args.judge_long)
    human_paths = list(args.human)
    if len(human_paths) == 1:
        human = pd.read_csv(human_paths[0])
        merged = human.merge(judge, on="eval_id", how="inner", suffixes=("_human", "_judge"))
    else:
        base = pd.read_csv(human_paths[0])[["eval_id"]]
        merged = base.merge(judge, on="eval_id", how="inner")
    if merged.empty:
        raise SystemExit("No overlapping eval_id between human and judge files.")

    rows = []
    score_dims = detect_score_dimensions(pd.read_csv(human_paths[0]))
    for dim in score_dims:
        if len(human_paths) > 1:
            maj = human_majority_from_files(human_paths, dim)
            merged[f"human_majority__{dim}"] = merged["eval_id"].map(maj)
        else:
            merged[f"human_majority__{dim}"] = merged.apply(lambda r: human_majority(r, dim), axis=1)
        h = merged[f"human_majority__{dim}"].to_numpy(dtype=float)
        j = pd.to_numeric(merged[dim], errors="coerce").to_numpy(dtype=float)
        mask = ~(np.isnan(h) | np.isnan(j))
        if mask.sum() == 0:
            continue
        pearson = float(np.corrcoef(h[mask], j[mask])[0, 1]) if mask.sum() > 2 else float("nan")
        kappa = weighted_kappa(h, j)
        mae = float(np.mean(np.abs(h[mask] - j[mask])))
        rows.append(
            {
                "dimension": dim,
                "n": int(mask.sum()),
                "pearson": pearson,
                "quadratic_kappa": kappa,
                "mae": mae,
            }
        )

    summary = pd.DataFrame(rows)
    out = args.out or human_paths[0].with_name(
        "human_vs_" + args.judge_long.stem + "_agreement.csv"
    )
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
