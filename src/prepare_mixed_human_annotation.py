"""
Prepare a single human annotation CSV for the mixed-100 persuasion rollout (turn 1).

Output layout:
  output_wood/persuasion/DeepSeek/mixed/human_annotation/
    human_annotation.csv
    readme.txt
    TASK_RU.md
    manifest.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from persuasion_arg_quality import HUMAN_ANNOTATION_GUIDE, TOP_LEVEL_QUALITY_DIMENSIONS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "mixed" / "rollout" / "csv" / "expl.csv"
)
DEFAULT_META = (
    PROJECT_ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "mixed"
    / "csv"
    / "mixed_100_34_33_33_fever_popqa_debateqa.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "mixed" / "human_annotation"
TASK_RU_SOURCE = PROJECT_ROOT / "src" / "mixed_human_annotation_TASK_RU.md"

HUMAN_README_MIXED = """Human annotation — mixed 100
============================

Полная инструкция на русском: TASK_RU.md

Кратко:
  1. claim, target_answer_before, counterargument, target_answer_after (контекст)
  2. Оцените counterargument по трём шкалам 0–3
  3. Баллы — за качество аргумента, не «за flip» ответа

{guide}

File: human_annotation.csv
"""


def dedupe_rollout(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the last dialogue per item_id (handles duplicate runs from resume)."""
    last_dialogue = df.groupby("item_id")["dialogue_id"].last()
    keep_ids = set(last_dialogue.values)
    return df[df["dialogue_id"].isin(keep_ids)].copy()


def join_item_meta(df: pd.DataFrame, meta_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)
    cols = [c for c in ("item_id", "complexity_level", "dataset") if c in meta.columns]
    sub = meta[cols].drop_duplicates("item_id")
    return df.merge(sub, on="item_id", how="left", suffixes=("", "_meta"))


def build_turn1_sample(full_df: pd.DataFrame, meta_path: Path) -> pd.DataFrame:
    deduped = dedupe_rollout(full_df)
    turn1 = deduped[
        (deduped["turn"] == 1)
        & (deduped["counterargument"].fillna("").astype(str).str.strip() != "")
    ].copy()
    turn1 = turn1.drop_duplicates("item_id", keep="last")
    if len(turn1) != 100:
        raise ValueError(f"Expected 100 turn-1 rows, found {len(turn1)}")

    ctx = join_item_meta(turn1, meta_path)
    if "complexity_level_meta" in ctx.columns:
        ctx = ctx.drop(columns=["complexity_level_meta"])
    if "dataset_meta" in ctx.columns:
        ctx["dataset"] = ctx["dataset"].fillna(ctx["dataset_meta"])
        ctx = ctx.drop(columns=["dataset_meta"])

    t0 = deduped[deduped["turn"] == 0][["dialogue_id", "answer"]].rename(
        columns={"answer": "target_answer_before"}
    )
    ctx = ctx.merge(t0, on="dialogue_id", how="left")
    ctx = ctx.rename(columns={"answer": "target_answer_after"})

    base = ctx[
        [
            "item_id",
            "dataset",
            "claim",
            "target_answer_before",
            "counterargument",
            "target_answer_after",
        ]
    ].copy()
    return base.sort_values("item_id").reset_index(drop=True)


def write_human_annotation_files(base: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_df = base.copy()
    for dim in TOP_LEVEL_QUALITY_DIMENSIONS:
        ann_df[dim] = pd.NA

    out_path = out_dir / "human_annotation.csv"
    ann_df.to_csv(out_path, index=False)

    readme = out_dir / "readme.txt"
    readme.write_text(
        HUMAN_README_MIXED.format(guide=HUMAN_ANNOTATION_GUIDE),
        encoding="utf-8",
    )
    (out_dir / "TASK_RU.md").write_text(TASK_RU_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")

    counts = base.groupby("dataset").size().to_dict()
    manifest = out_dir / "manifest.txt"
    manifest.write_text(
        f"scheme=top3 (cogency, effectiveness, reasonableness)\n"
        f"n_rows={len(base)}\n"
        f"turn=1\n"
        f"counts={counts}\n"
        f"annotation_file=human_annotation.csv\n",
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mixed-100 human annotation CSV (turn 1).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.meta.exists():
        raise FileNotFoundError(args.meta)

    base = build_turn1_sample(pd.read_csv(args.input), args.meta)
    out_path = write_human_annotation_files(base, args.out_dir)

    counts = base.groupby("dataset").size().to_dict()
    print(f"Wrote {len(base)} rows -> {out_path}")
    print(f"Per dataset: {counts}")


if __name__ == "__main__":
    main()
