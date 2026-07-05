"""
Build a mixed 100-question subset from FEVER, ConflictQA PopQA, and DebateQA.

Takes an equal random sample from each source (34 + 33 + 33 = 100) using existing
480-row Wood-resplit subsets as the sampling pool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "output_wood" / "dataset_subsampling" / "mixed" / "csv"

FEVER_DEFAULT = (
    PROJECT_ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "fever"
    / "csv"
    / "fever480_160x3_complexity_wood_v1_lr_40_resplit.csv"
)
POPQA_DEFAULT = (
    PROJECT_ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "conflictqa"
    / "csv"
    / "conflictqa_popqa480_160x3_wood_v2_resplit.csv"
)
DEBATEQA_DEFAULT = (
    PROJECT_ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "debateqa"
    / "csv"
    / "debateqa_480_160x3.csv"
)

SEED = 42
TOTAL = 100
N_DATASETS = 3


def equal_split(total: int, n_groups: int) -> list[int]:
    """Split total into n_groups as evenly as possible (e.g. 100 -> [34, 33, 33])."""
    base, rem = divmod(total, n_groups)
    return [base + (1 if i < rem else 0) for i in range(n_groups)]


def random_sample_n(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(f"Not enough rows: need {n}, found {len(df)}")
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def normalize_fever(df: pd.DataFrame) -> pd.DataFrame:
    claim_col = "claim" if "claim" in df.columns else "question"
    return pd.DataFrame(
        {
            "question": df[claim_col].fillna("").astype(str),
            "ground_truth": df["label"].fillna("").astype(str),
            "original_index": pd.to_numeric(df["original_index"], errors="coerce").fillna(0).astype(int),
            "complexity_level": pd.to_numeric(df.get("complexity_level"), errors="coerce"),
            "dataset": "fever",
        }
    )


def normalize_popqa(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": df["question"].fillna("").astype(str),
            "ground_truth": df["ground_truth"].fillna("").astype(str),
            "original_index": pd.to_numeric(df["original_index"], errors="coerce").fillna(0).astype(int),
            "complexity_level": pd.to_numeric(df.get("complexity_level"), errors="coerce"),
            "dataset": "conflictqa_popqa",
        }
    )


def normalize_debateqa(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": df["question"].fillna("").astype(str),
            "ground_truth": df["ground_truth"].fillna("").astype(str),
            "original_index": pd.to_numeric(df["original_index"], errors="coerce").fillna(0).astype(int),
            "complexity_level": pd.to_numeric(df.get("complexity_level"), errors="coerce"),
            "dataset": "debateqa",
        }
    )


def build_mixed(
    fever_path: Path,
    popqa_path: Path,
    debateqa_path: Path,
    total: int,
    seed: int,
) -> pd.DataFrame:
    counts = equal_split(total, N_DATASETS)
    loaders = [
        (normalize_fever(pd.read_csv(fever_path)), counts[0], seed),
        (normalize_popqa(pd.read_csv(popqa_path)), counts[1], seed + 1),
        (normalize_debateqa(pd.read_csv(debateqa_path)), counts[2], seed + 2),
    ]

    parts = [random_sample_n(df, n, s) for df, n, s in loaders]
    mixed = pd.concat(parts, ignore_index=True)
    mixed = mixed.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    mixed.insert(0, "item_id", range(1, len(mixed) + 1))
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mixed 100-question subset (equal random sample from FEVER, PopQA, DebateQA)."
    )
    parser.add_argument("--fever-input", type=Path, default=FEVER_DEFAULT)
    parser.add_argument("--popqa-input", type=Path, default=POPQA_DEFAULT)
    parser.add_argument("--debateqa-input", type=Path, default=DEBATEQA_DEFAULT)
    parser.add_argument("--total", type=int, default=TOTAL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: mixed_100_... under dataset_subsampling/mixed/csv).",
    )
    args = parser.parse_args()

    for path in (args.fever_input, args.popqa_input, args.debateqa_input):
        if not path.exists():
            raise FileNotFoundError(path)

    mixed = build_mixed(
        args.fever_input,
        args.popqa_input,
        args.debateqa_input,
        args.total,
        args.seed,
    )

    split = equal_split(args.total, N_DATASETS)
    out_path = args.out or (OUT_DIR / f"mixed_{args.total}_{split[0]}_{split[1]}_{split[2]}_fever_popqa_debateqa.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_csv(out_path, index=False)

    manifest = out_path.with_suffix(".manifest.txt")
    counts = mixed.groupby("dataset").size().to_dict()
    manifest.write_text(
        f"total={len(mixed)}\n"
        f"split_per_dataset={split}\n"
        f"sampling=random (no complexity stratification)\n"
        f"seed={args.seed}\n"
        f"counts={json.dumps(counts, ensure_ascii=False)}\n"
        f"sources:\n"
        f"  fever: {args.fever_input}\n"
        f"  popqa: {args.popqa_input}\n"
        f"  debateqa: {args.debateqa_input}\n"
        f"output: {out_path}\n",
        encoding="utf-8",
    )

    print(f"Saved {len(mixed)} rows to {out_path}")
    print(f"Per dataset: {counts}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
