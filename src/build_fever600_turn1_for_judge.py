"""
Build a single turn-1 rollout CSV for all FEVER-600 claims (for LLM arg-quality judge).

Merges turn-1 rows from expl.csv (and optional t1_expl.csv), reports missing claims.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEVER600 = (
    PROJECT_ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "fever"
    / "csv"
    / "fever600_200x3_complexity_wood_v1_lr_40.csv"
)
EXPL = PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever" / "csv" / "expl.csv"
DEFAULT_OUT = PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever" / "csv" / "fever600_turn1_for_judge.csv"
MISSING_OUT = (
    PROJECT_ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "fever"
    / "csv"
    / "fever600_missing_persuasion.csv"
)


def turn1_with_counterarg(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    t1 = df[df["turn"] == 1].copy()
    t1 = t1[t1["counterargument"].fillna("").astype(str).str.strip() != ""]
    return t1.drop_duplicates(subset="original_index", keep="first")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fever-meta", type=Path, default=FEVER600)
    ap.add_argument("--expl", type=Path, default=EXPL)
    ap.add_argument("--t1-expl", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--missing-out", type=Path, default=MISSING_OUT)
    args = ap.parse_args()

    meta = pd.read_csv(args.fever_meta)
    target_oids = set(meta["original_index"].astype(int))

    parts = [turn1_with_counterarg(args.expl)]
    for p in args.t1_expl or []:
        parts.append(turn1_with_counterarg(Path(p)))

    merged = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if len(merged):
        merged = merged.drop_duplicates(subset="original_index", keep="last")

    have = set(merged["original_index"].astype(int)) if len(merged) else set()
    missing = meta[~meta["original_index"].astype(int).isin(have)].copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if len(merged):
        merged.to_csv(args.out, index=False)

    args.missing_out.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(args.missing_out, index=False)

    print(f"FEVER-600 target: {len(target_oids)}")
    print(f"Turn-1 with counterargument: {len(have)}")
    print(f"Missing persuasion: {len(missing)} -> {args.missing_out}")
    print(f"Judge input CSV: {args.out} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
