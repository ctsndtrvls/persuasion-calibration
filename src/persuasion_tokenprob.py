"""Load and merge DeepSeek token-probability confidence for persuasion FEVER dialogues."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSITE_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever" / "composite_score"
DEFAULT_FEATURES = COMPOSITE_DIR / "01_features" / "csv" / "persuasion_uncertainty_features.csv"
TOKEN_DIR = _PROJECT_ROOT / "output_wood" / "token_prob_confidence" / "deepseek" / "csv"

FEVER480_TOKENPROB = TOKEN_DIR / (
    "deepseek_tokenprob_fever480_160x3_complexity_wood_v1_lr_40_resplit_20260504_162309.csv"
)
FEVER600_SUPPLEMENT = TOKEN_DIR / "deepseek_tokenprob_fever600_persuasion214_missing41_20260714.csv"
PERSUASION214_TOKENPROB = TOKEN_DIR / "deepseek_tokenprob_persuasion214_full.csv"


def _normalize_tokenprob_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "original_index" not in out.columns and "row_key" in out.columns:
        out = out.rename(columns={"row_key": "original_index"})
    out["original_index"] = pd.to_numeric(out["original_index"], errors="coerce").astype("Int64")
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["U_token"] = 1.0 - out["confidence"]
    return out.dropna(subset=["original_index"]).astype({"original_index": int})


def merge_persuasion_tokenprob(
    fever480_csv: Path = FEVER480_TOKENPROB,
    supplement_csv: Path = FEVER600_SUPPLEMENT,
    out_csv: Path = PERSUASION214_TOKENPROB,
    original_indices: pd.Series | None = None,
    features_csv: Path = DEFAULT_FEATURES,
) -> pd.DataFrame:
    """Merge fever480 token-prob rows with the fever600 supplement for persuasion214."""
    if original_indices is None and features_csv.exists():
        original_indices = pd.read_csv(features_csv)["original_index"]

    parts = [_normalize_tokenprob_df(pd.read_csv(fever480_csv))]
    if supplement_csv.exists():
        parts.append(_normalize_tokenprob_df(pd.read_csv(supplement_csv)))

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["original_index"], keep="last")

    if original_indices is not None:
        idx = pd.to_numeric(original_indices, errors="coerce").dropna().astype(int).unique()
        merged = merged[merged["original_index"].isin(idx)].copy()

    keep_cols = [
        c
        for c in (
            "original_index",
            "model",
            "dataset",
            "question",
            "answer",
            "confidence",
            "U_token",
            "correct",
            "error",
        )
        if c in merged.columns
    ]
    merged = merged[keep_cols].sort_values("original_index").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def load_persuasion_tokenprob(
    token_csv: Path = PERSUASION214_TOKENPROB,
    original_indices: pd.Series | None = None,
) -> pd.DataFrame:
    """Load merged persuasion214 token-prob table, building it if missing."""
    if not token_csv.exists():
        merge_persuasion_tokenprob(out_csv=token_csv, original_indices=original_indices)
    df = _normalize_tokenprob_df(pd.read_csv(token_csv))
    if original_indices is not None:
        idx = pd.to_numeric(original_indices, errors="coerce").dropna().astype(int).unique()
        df = df[df["original_index"].isin(idx)].copy()
    return df.reset_index(drop=True)
