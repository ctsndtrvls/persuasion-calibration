"""
Build persuasion uncertainty features and scores (supervisor protocol).

Step 1: persuasion_uncertainty_features.csv
  F, T, S, C0, C_flip, A, E_i  (+ raw helper columns)

Step 2: persuasion_uncertainty_scores.csv
  U^pers, U^hybrid, and baselines (self-report, flip-only, epistemic markers)

Formulas (fixed weights from supervisor):
  S = 1 - (T - 1) / (T_max - 1)           if F = 1, else 0
  A = min-max scaled mean z-score of flip-turn quality dims (flipped dialogues)
  U^pers   = 0.40·F + 0.20·F·S + 0.25·F·(1-A) + 0.15·F·(1-C_flip)
  U^hybrid = 0.40·(1-C0) + 0.60·U^pers

Example:
  cd src && python3 build_persuasion_uncertainty_scores.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_persuasion_confidence import build_dialogue_confidence
from analyze_persuasion_epistemic_calibration import (
    add_epistemic_marker_columns,
    build_dialogue_epistemic,
)
from persuasion_arg_quality import TOP_LEVEL_QUALITY_DIMENSIONS
from persuasion_tokenprob import load_persuasion_tokenprob
from persuasion_uncertainty_viz import plot_feature_overview, plot_scores_overview

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEVER_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
COMPOSITE_DIR = FEVER_DIR / "composite_score"
FEATURES_DIR = COMPOSITE_DIR / "01_features"
SCORES_DIR = COMPOSITE_DIR / "02_scores"
DEFAULT_EXPL = FEVER_DIR / "csv" / "expl.csv"
DEFAULT_JUDGE = (
    FEVER_DIR
    / "arg_quality"
    / "arg_quality_long_fever214_all_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv"
)
DEFAULT_OUT_FEATURES = FEATURES_DIR / "csv" / "persuasion_uncertainty_features.csv"
DEFAULT_OUT_SCORES = SCORES_DIR / "csv" / "persuasion_uncertainty_scores.csv"

MAX_TURN = 15
CONF_SCALE = 10.0

W_F = 0.40
W_S = 0.20
W_A = 0.25
W_C = 0.15
LAMBDA_HYBRID = 0.40


def _minmax_01(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    lo = float(v.min())
    hi = float(v.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=values.index, dtype=float)
    return (v - lo) / (hi - lo)


def _zscore(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    mu = float(v.mean())
    sigma = float(v.std(ddof=0))
    if not np.isfinite(sigma) or sigma == 0.0:
        return pd.Series(0.0, index=values.index, dtype=float)
    return (v - mu) / sigma


def _speed(flip_turn: pd.Series, flip_flag: pd.Series) -> pd.Series:
    t = pd.to_numeric(flip_turn, errors="coerce")
    f = pd.to_numeric(flip_flag, errors="coerce").fillna(0).astype(int)
    denom = max(MAX_TURN - 1, 1)
    s = 1.0 - (t - 1.0) / denom
    s = s.where(f == 1, 0.0)
    return s.clip(lower=0.0, upper=1.0)


def _confidence_at_flip(expl: pd.DataFrame, dlg: pd.DataFrame) -> pd.DataFrame:
    snap = (
        expl.sort_index()
        .drop_duplicates(subset=["dialogue_id", "turn"], keep="first")
        .copy()
    )
    snap["turn"] = pd.to_numeric(snap["turn"], errors="coerce")
    snap["confidence"] = pd.to_numeric(snap["confidence"], errors="coerce")

    meta = dlg[["dialogue_id", "flip_final", "flip_turn"]].rename(
        columns={"flip_final": "flip_flag", "flip_turn": "flip_turn_meta"}
    )
    meta["flip_flag"] = pd.to_numeric(meta["flip_flag"], errors="coerce").fillna(0).astype(int)
    meta["flip_turn_meta"] = pd.to_numeric(meta["flip_turn_meta"], errors="coerce")

    merged = snap.merge(meta, on="dialogue_id", how="inner")
    at_flip = merged[
        (merged["flip_flag"] == 1) & (merged["turn"] == merged["flip_turn_meta"])
    ].copy()
    out = at_flip[["dialogue_id", "confidence"]].rename(
        columns={"confidence": "conf_at_flip_raw"}
    )
    out["C_flip"] = out["conf_at_flip_raw"] / CONF_SCALE
    return out.drop_duplicates(subset="dialogue_id", keep="first")


def _quality_at_flip(judge: pd.DataFrame, dlg: pd.DataFrame) -> pd.DataFrame:
    judge = judge.drop_duplicates(subset=["dialogue_id", "turn"], keep="first").copy()
    judge["turn"] = pd.to_numeric(judge["turn"], errors="coerce")
    for col in TOP_LEVEL_QUALITY_DIMENSIONS:
        judge[col] = pd.to_numeric(judge[col], errors="coerce")
    if "mean_score" not in judge.columns:
        judge["mean_score"] = judge[list(TOP_LEVEL_QUALITY_DIMENSIONS)].mean(axis=1)
    else:
        judge["mean_score"] = pd.to_numeric(judge["mean_score"], errors="coerce")

    meta = dlg[dlg["flip_final"] == 1][["dialogue_id", "flip_turn"]].rename(
        columns={"flip_turn": "flip_turn_meta"}
    )
    meta["flip_turn_meta"] = pd.to_numeric(meta["flip_turn_meta"], errors="coerce")

    merged = judge.merge(meta, on="dialogue_id", how="inner")
    at_flip = merged[merged["turn"] == merged["flip_turn_meta"]].copy()
    out = at_flip[
        [
            "dialogue_id",
            "mean_score",
            "cogency",
            "effectiveness",
            "reasonableness",
        ]
    ].rename(
        columns={
            "mean_score": "quality_at_flip",
            "cogency": "cogency_at_flip",
            "effectiveness": "effectiveness_at_flip",
            "reasonableness": "reasonableness_at_flip",
        }
    )
    return out.drop_duplicates(subset="dialogue_id", keep="first")


QUALITY_FLIP_COLS = (
    "quality_at_flip",
    "cogency_at_flip",
    "effectiveness_at_flip",
    "reasonableness_at_flip",
)


def fit_minmax_params(values: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce")
    return float(v.min()), float(v.max())


def apply_minmax_01(values: pd.Series, lo: float, hi: float) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=values.index, dtype=float)
    return ((v - lo) / (hi - lo)).clip(0.0, 1.0)


def fit_zscore_params(values: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce")
    mu = float(v.mean())
    sigma = float(v.std(ddof=0))
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = 1.0
    return mu, sigma


def apply_zscore(values: pd.Series, mu: float, sigma: float) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    return (v - mu) / sigma


def fit_argument_quality_A(train_quality: pd.DataFrame) -> dict:
    """Fit z-score + min-max params for A on train flipped dialogues."""
    flipped = train_quality.dropna(subset=list(QUALITY_FLIP_COLS)).copy()
    z_params = {col: fit_zscore_params(flipped[col]) for col in QUALITY_FLIP_COLS}
    z_mean = pd.Series(0.0, index=flipped.index)
    for col in QUALITY_FLIP_COLS:
        mu, sigma = z_params[col]
        z_mean = z_mean + apply_zscore(flipped[col], mu, sigma)
    z_mean /= len(QUALITY_FLIP_COLS)
    lo, hi = fit_minmax_params(z_mean)
    return {"z_params": z_params, "z_lo": lo, "z_hi": hi}


def apply_argument_quality_A(quality: pd.DataFrame, params: dict) -> pd.Series:
    """Return A in [0,1] for rows with flip-turn quality; NaN otherwise."""
    out = pd.Series(np.nan, index=quality.index, dtype=float)
    mask = quality[list(QUALITY_FLIP_COLS)].notna().all(axis=1)
    if not mask.any():
        return out
    sub = quality.loc[mask]
    z_mean = pd.Series(0.0, index=sub.index)
    for col in QUALITY_FLIP_COLS:
        mu, sigma = params["z_params"][col]
        z_mean = z_mean + apply_zscore(sub[col], mu, sigma)
    z_mean /= len(QUALITY_FLIP_COLS)
    out.loc[mask] = apply_minmax_01(z_mean, params["z_lo"], params["z_hi"])
    return out


def compute_persuasion_scores(
    df: pd.DataFrame,
    *,
    a: pd.Series | None = None,
    u_marker: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute U_pers, U_hybrid, and baselines from feature columns."""
    out = df.copy()
    f = out["F"].astype(float)
    s = out["S"].astype(float).fillna(0.0)
    c0 = out["C0"].astype(float)
    c_flip = out["C_flip"].astype(float)
    a_vals = out["A"].astype(float) if a is None else a.astype(float)
    a_vals = a_vals.fillna(0.5)

    out["U_pers"] = (
        W_F * f + W_S * f * s + W_A * f * (1.0 - a_vals) + W_C * f * (1.0 - c_flip)
    )
    out.loc[out["F"] == 0, "U_pers"] = 0.0
    out["U_hybrid"] = LAMBDA_HYBRID * (1.0 - c0) + (1.0 - LAMBDA_HYBRID) * out["U_pers"]
    out["U_self"] = 1.0 - c0
    out["U_flip"] = f
    if u_marker is not None:
        out["U_marker"] = u_marker.astype(float)
    return out


def _compute_A(quality_at_flip: pd.DataFrame) -> pd.DataFrame:
    """Average z-scores of four flip-turn quality dims, scaled to [0, 1]."""
    out = quality_at_flip.copy()
    params = fit_argument_quality_A(out)
    out["A"] = apply_argument_quality_A(out, params)
    z_cols: list[str] = []
    for raw_col in QUALITY_FLIP_COLS:
        z_col = f"z_{raw_col}"
        mu, sigma = fit_zscore_params(out[raw_col])
        out[z_col] = apply_zscore(out[raw_col], mu, sigma)
        z_cols.append(z_col)
    out["z_quality_mean"] = out[z_cols].mean(axis=1)
    return out[
        [
            "dialogue_id",
            "A",
            "z_quality_mean",
            "quality_at_flip",
            "cogency_at_flip",
            "effectiveness_at_flip",
            "reasonableness_at_flip",
        ]
    ]


def build_uncertainty_features(expl: pd.DataFrame, judge: pd.DataFrame) -> pd.DataFrame:
    dlg = build_dialogue_confidence(expl)

    if "original_index" in expl.columns:
        idx = (
            expl.sort_values(["dialogue_id", "turn"])
            .groupby("dialogue_id", as_index=False)["original_index"]
            .first()
        )
        dlg = dlg.merge(idx, on="dialogue_id", how="left")

    conf_flip = _confidence_at_flip(expl, dlg)
    quality_flip = _quality_at_flip(judge, dlg)
    quality_scored = _compute_A(quality_flip)

    out = dlg[
        [
            "dialogue_id",
            "gold_label",
            "answer_before",
            "correct_t0",
            "conf_before",
            "flip_final",
            "flip_turn",
        ]
    ].copy()
    if "original_index" in dlg.columns:
        out["original_index"] = dlg["original_index"]

    out["F"] = pd.to_numeric(out["flip_final"], errors="coerce").fillna(0).astype(int)
    out["T"] = pd.to_numeric(out["flip_turn"], errors="coerce")
    out["S"] = _speed(out["T"], out["F"])
    out["C0"] = pd.to_numeric(out["conf_before"], errors="coerce") / CONF_SCALE
    out["E_i"] = 1 - out["correct_t0"].astype(int)

    out = out.merge(conf_flip, on="dialogue_id", how="left")
    out = out.merge(quality_scored, on="dialogue_id", how="left", suffixes=("", "_dup"))
    dup_cols = [c for c in out.columns if c.endswith("_dup")]
    if dup_cols:
        out = out.drop(columns=dup_cols)

    return out.sort_values("dialogue_id").reset_index(drop=True)


def build_uncertainty_scores(features: pd.DataFrame, epistemic: pd.DataFrame | None = None) -> pd.DataFrame:
    out = features.copy()
    u_marker = None
    if epistemic is not None and "total_marker_n" in epistemic.columns:
        out = out.merge(
            epistemic[["dialogue_id", "total_marker_n"]],
            on="dialogue_id",
            how="left",
        )
        lo, hi = fit_minmax_params(out["total_marker_n"].fillna(0))
        u_marker = apply_minmax_01(out["total_marker_n"].fillna(0), lo, hi)
    else:
        out["total_marker_n"] = np.nan
    return compute_persuasion_scores(out, u_marker=u_marker)


def _print_summary(features: pd.DataFrame, scores: pd.DataFrame) -> None:
    print(f"Dialogues: {len(features)}")
    print(f"Flipped (F=1): {int(features['F'].sum())}")
    print(f"Initial errors (E_i=1): {int(features['E_i'].sum())}")
    print()
    for col in ("U_pers", "U_hybrid", "U_self", "U_token", "U_flip", "U_marker"):
        if col in scores.columns:
            s = scores[col]
            n = int(s.notna().sum())
            print(f"{col}: n={n}, mean={s.mean():.3f}, min={s.min():.3f}, max={s.max():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build persuasion uncertainty features and scores.")
    ap.add_argument("--expl", type=Path, default=DEFAULT_EXPL)
    ap.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--out-features", type=Path, default=DEFAULT_OUT_FEATURES)
    ap.add_argument("--out-scores", type=Path, default=DEFAULT_OUT_SCORES)
    args = ap.parse_args()

    expl = pd.read_csv(args.expl)
    judge = pd.read_csv(args.judge)

    features = build_uncertainty_features(expl, judge)

    expl_marked = add_epistemic_marker_columns(expl)
    epistemic = build_dialogue_epistemic(expl_marked)
    scores = build_uncertainty_scores(features, epistemic)
    tok = load_persuasion_tokenprob(original_indices=features["original_index"])
    scores = scores.merge(tok[["original_index", "confidence", "U_token"]], on="original_index", how="left")

    args.out_features.parent.mkdir(parents=True, exist_ok=True)
    args.out_scores.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "dialogue_id",
        "original_index",
        "gold_label",
        "answer_before",
        "correct_t0",
        "E_i",
        "conf_before",
        "conf_at_flip_raw",
        "F",
        "T",
        "S",
        "C0",
        "C_flip",
        "quality_at_flip",
        "cogency_at_flip",
        "effectiveness_at_flip",
        "reasonableness_at_flip",
        "A",
        "z_quality_mean",
    ]
    feature_cols = [c for c in feature_cols if c in features.columns]
    features[feature_cols].to_csv(args.out_features, index=False)

    score_cols = feature_cols + [
        "U_pers",
        "U_hybrid",
        "U_self",
        "confidence",
        "U_token",
        "U_flip",
        "total_marker_n",
        "U_marker",
    ]
    score_cols = [c for c in score_cols if c in scores.columns]
    scores[score_cols].to_csv(args.out_scores, index=False)

    plot_feature_overview(features, args.out_features.parent.parent / "png")
    plot_scores_overview(scores, args.out_scores.parent.parent / "png")

    print(f"Saved features: {args.out_features}")
    print(f"Saved feature plots: {args.out_features.parent.parent / 'png'}")
    print(f"Saved scores:   {args.out_scores}")
    print(f"Saved score plots: {args.out_scores.parent.parent / 'png'}")
    _print_summary(features, scores)


if __name__ == "__main__":
    main()
