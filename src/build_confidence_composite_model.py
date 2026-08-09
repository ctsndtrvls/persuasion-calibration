"""
Build a confidence-driven persuasion composite score.

This script aggregates dialogue-level factors requested for FEVER persuasion analysis:
- argument quality (cogency/effectiveness/reasonableness)
- argument quality × verdict flip interaction
- confidence dynamics (before / turn1 / final, plus trajectory slope)
- epistemic uncertainty markers in decision_explanation
- flip rate / flip flag
- dialogue length (turns)
- FEVER verdict groups (SUPPORTS / REFUTES / NEI)
- "refuted + correct at t0" interaction

Outputs:
  - csv/confidence_composite_features.csv
  - csv/confidence_model_coefficients.csv
  - csv/accuracy_model_coefficients.csv
  - csv/confidence_composite_scores.csv
  - csv/confidence_by_turn.csv
  - csv/turns_by_initial_verdict.csv
  - csv/flip_by_verdict_accuracy.csv
  - txt/confidence_model_formula.txt

Example:
  cd src && python3 build_confidence_composite_model.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analyze_arg_quality_flip import build_dialogue_quality, dialogue_flip_meta
from analyze_persuasion_confidence import (
    build_dialogue_confidence,
    first_turn_snapshot,
    summarize_verdict_accuracy,
)
from analyze_persuasion_epistemic_calibration import (
    add_epistemic_marker_columns,
    build_dialogue_epistemic,
)
from persuasion_arg_quality import TOP_LEVEL_QUALITY_DIMENSIONS

FEVER_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
COMPOSITE_DIR = FEVER_DIR / "composite_score"
DEFAULT_EXPL = FEVER_DIR / "csv" / "expl.csv"
DEFAULT_JUDGE = (
    FEVER_DIR
    / "arg_quality"
    / "arg_quality_long_fever214_all_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv"
)
MAX_TURN = 15
VERDICT_ORDER = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def _normalize_label(x: object) -> str:
    s = str(x or "").strip().upper()
    if "NOT ENOUGH" in s or s == "NEI":
        return "NOT ENOUGH INFO"
    if "REFUTE" in s:
        return "REFUTES"
    if "SUPPORT" in s:
        return "SUPPORTS"
    return s


def _trajectory_slope_per_dialogue(expl: pd.DataFrame) -> pd.DataFrame:
    snap = first_turn_snapshot(expl)
    snap["confidence"] = pd.to_numeric(snap["confidence"], errors="coerce")
    rows: list[dict[str, float | str]] = []
    for did, g in snap.groupby("dialogue_id", sort=False):
        gg = g.dropna(subset=["turn", "confidence"]).sort_values("turn")
        if len(gg) < 2:
            slope = 0.0
        else:
            slope = float(np.polyfit(gg["turn"].to_numpy(dtype=float), gg["confidence"].to_numpy(dtype=float), 1)[0])
        rows.append({"dialogue_id": did, "confidence_slope_per_turn": slope})
    return pd.DataFrame(rows)


def _mean_confidence_by_turn(expl: pd.DataFrame) -> pd.DataFrame:
    snap = first_turn_snapshot(expl)
    snap["confidence"] = pd.to_numeric(snap["confidence"], errors="coerce")
    out = (
        snap.groupby("turn", as_index=False)["confidence"]
        .agg(mean_confidence="mean", n_dialogues="count")
        .sort_values("turn")
    )
    return out[out["turn"] <= MAX_TURN].copy()


def _turns_by_initial_verdict(dlg_conf: pd.DataFrame) -> pd.DataFrame:
    out = (
        dlg_conf.groupby("answer_before", as_index=False)["final_turn"]
        .agg(n_dialogues="count", mean_turns="mean", median_turns="median")
        .rename(columns={"answer_before": "initial_verdict"})
    )
    out["initial_verdict"] = pd.Categorical(out["initial_verdict"], categories=VERDICT_ORDER, ordered=True)
    out = out.sort_values("initial_verdict")
    out["mean_turns_norm"] = out["mean_turns"] / MAX_TURN
    out["median_turns_norm"] = out["median_turns"] / MAX_TURN
    return out


def _build_feature_table(expl: pd.DataFrame, judge: pd.DataFrame) -> pd.DataFrame:
    dlg_conf = build_dialogue_confidence(expl)
    flip_meta = dialogue_flip_meta(expl)
    dlg_quality = build_dialogue_quality(judge, flip_meta)
    epi_rows = add_epistemic_marker_columns(expl)
    dlg_epi = build_dialogue_epistemic(epi_rows)
    slope = _trajectory_slope_per_dialogue(expl)

    merged = dlg_conf.merge(dlg_quality, on="dialogue_id", how="left", suffixes=("_conf", "_quality"))
    merged = merged.merge(dlg_epi, on="dialogue_id", how="left")
    merged = merged.merge(slope, on="dialogue_id", how="left")

    # quality × flip interaction
    merged["mean_overall"] = pd.to_numeric(merged["mean_overall"], errors="coerce")
    if "flip_final_conf" in merged.columns:
        merged["flip_final"] = pd.to_numeric(merged["flip_final_conf"], errors="coerce")
    elif "flip_final_quality" in merged.columns:
        merged["flip_final"] = pd.to_numeric(merged["flip_final_quality"], errors="coerce")
    elif "flip_final" in merged.columns:
        merged["flip_final"] = pd.to_numeric(merged["flip_final"], errors="coerce")
    else:
        merged["flip_final"] = 0
    merged["flip_final"] = merged["flip_final"].fillna(0).astype(int)
    if "flip_turn_conf" in merged.columns:
        merged["flip_turn"] = pd.to_numeric(merged["flip_turn_conf"], errors="coerce")
    elif "flip_turn_quality" in merged.columns:
        merged["flip_turn"] = pd.to_numeric(merged["flip_turn_quality"], errors="coerce")
    elif "flip_turn" in merged.columns:
        merged["flip_turn"] = pd.to_numeric(merged["flip_turn"], errors="coerce")
    else:
        merged["flip_turn"] = np.nan
    merged["quality_x_flip"] = merged["mean_overall"] * merged["flip_final"]

    # verdict dummies + required interaction: refuted + correct
    merged["answer_before"] = merged["answer_before"].map(_normalize_label)
    merged["is_supports"] = (merged["answer_before"] == "SUPPORTS").astype(int)
    merged["is_refutes"] = (merged["answer_before"] == "REFUTES").astype(int)
    merged["is_nei"] = (merged["answer_before"] == "NOT ENOUGH INFO").astype(int)
    merged["refutes_x_correct_t0"] = merged["is_refutes"] * merged["correct_t0"].astype(int)

    # normalized helper factors
    merged["turns_norm"] = pd.to_numeric(merged["final_turn"], errors="coerce") / MAX_TURN
    merged["quality_norm"] = merged["mean_overall"] / 3.0
    merged["marker_density"] = (
        pd.to_numeric(merged["total_marker_n"], errors="coerce")
        / pd.to_numeric(merged["n_log_rows"], errors="coerce").replace(0, np.nan)
    )

    # accuracy final from final answer vs gold label
    merged["gold_label"] = merged["gold_label"].map(_normalize_label)
    merged["answer_final"] = merged["answer_final"].map(_normalize_label)
    merged["correct_final"] = (merged["answer_final"] == merged["gold_label"]).astype(int)
    return merged


def _fit_models(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "mean_cogency",
        "mean_effectiveness",
        "mean_reasonableness",
        "quality_x_flip",
        "conf_before",
        "conf_delta_turn1",
        "confidence_slope_per_turn",
        "total_marker_n",
        "flip_final",
        "final_turn",
        "is_refutes",
        "is_nei",
        "refutes_x_correct_t0",
    ]
    work = features.dropna(subset=feature_cols + ["conf_final", "correct_final"]).copy()

    x_raw = work[feature_cols].to_numpy(dtype=float)
    mu = np.nanmean(x_raw, axis=0)
    sigma = np.nanstd(x_raw, axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    xs = (x_raw - mu) / sigma

    # Model 1: confidence regression (target requested by supervisor)
    y_conf = work["conf_final"].to_numpy(dtype=float)
    x_aug = np.c_[np.ones(len(xs)), xs]
    beta_conf, *_ = np.linalg.lstsq(x_aug, y_conf, rcond=None)
    intercept_conf = float(beta_conf[0])
    coef_conf = beta_conf[1:]
    conf_pred = x_aug @ beta_conf
    work["conf_pred"] = conf_pred
    ss_res = float(np.sum((y_conf - conf_pred) ** 2))
    ss_tot = float(np.sum((y_conf - y_conf.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    conf_coef = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef_std": coef_conf.astype(float),
            "abs_coef_std": np.abs(coef_conf.astype(float)),
        }
    )
    conf_coef["weight_absnorm"] = conf_coef["abs_coef_std"] / conf_coef["abs_coef_std"].sum()
    conf_coef["intercept"] = intercept_conf
    conf_coef["r2"] = r2
    conf_coef["n"] = len(work)

    # Model 2: accuracy proxy (linear probability on standardized features)
    y_acc = work["correct_final"].to_numpy(dtype=int)
    beta_acc, *_ = np.linalg.lstsq(x_aug, y_acc, rcond=None)
    intercept_acc = float(beta_acc[0])
    coef_acc = beta_acc[1:]
    acc_pred = x_aug @ beta_acc
    acc_proba = np.clip(acc_pred, 0.0, 1.0)
    work["acc_pred_proba"] = acc_proba
    acc_hat_label = (acc_proba >= 0.5).astype(int)
    acc_fit = float((acc_hat_label == y_acc).mean())

    acc_coef = pd.DataFrame(
        {
            "feature": feature_cols,
            "coef_std": coef_acc.astype(float),
        }
    )
    acc_coef["intercept"] = intercept_acc
    acc_coef["accuracy"] = acc_fit
    acc_coef["n"] = len(work)

    # Composite: explanatory confidence + predicted accuracy + stability + quality + epistemic consistency
    conf_component = np.clip(work["conf_pred"] / 10.0, 0.0, 1.0)
    acc_component = np.clip(work["acc_pred_proba"], 0.0, 1.0)
    stability_component = 1.0 - work["flip_final"].astype(float)
    quality_component = np.clip(work["quality_norm"], 0.0, 1.0)
    epistemic_component = 1.0 - np.clip(work["marker_density"].fillna(0.0), 0.0, 1.0)
    turn_efficiency = 1.0 - np.clip(work["turns_norm"], 0.0, 1.0)

    work["confidence_composite_score"] = (
        0.30 * conf_component
        + 0.20 * acc_component
        + 0.15 * quality_component
        + 0.15 * stability_component
        + 0.10 * epistemic_component
        + 0.10 * turn_efficiency
    )

    # Model 3: change in confidence (external persuasion factors only; no conf_slope)
    delta_feature_cols = [
        "mean_cogency",
        "mean_effectiveness",
        "mean_reasonableness",
        "quality_x_flip",
        "flip_final",
        "final_turn",
        "total_marker_n",
        "is_refutes",
        "is_nei",
        "refutes_x_correct_t0",
    ]
    delta_work = work.dropna(subset=delta_feature_cols + ["conf_delta_final"]).copy()
    x_delta_raw = delta_work[delta_feature_cols].to_numpy(dtype=float)
    mu_d = np.nanmean(x_delta_raw, axis=0)
    sigma_d = np.nanstd(x_delta_raw, axis=0)
    sigma_d = np.where(sigma_d == 0, 1.0, sigma_d)
    xs_d = (x_delta_raw - mu_d) / sigma_d
    y_delta = delta_work["conf_delta_final"].to_numpy(dtype=float)
    x_aug_d = np.c_[np.ones(len(xs_d)), xs_d]
    beta_delta, *_ = np.linalg.lstsq(x_aug_d, y_delta, rcond=None)
    delta_pred = x_aug_d @ beta_delta
    ss_res_d = float(np.sum((y_delta - delta_pred) ** 2))
    ss_tot_d = float(np.sum((y_delta - y_delta.mean()) ** 2))
    r2_delta = 1.0 - ss_res_d / ss_tot_d if ss_tot_d > 0 else float("nan")

    delta_coef = pd.DataFrame(
        {
            "feature": delta_feature_cols,
            "coef_std": beta_delta[1:].astype(float),
            "abs_coef_std": np.abs(beta_delta[1:].astype(float)),
        }
    )
    delta_coef["weight_absnorm"] = delta_coef["abs_coef_std"] / delta_coef["abs_coef_std"].sum()
    delta_coef["intercept"] = float(beta_delta[0])
    delta_coef["r2"] = r2_delta
    delta_coef["n"] = len(delta_work)

    return work, conf_coef, acc_coef, delta_coef


def _formula_text(conf_coef: pd.DataFrame) -> str:
    intercept = float(conf_coef["intercept"].iloc[0])
    r2 = float(conf_coef["r2"].iloc[0])
    terms = []
    for _, r in conf_coef.sort_values("abs_coef_std", ascending=False).iterrows():
        s = "+" if r["coef_std"] >= 0 else "-"
        terms.append(f" {s} {abs(float(r['coef_std'])):.3f}*z({r['feature']})")
    return (
        "Confidence model (standardized features):\n"
        f"conf_final_hat = {intercept:.3f}" + "".join(terms) + "\n"
        f"R^2 = {r2:.3f}\n\n"
        "Composite score:\n"
        "S = 0.30*clip(conf_pred/10) + 0.20*acc_pred_proba + 0.15*quality_norm + "
        "0.15*(1-flip_final) + 0.10*(1-marker_density) + 0.10*(1-turns_norm)\n"
    )


def _plot_visual_pack(
    scored: pd.DataFrame,
    conf_coef: pd.DataFrame,
    conf_by_turn: pd.DataFrame,
    turns_by_verdict: pd.DataFrame,
    flip_by_verdict_accuracy: pd.DataFrame,
    out_png: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (1) Confidence model factor importance (|standardized beta|)
    ax = axes[0, 0]
    top = conf_coef.sort_values("abs_coef_std", ascending=False).head(8).copy()
    sns.barplot(data=top, y="feature", x="abs_coef_std", ax=ax, color="#4C78A8")
    ax.set_title("Confidence drivers (|standardized coefficient|)")
    ax.set_xlabel("|beta|")
    ax.set_ylabel("")

    # (2) Mean confidence by turn
    ax = axes[0, 1]
    sns.lineplot(data=conf_by_turn, x="turn", y="mean_confidence", marker="o", ax=ax, color="#E45756")
    ax.set_xlim(-0.2, MAX_TURN + 0.2)
    ax.set_xticks(range(0, MAX_TURN + 1))
    ax.set_title("Confidence by turn")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean confidence")

    # (3) Mean turns by initial verdict
    ax = axes[0, 2]
    tbv = turns_by_verdict.copy()
    tbv["initial_verdict"] = tbv["initial_verdict"].astype(str).replace({"NOT ENOUGH INFO": "NEI"})
    sns.barplot(data=tbv, x="initial_verdict", y="mean_turns", ax=ax, palette=["#4C78A8", "#E45756", "#F58518"])
    ax.set_title("How many turns by initial verdict")
    ax.set_xlabel("")
    ax.set_ylabel("Mean dialogue length")

    # (4) Flip rate by verdict × accuracy at t0
    ax = axes[1, 0]
    fva = flip_by_verdict_accuracy[flip_by_verdict_accuracy["initial_verdict"] != "ALL"].copy()
    fva["initial_verdict_short"] = fva["initial_verdict"].replace({"NOT ENOUGH INFO": "NEI"})
    sns.barplot(
        data=fva,
        x="initial_verdict_short",
        y="flip_rate_pct",
        hue="accuracy_t0",
        ax=ax,
        palette={"Correct at t0": "#4C78A8", "Incorrect at t0": "#BAB0AC"},
    )
    ax.set_title("Flip rate by verdict × initial correctness")
    ax.set_xlabel("")
    ax.set_ylabel("Flip rate (%)")
    ax.legend(title="")

    # (5) Composite score distribution
    ax = axes[1, 1]
    sns.histplot(scored["confidence_composite_score"], bins=16, ax=ax, color="#54A24B")
    ax.set_title("Composite score distribution")
    ax.set_xlabel("confidence_composite_score")
    ax.set_ylabel("Dialogues")

    # (6) Composite score by verdict
    ax = axes[1, 2]
    sv = scored.copy()
    sv["answer_before_short"] = sv["answer_before"].replace({"NOT ENOUGH INFO": "NEI"})
    sns.boxplot(
        data=sv,
        x="answer_before_short",
        y="confidence_composite_score",
        ax=ax,
        palette=["#4C78A8", "#E45756", "#F58518"],
    )
    ax.set_title("Composite score by initial verdict")
    ax.set_xlabel("")
    ax.set_ylabel("Score")

    fig.suptitle("Confidence-driven composite score: visual summary", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build confidence-driven persuasion composite score.")
    ap.add_argument("--expl", type=Path, default=DEFAULT_EXPL)
    ap.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--out-dir", type=Path, default=COMPOSITE_DIR)
    args = ap.parse_args()

    expl = pd.read_csv(args.expl)
    judge = pd.read_csv(args.judge)
    features = _build_feature_table(expl, judge)
    scored, conf_coef, acc_coef, delta_coef = _fit_models(features)

    out_csv = args.out_dir / "csv"
    out_txt = args.out_dir / "txt"
    out_png = args.out_dir / "png"
    out_csv.mkdir(parents=True, exist_ok=True)
    out_txt.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)

    feature_export_cols = [
        "dialogue_id",
        "gold_label",
        "answer_before",
        "correct_t0",
        "correct_final",
        "conf_before",
        "conf_after_turn1",
        "conf_final",
        "conf_delta_turn1",
        "conf_delta_final",
        "confidence_slope_per_turn",
        "flip_final",
        "flip_turn",
        "final_turn",
        "mean_cogency",
        "mean_effectiveness",
        "mean_reasonableness",
        "mean_overall",
        "quality_x_flip",
        "total_marker_n",
        "marker_density",
        "refutes_x_correct_t0",
        "confidence_composite_score",
    ]
    scored[feature_export_cols].to_csv(out_csv / "confidence_composite_features.csv", index=False)
    conf_coef.sort_values("abs_coef_std", ascending=False).to_csv(
        out_csv / "confidence_model_coefficients.csv", index=False
    )
    acc_coef.reindex(acc_coef["coef_std"].abs().sort_values(ascending=False).index).to_csv(
        out_csv / "accuracy_model_coefficients.csv", index=False
    )
    delta_coef.sort_values("abs_coef_std", ascending=False).to_csv(
        out_csv / "delta_confidence_model_coefficients.csv", index=False
    )
    scored[
        [
            "dialogue_id",
            "conf_final",
            "conf_pred",
            "correct_final",
            "acc_pred_proba",
            "confidence_composite_score",
        ]
    ].sort_values("confidence_composite_score", ascending=False).to_csv(
        out_csv / "confidence_composite_scores.csv", index=False
    )

    conf_by_turn = _mean_confidence_by_turn(expl)
    turns_by_verdict = _turns_by_initial_verdict(build_dialogue_confidence(expl))
    flip_by_verdict_accuracy = summarize_verdict_accuracy(build_dialogue_confidence(expl))
    conf_by_turn.to_csv(out_csv / "confidence_by_turn.csv", index=False)
    turns_by_verdict.to_csv(out_csv / "turns_by_initial_verdict.csv", index=False)
    flip_by_verdict_accuracy.to_csv(out_csv / "flip_by_verdict_accuracy.csv", index=False)
    (out_txt / "confidence_model_formula.txt").write_text(_formula_text(conf_coef), encoding="utf-8")
    _plot_visual_pack(
        scored,
        conf_coef,
        conf_by_turn,
        turns_by_verdict,
        flip_by_verdict_accuracy,
        out_png / "confidence_composite_visual_pack.png",
    )

    print(f"Saved features: {out_csv / 'confidence_composite_features.csv'}")
    print(f"Saved confidence model coefficients: {out_csv / 'confidence_model_coefficients.csv'}")
    print(f"Saved accuracy model coefficients: {out_csv / 'accuracy_model_coefficients.csv'}")
    print(f"Saved composite scores: {out_csv / 'confidence_composite_scores.csv'}")
    print(f"Saved confidence by turn: {out_csv / 'confidence_by_turn.csv'}")
    print(f"Saved turns by verdict: {out_csv / 'turns_by_initial_verdict.csv'}")
    print(f"Saved flip by verdict+accuracy: {out_csv / 'flip_by_verdict_accuracy.csv'}")
    print(f"Saved formula text: {out_txt / 'confidence_model_formula.txt'}")
    print(f"Saved figure: {out_png / 'confidence_composite_visual_pack.png'}")


if __name__ == "__main__":
    main()
