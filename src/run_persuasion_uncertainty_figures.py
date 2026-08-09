"""
Build uncertainty scores and the 4 thesis figures for any model × dataset.

Figures (only these):
  metrics_comparison.png
  ablation_comparison.png
  bootstrap_brier_ci.png
  recalibration_brier_comparison.png

Output layout:
  output_wood/persuasion/{Model}/{dataset}/uncertainty_scores/
    csv/
    figures/
    summary.txt

Without arg-quality judge: A = 0.5 for flipped dialogues (neutral default).
Without token-prob join: U_token is omitted.

Example:
  cd src && python3 run_persuasion_uncertainty_figures.py --all
  cd src && python3 run_persuasion_uncertainty_figures.py --model GPT-4o --dataset fever
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from analyze_persuasion_epistemic_calibration import (
    add_epistemic_marker_columns,
    build_dialogue_epistemic,
)
from build_persuasion_uncertainty_scores import (
    QUALITY_FLIP_COLS,
    build_uncertainty_features,
    build_uncertainty_scores,
    compute_persuasion_scores,
)
from cross_validate_persuasion_uncertainty import N_FOLDS, grouped_kfold_indices
from evaluate_persuasion_uncertainty import (
    METHODS,
    N_BOOT,
    cluster_bootstrap_brier,
    compute_ablation_scores,
    evaluate_ablation,
    evaluate_methods,
)
from persuasion_tokenprob import load_persuasion_tokenprob
from persuasion_uncertainty_viz import (
    METHOD_LABELS,
    plot_ablation,
    plot_bootstrap_ci,
    plot_metrics_bar,
)
from recalibrate_persuasion_uncertainty import (
    CALIBRATORS,
    calibrate_oof,
    evaluate_raw_and_calibrated,
    plot_brier_comparison,
)

PERSUASION_ROOT = _PROJECT_ROOT / "output_wood" / "persuasion"
KEEP_FIGURES = (
    "metrics_comparison.png",
    "ablation_comparison.png",
    "bootstrap_brier_ci.png",
    "recalibration_brier_comparison.png",
)

DEEPSEEK_JUDGE = (
    PERSUASION_ROOT
    / "DeepSeek"
    / "fever"
    / "arg_quality"
    / "arg_quality_long_fever214_all_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv"
)


def find_judge_csv(model: str, dataset: str) -> Path | None:
    """Prefer all-turns judge, then flip-turns, then DeepSeek legacy path."""
    aq = PERSUASION_ROOT / model / dataset / "arg_quality"
    if model == "DeepSeek" and dataset == "fever" and DEEPSEEK_JUDGE.exists():
        # Prefer dedicated DeepSeek all-turns file if present
        if DEEPSEEK_JUDGE.exists():
            candidates_first = [DEEPSEEK_JUDGE]
        else:
            candidates_first = []
    else:
        candidates_first = []
    patterns = [
        f"arg_quality_long_{dataset}_all_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
        f"arg_quality_long_{dataset}_flip_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv",
        "arg_quality_long_*__top3__openrouter__openai_gpt-5.4-mini.csv",
    ]
    found: list[Path] = list(candidates_first)
    if aq.exists():
        for pat in patterns:
            found.extend(sorted(aq.glob(pat)))
    # unique preserve order
    seen: set[Path] = set()
    for p in found:
        rp = p.resolve()
        if rp in seen or not p.exists():
            continue
        seen.add(rp)
        return p
    return None

CONDITIONS = [
    ("DeepSeek", "fever"),
    ("DeepSeek", "popqa"),
    ("DeepSeek", "debateqa"),
    ("Qwen", "fever"),
    ("Qwen", "popqa"),
    ("Qwen", "debateqa"),
    ("Gemma", "fever"),
    ("Gemma", "popqa"),
    ("Gemma", "debateqa"),
    ("GPT-4o", "fever"),
    ("GPT-4o", "popqa"),
    ("GPT-4o", "debateqa"),
]


def find_expl(model: str, dataset: str) -> Path | None:
    candidates = [
        PERSUASION_ROOT / model / dataset / "csv" / "expl.csv",
        PERSUASION_ROOT / model / dataset / "rollout" / "csv" / "expl.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def out_dir(model: str, dataset: str) -> Path:
    return PERSUASION_ROOT / model / dataset / "uncertainty_scores"


def features_without_judge(expl: pd.DataFrame) -> pd.DataFrame:
    """Same features as build_uncertainty_features, with neutral A=0.5 on flips."""
    from persuasion_arg_quality import TOP_LEVEL_QUALITY_DIMENSIONS

    stub = pd.DataFrame(
        {
            "dialogue_id": [],
            "turn": [],
            **{c: [] for c in TOP_LEVEL_QUALITY_DIMENSIONS},
            "mean_score": [],
        }
    )
    features = build_uncertainty_features(expl, stub)
    flipped = features["F"] == 1
    features.loc[flipped, "A"] = 0.5
    features["A_source"] = np.where(flipped, "neutral_0.5", "na")
    return features


def attach_token_if_available(scores: pd.DataFrame, model: str, dataset: str) -> pd.DataFrame:
    if model != "DeepSeek" or dataset != "fever":
        return scores
    if "original_index" not in scores.columns:
        return scores
    try:
        tok = load_persuasion_tokenprob(original_indices=scores["original_index"])
    except Exception as e:
        print(f"  [warn] token-prob unavailable: {e}")
        return scores
    return scores.merge(
        tok[["original_index", "confidence", "U_token"]],
        on="original_index",
        how="left",
        suffixes=("", "_tok"),
    )


def active_methods(scores: pd.DataFrame) -> list[str]:
    out = []
    for m in METHODS:
        if m not in scores.columns:
            continue
        if scores[m].isna().all():
            continue
        out.append(m)
    return out


def assign_cv_folds(df: pd.DataFrame, n_folds: int = N_FOLDS) -> pd.DataFrame:
    out = df.copy()
    if "original_index" not in out.columns or out["original_index"].isna().all():
        out["original_index"] = np.arange(len(out))
    groups = out["original_index"].to_numpy()
    out["cv_fold"] = -1
    for fold_id, (_, test_idx) in enumerate(grouped_kfold_indices(groups, n_folds)):
        out.loc[out.index[test_idx], "cv_fold"] = fold_id
    return out


def run_condition(model: str, dataset: str, n_boot: int = N_BOOT) -> Path:
    expl_path = find_expl(model, dataset)
    if expl_path is None:
        raise FileNotFoundError(f"No expl.csv for {model}/{dataset}")

    dest = out_dir(model, dataset)
    csv_dir = dest / "csv"
    fig_dir = dest / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {model} / {dataset} ===")
    print(f"  expl: {expl_path}")

    expl = pd.read_csv(expl_path)
    judge_path = find_judge_csv(model, dataset)

    if judge_path is not None:
        judge = pd.read_csv(judge_path)
        features = build_uncertainty_features(expl, judge)
        features["A_source"] = np.where(features["F"] == 1, "judge", "na")
        print(f"  judge: {judge_path.name}")
    else:
        features = features_without_judge(expl)
        print("  judge: none → A=0.5 for flipped dialogues")

    # Ablation / U^pers need numeric A on flipped rows
    features["A"] = features["A"].fillna(0.5)

    expl_marked = add_epistemic_marker_columns(expl)
    epistemic = build_dialogue_epistemic(expl_marked)
    scores = build_uncertainty_scores(features, epistemic)
    scores = attach_token_if_available(scores, model, dataset)

    methods = active_methods(scores)
    print(f"  n={len(scores)} dialogues, methods={methods}")

    # ── Save tables ─────────────────────────────────────────────────────────
    feature_cols = [
        c
        for c in (
            "dialogue_id",
            "original_index",
            "gold_label",
            "answer_before",
            "correct_t0",
            "E_i",
            "conf_before",
            "F",
            "T",
            "S",
            "C0",
            "C_flip",
            "A",
            "A_source",
            "quality_at_flip",
            "cogency_at_flip",
            "effectiveness_at_flip",
            "reasonableness_at_flip",
        )
        if c in features.columns
    ]
    features[feature_cols].to_csv(csv_dir / "persuasion_uncertainty_features.csv", index=False)

    score_cols = [
        c
        for c in (
            *feature_cols,
            "U_pers",
            "U_hybrid",
            "U_self",
            "U_token",
            "U_flip",
            "U_marker",
            "total_marker_n",
            "confidence",
        )
        if c in scores.columns
    ]
    scores[score_cols].to_csv(csv_dir / "persuasion_uncertainty_scores.csv", index=False)

    # ── Evaluation figures ──────────────────────────────────────────────────
    metrics = evaluate_methods(scores, methods)
    metrics.to_csv(csv_dir / "uncertainty_metrics.csv", index=False)
    plot_metrics_bar(
        metrics,
        fig_dir / "metrics_comparison.png",
        f"{model} / {dataset} — method comparison (Brier primary)",
    )

    ablation_scores = compute_ablation_scores(scores)
    ablation_metrics = evaluate_ablation(ablation_scores)
    ablation_metrics.to_csv(csv_dir / "ablation_metrics.csv", index=False)
    plot_ablation(ablation_metrics, fig_dir / "ablation_comparison.png")

    boot_rows = []
    for baseline in ("U_self", "U_marker", "U_flip"):
        if baseline in scores.columns and "U_pers" in scores.columns:
            boot_rows.append(cluster_bootstrap_brier(scores, "U_pers", baseline, n_boot=n_boot))
    if "U_self" in scores.columns and "U_hybrid" in scores.columns:
        boot_rows.append(cluster_bootstrap_brier(scores, "U_hybrid", "U_self", n_boot=n_boot))
    bootstrap = pd.DataFrame(boot_rows)
    bootstrap.to_csv(csv_dir / "bootstrap_brier_ci.csv", index=False)
    if not bootstrap.empty:
        plot_bootstrap_ci(bootstrap, fig_dir / "bootstrap_brier_ci.png")

    # ── Recalibration (isotonic + Platt on fold splits) ─────────────────────
    folded = assign_cv_folds(scores)
    cal_df = folded[["dialogue_id", "original_index", "E_i", "cv_fold", *methods]].copy()
    for method in methods:
        for cal in CALIBRATORS:
            cal_df[f"{method}__{cal}"] = calibrate_oof(folded, method, cal, n_folds=N_FOLDS)

    cal_metrics = evaluate_raw_and_calibrated(cal_df, methods)
    cal_metrics.to_csv(csv_dir / "recalibration_metrics.csv", index=False)
    cal_df.to_csv(csv_dir / "oof_calibrated_scores.csv", index=False)
    plot_brier_comparison(cal_metrics, fig_dir / "recalibration_brier_comparison.png")

    # ── Summary ─────────────────────────────────────────────────────────────
    e_rate = float(scores["E_i"].mean()) if "E_i" in scores.columns else float("nan")
    lines = [
        f"Persuasion uncertainty — {model} / {dataset}",
        f"n = {len(scores)} dialogues, initial error rate = {e_rate:.3f}",
        f"A source: {'arg-quality judge' if judge_path else 'neutral A=0.5 (no judge)'}",
        f"Methods: {', '.join(methods)}",
        "",
        "Primary metric (Brier, lower is better):",
    ]
    for _, r in metrics.sort_values("brier").iterrows():
        lines.append(
            f"  {r['label']:30s}  Brier={r['brier']:.4f}  AUROC={r['auroc']:.3f}  UCE={r['uce']:.4f}"
        )
    lines += ["", "Ablation (Brier):"]
    for _, r in ablation_metrics.iterrows():
        lines.append(f"  {r['variant']:35s}  Brier={r['brier']:.4f}  AUROC={r['auroc']:.3f}")

    iso = cal_metrics[cal_metrics["calibrator"] == "isotonic"].sort_values("brier_cal")
    lines += ["", "Isotonic recalibration (OOF Brier):"]
    for _, r in iso.iterrows():
        lines.append(
            f"  {r['label']:30s}  raw={r['brier_raw']:.4f}  cal={r['brier_cal']:.4f}  Δ={r['delta_brier']:+.4f}"
        )

    summary_path = dest / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  figures → {fig_dir}")
    print(f"  summary → {summary_path}")
    print("\n".join(lines[4:12]))
    return dest


def clean_deepseek_composite_pngs() -> None:
    """Keep only the 4 thesis figures under DeepSeek/fever/composite_score/figures/."""
    root = PERSUASION_ROOT / "DeepSeek" / "fever" / "composite_score"
    if not root.exists():
        return
    keep_names = set(KEEP_FIGURES)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Collect keepers from anywhere under composite_score, then delete all PNGs there
    found: dict[str, Path] = {}
    for png in root.rglob("*.png"):
        if png.name in keep_names and png.name not in found:
            found[png.name] = png

    for name, src in found.items():
        if src.parent.resolve() != fig_dir.resolve():
            shutil.copy2(src, fig_dir / name)

    deleted = 0
    for png in list(root.rglob("*.png")):
        if png.parent.resolve() == fig_dir.resolve() and png.name in keep_names:
            continue
        png.unlink(missing_ok=True)
        deleted += 1

    print(f"Cleaned DeepSeek composite_score: kept {len(list(fig_dir.glob('*.png')))} in figures/, deleted {deleted} extras.")


def write_index(results: list[tuple[str, str, Path]]) -> None:
    rows = []
    for model, dataset, path in results:
        summary = path / "summary.txt"
        metrics = path / "csv" / "uncertainty_metrics.csv"
        n = ""
        best = ""
        if metrics.exists():
            m = pd.read_csv(metrics).sort_values("brier")
            n = str(int(m["n"].iloc[0])) if "n" in m.columns else ""
            best = f"{m.iloc[0]['label']} (Brier={m.iloc[0]['brier']:.4f})"
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "n": n,
                "best_by_brier": best,
                "path": str(path.relative_to(_PROJECT_ROOT)),
            }
        )
    out = PERSUASION_ROOT / "uncertainty_scores_index.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nIndex: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 4 thesis figures per model×dataset.")
    ap.add_argument("--model", default=None, help="DeepSeek|Qwen|Gemma|GPT-4o")
    ap.add_argument("--dataset", default=None, help="fever|popqa|debateqa")
    ap.add_argument("--all", action="store_true", help="Run all available conditions")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean DeepSeek composite_score PNGs",
    )
    args = ap.parse_args()

    clean_deepseek_composite_pngs()
    if args.clean_only:
        return

    if args.all or (args.model is None and args.dataset is None):
        todo = [(m, d) for m, d in CONDITIONS if find_expl(m, d) is not None]
    else:
        if not args.model or not args.dataset:
            raise SystemExit("Pass --model and --dataset, or --all")
        todo = [(args.model, args.dataset)]

    results: list[tuple[str, str, Path]] = []
    for model, dataset in todo:
        try:
            path = run_condition(model, dataset, n_boot=args.n_boot)
            results.append((model, dataset, path))
        except Exception as e:
            print(f"FAILED {model}/{dataset}: {e}")
            raise

    write_index(results)
    print(f"\nDone: {len(results)} condition(s).")


if __name__ == "__main__":
    main()
