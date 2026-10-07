"""
Learned persuasion logistic regression vs the manual U^pers score.

Supervisor question:
  Can the persuasion features predict whether the initial answer was wrong,
  and does learning their contribution improve over the manually designed score?

Model (hybrid score is not used):

  P(E_i = 1) = sigmoid(
        b0
      + w1 * F
      + w2 * F * S
      + w3 * F * (1 - A)
      + w4 * F * (1 - C_flip)
  )

Local: one regression per dataset. The four target models are pooled.
       5-fold CV is grouped by original_index, so the same item stays
       inside one fold across models.
Global: one regression on all three datasets.
        Folds are assigned inside each dataset, then combined, so every
        fold still contains FEVER, PopQA, and DebateQA.

Metrics for the learned model are out-of-fold. The manual score is the
fixed formula already stored as U_pers. An intercept-only model is the
base-rate reference for "can these features predict at all".

Example:
  cd src && python3 learn_persuasion_logistic.py
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_persuasion_uncertainty import (
    auprc_score,
    auroc_score,
    brier_score,
    log_loss,
    uce_score,
)
from incremental_persuasion_regression import fit_logistic, predict_proba
from persuasion_uncertainty_viz import BG

PERSUASION_ROOT = _PROJECT_ROOT / "output_wood" / "persuasion"
OUT_DIR = PERSUASION_ROOT / "learned_logistic"

MODELS = ("DeepSeek", "GPT-4o", "Gemma", "Qwen")
DATASETS = ("fever", "popqa", "debateqa")
N_FOLDS = 5
N_BOOT = 2000
SEED = 42

TERMS = (
    "intercept",
    "flip",
    "flip_x_speed",
    "flip_x_weak_argument",
    "flip_x_uncertainty_at_flip",
)
FEATURE_TERMS = TERMS[1:]


def load_dialogues() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        for dataset in DATASETS:
            path = (
                PERSUASION_ROOT
                / model
                / dataset
                / "uncertainty_scores"
                / "csv"
                / "persuasion_uncertainty_scores.csv"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
            df["model"] = model
            df["dataset"] = dataset
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["E_i"] = pd.to_numeric(out["E_i"], errors="coerce").astype(int)
    out["F"] = pd.to_numeric(out["F"], errors="coerce").fillna(0.0)
    out["S"] = pd.to_numeric(out["S"], errors="coerce").fillna(0.0)
    out["A"] = pd.to_numeric(out["A"], errors="coerce").fillna(0.5)
    out["C_flip"] = pd.to_numeric(out["C_flip"], errors="coerce").fillna(0.5)
    out["U_pers"] = pd.to_numeric(out["U_pers"], errors="coerce")
    out["original_index"] = pd.to_numeric(out["original_index"], errors="coerce").astype(int)
    out["group_id"] = out["dataset"].astype(str) + ":" + out["original_index"].astype(str)
    return out


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Same four persuasion terms as U^pers, plus an intercept."""
    f = df["F"].to_numpy(dtype=float)
    speed = df["S"].to_numpy(dtype=float)
    weak_argument = 1.0 - df["A"].to_numpy(dtype=float)
    uncertainty_at_flip = 1.0 - df["C_flip"].to_numpy(dtype=float)
    return np.column_stack(
        [
            np.ones(len(df)),
            f,
            f * speed,
            f * weak_argument,
            f * uncertainty_at_flip,
        ]
    )


def assign_folds(df: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """Group items by original_index inside each dataset, then split."""
    fold_of_group: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    for dataset, sub in df.groupby("dataset", sort=False):
        groups = sub["group_id"].drop_duplicates().to_numpy()
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        for fold_id, chunk in enumerate(np.array_split(shuffled, n_folds)):
            for group in chunk:
                fold_of_group[str(group)] = fold_id
    return df["group_id"].map(fold_of_group).to_numpy(dtype=int)


def fit_predict_fold(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return np.full(len(X_test), float(y_train.mean()))
    beta = fit_logistic(X_train, y_train)
    return predict_proba(X_test, beta)


def oof_logistic(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return learned OOF probabilities, intercept-only OOF probabilities, fold ids, and fold coefficients.

    The intercept-only model is fit on the same folds, with every persuasion term removed.
    """
    X = design_matrix(df)
    y = df["E_i"].to_numpy(dtype=int)
    folds = assign_folds(df, N_FOLDS, seed)
    oof = np.full(len(df), np.nan)
    oof_intercept = np.full(len(df), np.nan)
    coef_rows: list[dict] = []
    intercept_only = np.ones((len(df), 1))
    for fold_id in range(N_FOLDS):
        test = folds == fold_id
        train = ~test
        if not test.any() or not train.any():
            continue
        oof[test] = fit_predict_fold(X[train], y[train], X[test])
        oof_intercept[test] = fit_predict_fold(intercept_only[train], y[train], intercept_only[test])
        if len(np.unique(y[train])) < 2:
            beta = np.full(X.shape[1], np.nan)
            beta[0] = np.log(
                np.clip(y[train].mean(), 1e-6, 1 - 1e-6)
                / np.clip(1.0 - y[train].mean(), 1e-6, 1 - 1e-6)
            )
        else:
            beta = fit_logistic(X[train], y[train])
        for term, coef in zip(TERMS, beta):
            coef_rows.append({"fold": fold_id, "term": term, "coef": float(coef)})
    if np.isnan(oof).any() or np.isnan(oof_intercept).any():
        raise RuntimeError("Out-of-fold predictions are incomplete.")
    return oof, oof_intercept, folds, pd.DataFrame(coef_rows)


def full_sample_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    X = design_matrix(df)
    y = df["E_i"].to_numpy(dtype=int)
    beta = fit_logistic(X, y)
    return pd.DataFrame({"term": TERMS, "coef": beta})


def metric_row(method: str, u: np.ndarray, e: np.ndarray) -> dict:
    return {
        "method": method,
        "n": int(len(e)),
        "error_rate": float(np.mean(e)),
        "brier": brier_score(u, e),
        "auroc": auroc_score(u, e),
        "auprc": auprc_score(u, e),
        "log_loss": log_loss(u, e),
        "uce": uce_score(u, e),
    }


def paired_delta_brier(
    y: np.ndarray,
    learned: np.ndarray,
    manual: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Cluster bootstrap of Brier(learned) - Brier(manual). Negative means learned is better."""
    frame = pd.DataFrame(
        {
            "group": groups,
            "sse_learned": (learned - y) ** 2,
            "sse_manual": (manual - y) ** 2,
        }
    )
    per_group = frame.groupby("group", sort=False).agg(
        sse_learned=("sse_learned", "sum"),
        sse_manual=("sse_manual", "sum"),
        n=("sse_learned", "size"),
    )
    sse_l = per_group["sse_learned"].to_numpy()
    sse_m = per_group["sse_manual"].to_numpy()
    counts = per_group["n"].to_numpy()
    point = float(sse_l.sum() / counts.sum() - sse_m.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    n_groups = len(per_group)
    for i in range(n_boot):
        take = rng.integers(0, n_groups, n_groups)
        n = counts[take].sum()
        draws[i] = sse_l[take].sum() / n - sse_m[take].sum() / n
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_brier": point,
        "delta_brier_ci_low": float(lo),
        "delta_brier_ci_high": float(hi),
        "n_groups": int(n_groups),
    }


def evaluate_split(
    scope: str,
    df: pd.DataFrame,
    learned: np.ndarray,
    intercept_only: np.ndarray,
    seed: int,
    *,
    primary_slice: str,
    also_by_dataset: bool,
) -> list[dict]:
    rows: list[dict] = []
    slices: list[tuple[str, np.ndarray]] = [(primary_slice, np.ones(len(df), dtype=bool))]
    if also_by_dataset:
        for dataset in DATASETS:
            slices.append((dataset, (df["dataset"] == dataset).to_numpy()))
    y_all = df["E_i"].to_numpy(dtype=int)
    manual_all = df["U_pers"].to_numpy(dtype=float)
    groups_all = df["group_id"].to_numpy()
    for slice_name, mask in slices:
        y = y_all[mask]
        manual = manual_all[mask]
        learned_slice = learned[mask]
        intercept_slice = intercept_only[mask]
        groups = groups_all[mask]
        for method, pred in (
            ("learned", learned_slice),
            ("manual", manual),
            ("intercept_only", intercept_slice),
        ):
            row = metric_row(method, pred, y)
            row["scope"] = scope
            row["slice"] = slice_name
            rows.append(row)
        delta = paired_delta_brier(y, learned_slice, manual, groups, N_BOOT, seed)
        rows.append(
            {
                "scope": scope,
                "slice": slice_name,
                "method": "learned_minus_manual",
                "n": int(mask.sum()),
                "error_rate": float(y.mean()),
                "brier": delta["delta_brier"],
                "auroc": np.nan,
                "auprc": np.nan,
                "log_loss": np.nan,
                "uce": np.nan,
                "delta_brier_ci_low": delta["delta_brier_ci_low"],
                "delta_brier_ci_high": delta["delta_brier_ci_high"],
                "n_groups": delta["n_groups"],
            }
        )
    return rows


def coefficient_table(scope: str, dataset: str, df: pd.DataFrame, fold_coefs: pd.DataFrame) -> pd.DataFrame:
    full = full_sample_coefficients(df)
    summary = (
        fold_coefs.groupby("term", sort=False)["coef"]
        .agg(fold_mean="mean", fold_std="std")
        .reset_index()
    )
    out = full.merge(summary, on="term", how="left")
    out.insert(0, "dataset", dataset)
    out.insert(0, "scope", scope)
    return out


def plot_brier(metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    fig.patch.set_facecolor(BG)
    panels = [
        (axes[0], "local", list(DATASETS), "Local: weights learned inside each dataset"),
        (axes[1], "global", ["all", *DATASETS], "Global: one weight set, scored overall and by dataset"),
    ]
    colors = {"manual": "#56B4E9", "learned": "#D55E00"}
    for ax, scope, slices, title in panels:
        ax.set_facecolor(BG)
        x = np.arange(len(slices))
        width = 0.36
        for offset, method in ((-width / 2, "manual"), (width / 2, "learned")):
            heights = []
            for slice_name in slices:
                hit = metrics[
                    (metrics["scope"] == scope)
                    & (metrics["slice"] == slice_name)
                    & (metrics["method"] == method)
                ]
                heights.append(float(hit["brier"].iloc[0]))
            ax.bar(x + offset, heights, width, label=method, color=colors[method])
        ax.set_xticks(x)
        ax.set_xticklabels(["all datasets" if s == "all" else s for s in slices])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Brier score (lower is better)" if ax is axes[0] else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def format_summary(metrics: pd.DataFrame, coefs: pd.DataFrame, audit: pd.DataFrame) -> str:
    lines = [
        "Learned persuasion logistic regression",
        "Target: E_i = 1 if the initial answer is wrong, else 0.",
        "Features: flip, flip x speed, flip x weak argument, flip x uncertainty at flip.",
        "Learned metrics are 5-fold out-of-fold. Manual is the fixed U^pers formula.",
        "delta Brier = learned - manual. Negative means the learned model is better.",
        "95% CIs are paired cluster bootstraps by item (dataset + original_index).",
        "",
        "Sample:",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"  {row['model']:10} {row['dataset']:10} n={int(row['n']):4}  "
            f"error_rate={row['error_rate']:.3f}  flip_rate={row['flip_rate']:.3f}"
        )
    lines.append("")
    lines.append("Brier / AUROC:")
    show = metrics[metrics["method"].isin(["learned", "manual", "intercept_only"])].copy()
    for _, row in show.sort_values(["scope", "slice", "method"]).iterrows():
        lines.append(
            f"  {row['scope']:6} {row['slice']:10} {row['method']:16} "
            f"Brier={row['brier']:.4f}  AUROC={row['auroc']:.3f}  n={int(row['n'])}"
        )
    lines.append("")
    lines.append("Learned minus manual (Brier):")
    deltas = metrics[metrics["method"] == "learned_minus_manual"]
    for _, row in deltas.iterrows():
        lines.append(
            f"  {row['scope']:6} {row['slice']:10} "
            f"ΔBrier={row['brier']:+.4f}  "
            f"CI [{row['delta_brier_ci_low']:+.4f}, {row['delta_brier_ci_high']:+.4f}]"
        )
    lines.append("")
    lines.append("Full-sample log-odds coefficients (fold mean ± sd in parentheses):")
    for _, row in coefs.iterrows():
        lines.append(
            f"  {row['scope']:6} {row['dataset']:10} {row['term']:32} "
            f"{row['coef']:+.3f}  ({row['fold_mean']:+.3f} ± {row['fold_std']:.3f})"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    df = load_dialogues()
    audit = (
        df.groupby(["model", "dataset"], sort=False)
        .agg(n=("E_i", "size"), error_rate=("E_i", "mean"), flip_rate=("F", "mean"))
        .reset_index()
    )

    metric_rows: list[dict] = []
    coef_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for dataset in DATASETS:
        local = df[df["dataset"] == dataset].reset_index(drop=True)
        learned, intercept_only, folds, fold_coefs = oof_logistic(local, SEED)
        metric_rows.extend(
            evaluate_split(
                "local",
                local,
                learned,
                intercept_only,
                SEED,
                primary_slice=dataset,
                also_by_dataset=False,
            )
        )
        coef_frames.append(coefficient_table("local", dataset, local, fold_coefs))
        pred = local[["model", "dataset", "dialogue_id", "original_index", "group_id", "E_i", "U_pers"]].copy()
        pred["scope"] = "local"
        pred["cv_fold"] = folds
        pred["p_learned"] = learned
        pred["p_intercept"] = intercept_only
        prediction_frames.append(pred)

    learned_g, intercept_g, folds_g, fold_coefs_g = oof_logistic(df, SEED)
    metric_rows.extend(
        evaluate_split(
            "global",
            df,
            learned_g,
            intercept_g,
            SEED,
            primary_slice="all",
            also_by_dataset=True,
        )
    )
    coef_frames.append(coefficient_table("global", "all", df, fold_coefs_g))
    pred_g = df[["model", "dataset", "dialogue_id", "original_index", "group_id", "E_i", "U_pers"]].copy()
    pred_g["scope"] = "global"
    pred_g["cv_fold"] = folds_g
    pred_g["p_learned"] = learned_g
    pred_g["p_intercept"] = intercept_g
    prediction_frames.append(pred_g)

    metrics = pd.DataFrame(metric_rows)
    coefs = pd.concat(coef_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    csv_dir = OUT_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(csv_dir / "metrics.csv", index=False)
    coefs.to_csv(csv_dir / "coefficients.csv", index=False)
    predictions.to_csv(csv_dir / "oof_predictions.csv", index=False)
    audit.to_csv(csv_dir / "sample_sizes.csv", index=False)
    plot_brier(metrics, OUT_DIR / "png" / "brier_manual_vs_learned.png")

    summary = format_summary(metrics, coefs, audit)
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
