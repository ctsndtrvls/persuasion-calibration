"""
Leave-one-feature-out ablation of the learned persuasion logistic model.

For each local dataset and for the global model, refit Equation (learned)
five times on the same grouped folds, each time dropping one term:

  flip, flip x speed, flip x weak argument, flip x low confidence at flip

The increase in out-of-fold Brier relative to the full model is how much
that feature was doing. A positive increase means the prediction got worse
without the feature.

Example:
  cd src && python3 ablate_persuasion_logistic.py
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

from evaluate_persuasion_uncertainty import auroc_score, brier_score
from learn_persuasion_logistic import (
    DATASETS,
    N_FOLDS,
    SEED,
    assign_folds,
    design_matrix,
    fit_predict_fold,
    load_dialogues,
    paired_delta_brier,
)

OUT_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "learned_logistic" / "ablation"
PAPER_FIG = _PROJECT_ROOT / "paper" / "figures"
N_BOOT = 2000

# Column index in design_matrix, after the intercept at 0.
DROP_AT = {
    "flip": 1,
    "speed": 2,
    "weak_argument": 3,
    "low_confidence": 4,
}
FEATURE_ORDER = list(DROP_AT)
FEATURE_LABELS = {
    "flip": "Flip",
    "speed": "Speed",
    "weak_argument": "Weak\nargument",
    "low_confidence": "Low\nconfidence\nat flip",
}


def oof_dropping(df: pd.DataFrame, seed: int, drop: str | None) -> np.ndarray:
    """Out-of-fold probabilities. drop=None keeps every feature."""
    X = design_matrix(df)
    if drop is not None:
        X = np.delete(X, DROP_AT[drop], axis=1)
    y = df["E_i"].to_numpy(dtype=int)
    folds = assign_folds(df, N_FOLDS, seed)
    oof = np.full(len(df), np.nan)
    for fold_id in range(N_FOLDS):
        test = folds == fold_id
        train = ~test
        oof[test] = fit_predict_fold(X[train], y[train], X[test])
    if np.isnan(oof).any():
        raise RuntimeError(f"Incomplete predictions when dropping {drop}.")
    return oof


def slices_for(scope: str, df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    if scope == "local":
        return [("all", np.ones(len(df), dtype=bool))]
    rows = [("all", np.ones(len(df), dtype=bool))]
    for dataset in DATASETS:
        rows.append((dataset, (df["dataset"] == dataset).to_numpy()))
    return rows


def collect(scope: str, slice_name: str, df: pd.DataFrame, predictions: dict[str, np.ndarray]) -> list[dict]:
    y = df["E_i"].to_numpy(dtype=int)
    groups = df["group_id"].to_numpy()
    full = predictions["full"]
    rows = []
    for feature in ["full", *FEATURE_ORDER]:
        pred = predictions[feature]
        row = {
            "scope": scope,
            "slice": slice_name,
            "feature_removed": feature,
            "n": int(len(y)),
            "brier": brier_score(pred, y),
            "auroc": auroc_score(pred, y),
        }
        if feature != "full":
            delta = paired_delta_brier(y, pred, full, groups, N_BOOT, SEED)
            row["delta_brier"] = delta["delta_brier"]
            row["delta_brier_ci_low"] = delta["delta_brier_ci_low"]
            row["delta_brier_ci_high"] = delta["delta_brier_ci_high"]
        rows.append(row)
    return rows


def run_scope_fixed(scope: str, df: pd.DataFrame) -> list[dict]:
    predictions = {"full": oof_dropping(df, SEED, None)}
    for feature in FEATURE_ORDER:
        predictions[feature] = oof_dropping(df, SEED, feature)
    rows: list[dict] = []
    for slice_name, mask in slices_for(scope, df):
        part = df.loc[mask].reset_index(drop=True)
        sliced = {name: pred[mask] for name, pred in predictions.items()}
        rows.extend(collect(scope, slice_name, part, sliced))
    return rows


def plot_ablation(metrics: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    show = metrics[
        (metrics["feature_removed"] != "full")
        & (
            ((metrics["scope"] == "local") & (metrics["slice"] == "all"))
            | ((metrics["scope"] == "global") & (metrics["slice"] == "all"))
        )
    ].copy()
    show["setting"] = show["scope"].map({"local": "Local", "global": "Global"})
    # Local rows have slice "all" but we need the dataset name. The caller
    # stores local results with slice equal to the dataset. Rebuild below.
    del show

    frames = []
    for dataset in DATASETS:
        part = metrics[
            (metrics["scope"] == "local")
            & (metrics["slice"] == dataset)
            & (metrics["feature_removed"] != "full")
        ].copy()
        part["setting"] = f"Local {dataset.upper() if dataset != 'popqa' else 'PopQA'}"
        part.loc[part["setting"] == "Local FEVER", "setting"] = "Local FEVER"
        part.loc[part["setting"] == "Local DEBATEQA", "setting"] = "Local DebateQA"
        frames.append(part)
    global_part = metrics[
        (metrics["scope"] == "global")
        & (metrics["slice"] == "all")
        & (metrics["feature_removed"] != "full")
    ].copy()
    global_part["setting"] = "Global"
    frames.append(global_part)
    show = pd.concat(frames, ignore_index=True)

    settings = ["Local FEVER", "Local PopQA", "Local DebateQA", "Global"]
    colors = {
        "Local FEVER": "#6E8CA0",
        "Local PopQA": "#C4A46A",
        "Local DebateQA": "#7D9A84",
        "Global": "#3E4C59",
    }
    # ~3.4 in is one ACL column. The axes are wide enough that
    # "argument" and "confidence" stay apart at 10 pt.
    fig, ax = plt.subplots(figsize=(3.48, 4.55))
    x = np.arange(len(FEATURE_ORDER))
    width = 0.18
    for i, setting in enumerate(settings):
        part = show[show["setting"] == setting].set_index("feature_removed")
        heights = [float(part.loc[feature, "delta_brier"]) for feature in FEATURE_ORDER]
        ax.bar(
            x + (i - 1.5) * width,
            heights,
            width,
            label=setting,
            color=colors[setting],
            edgecolor="none",
        )
    ax.axhline(0, color="#B0B0B0", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[feature] for feature in FEATURE_ORDER], fontsize=10)
    ax.set_ylabel("Increase in Brier when removed", fontsize=10)
    ax.set_title(
        "Above zero: the prediction got worse\nwithout this feature",
        loc="left",
        fontsize=10,
        pad=6,
    )
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.48, 1.34),
        fontsize=9,
        handlelength=1.3,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.6)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.20, top=0.74)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_dialogues()
    rows: list[dict] = []
    for dataset in DATASETS:
        local = df[df["dataset"] == dataset].reset_index(drop=True)
        part_rows = run_scope_fixed("local", local)
        for row in part_rows:
            row["slice"] = dataset
        rows.extend(part_rows)
    rows.extend(run_scope_fixed("global", df))
    metrics = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_DIR / "ablation_metrics.csv", index=False)
    plot_ablation(metrics, PAPER_FIG / "uncertainty_ablation")

    lines = [
        "Leave-one-feature-out. Delta Brier = Brier without the feature minus Brier of the full model.",
        "Positive delta: removing the feature made the prediction worse.",
        "",
    ]
    view = metrics[metrics["feature_removed"] != "full"].sort_values(["scope", "slice", "feature_removed"])
    for _, row in view.iterrows():
        lines.append(
            f"{row['scope']:6} {row['slice']:10} without {row['feature_removed']:16} "
            f"Brier={row['brier']:.4f}  Δ={row['delta_brier']:+.4f}  "
            f"CI [{row['delta_brier_ci_low']:+.4f}, {row['delta_brier_ci_high']:+.4f}]  "
            f"AUROC={row['auroc']:.3f}"
        )
    full = metrics[metrics["feature_removed"] == "full"]
    lines += ["", "Full model:"]
    for _, row in full.iterrows():
        lines.append(
            f"{row['scope']:6} {row['slice']:10} Brier={row['brier']:.4f}  AUROC={row['auroc']:.3f}"
        )
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
