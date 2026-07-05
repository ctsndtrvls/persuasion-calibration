"""
Correlations / dependencies across thesis metrics (per dialogue, n=214).

Builds a dialogue-level feature table and plots:
  - Spearman correlation heatmap (all metric pairs)
  - Scatter panel for the strongest associations

Example:
  cd src && python3 analyze_persuasion_dependencies.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from persuasion_arg_quality import TOP_LEVEL_QUALITY_DIMENSIONS
from plot_persuasion_fever_summary import prepare_dialogue_view

FEVER_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_EXPL = FEVER_DIR / "csv" / "expl.csv"
DEFAULT_JUDGE = (
    FEVER_DIR
    / "arg_quality"
    / "arg_quality_long_fever214_all_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv"
)
MAX_TURN = 15

METRIC_COLUMNS = [
    "conf_t0",
    "conf_final",
    "cal_gap_t0",
    "cal_gap_final",
    "correct_t0",
    "correct_final",
    "final_turn",
    "flip_turn",
    "mean_cogency",
    "mean_effectiveness",
    "mean_reasonableness",
]

METRIC_LABELS = {
    "conf_t0": "Confidence (before persuasion)",
    "conf_final": "Confidence (after persuasion)",
    "cal_gap_t0": "Calibration gap (before)",
    "cal_gap_final": "Calibration gap (after)",
    "correct_t0": "Accuracy (before persuasion)",
    "correct_final": "Accuracy (after persuasion)",
    "final_turn": "Dialogue length",
    "flip_turn": "Turn of verdict change",
    "mean_cogency": "Cogency (mean)",
    "mean_effectiveness": "Effectiveness (mean)",
    "mean_reasonableness": "Reasonableness (mean)",
}


def dialogue_arg_quality(judge: pd.DataFrame) -> pd.DataFrame:
    judge = judge.drop_duplicates(subset=["dialogue_id", "turn"], keep="first").copy()
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if d in judge.columns]
    mean_all = judge.groupby("dialogue_id", as_index=False)[dims].mean()
    for d in dims:
        mean_all = mean_all.rename(columns={d: f"mean_{d}"})
    return mean_all


def build_dialogue_features(expl: pd.DataFrame, judge: pd.DataFrame) -> pd.DataFrame:
    view = prepare_dialogue_view(expl)
    aq = dialogue_arg_quality(judge)
    out = view.merge(aq, on="dialogue_id", how="left")

    out["conf_t0"] = pd.to_numeric(out["conf_t0"], errors="coerce")
    out["conf_final"] = pd.to_numeric(out["conf_final"], errors="coerce")
    out["conf_delta"] = pd.to_numeric(out["conf_delta"], errors="coerce")
    out["correct_t0"] = out["correct_t0"].astype(int)
    out["correct_final"] = out["correct_final"].astype(int)
    out["flip"] = out["flip"].astype(int)
    out["cal_gap_t0"] = (out["conf_t0"] / 10.0 - out["correct_t0"]).abs()
    out["cal_gap_final"] = (out["conf_final"] / 10.0 - out["correct_final"]).abs()
    out["appropriate"] = (
        (out["correct_t0"].astype(bool) & ~out["flip"].astype(bool))
        | (~out["correct_t0"].astype(bool) & out["flip"].astype(bool))
    ).astype(int)
    out["final_turn"] = pd.to_numeric(out["final_turn"], errors="coerce")
    out["flip_turn"] = pd.to_numeric(out["flip_turn_num"], errors="coerce")
    return out


def correlation_tables(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = features[METRIC_COLUMNS].copy()
    n = len(METRIC_COLUMNS)
    rho = np.eye(n)
    pval = np.zeros((n, n))
    cols = METRIC_COLUMNS
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            pair = data[[a, b]].dropna()
            if len(pair) < 3 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
                rho[i, j] = rho[j, i] = np.nan
                pval[i, j] = pval[j, i] = np.nan
                continue
            r, p = stats.spearmanr(pair[a], pair[b])
            r, p = float(r), float(p)
            rho[i, j] = rho[j, i] = r
            pval[i, j] = pval[j, i] = p
    rho_df = pd.DataFrame(rho, index=cols, columns=cols)
    pval_df = pd.DataFrame(pval, index=cols, columns=cols)
    return rho_df, pval_df


def top_correlation_pairs(rho: pd.DataFrame, pval: pd.DataFrame, *, top_n: int = 6) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cols = list(rho.columns)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            rows.append(
                {
                    "metric_a": a,
                    "metric_b": b,
                    "label_a": METRIC_LABELS[a],
                    "label_b": METRIC_LABELS[b],
                    "spearman_rho": rho.iloc[i, j],
                    "p_value": pval.iloc[i, j],
                }
            )
    out = pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False)
    return out.head(top_n).reset_index(drop=True)


def _star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_correlation_heatmap(
    rho: pd.DataFrame,
    pval: pd.DataFrame,
    dataset: str,
    target: str,
    out_png: Path,
) -> None:
    labels = [METRIC_LABELS[c] for c in rho.columns]
    annot = np.empty(rho.shape, dtype=object)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            r = rho.iloc[i, j]
            p = pval.iloc[i, j]
            if i == j or np.isnan(r):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{r:.2f}{_star(p)}"

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        rho,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=annot,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(
        f"Dependencies between thesis metrics — {dataset} · {target}\n"
        "Spearman ρ per dialogue (n=214); stars mark significance: * p<0.05, ** p<0.01, *** p<0.001",
        fontsize=12,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_scatter_pairs(
    features: pd.DataFrame,
    pairs: pd.DataFrame,
    dataset: str,
    target: str,
    out_png: Path,
) -> None:
    n = len(pairs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.8 * nrows))
    axes = np.atleast_1d(axes).flatten()

    palette = {"Accurate before persuasion": "#4C78A8", "Inaccurate before persuasion": "#F58518"}
    features = features.copy()
    features["accuracy_group"] = np.where(
        features["correct_t0"] == 1,
        "Accurate before persuasion",
        "Inaccurate before persuasion",
    )

    for ax, (_, row) in zip(axes, pairs.iterrows()):
        a, b = row["metric_a"], row["metric_b"]
        plot_df = features[[a, b, "accuracy_group"]].dropna()
        sns.scatterplot(
            data=plot_df,
            x=a,
            y=b,
            hue="accuracy_group",
            palette=palette,
            ax=ax,
            alpha=0.55,
            s=28,
            edgecolor="white",
            linewidth=0.3,
        )
        sns.regplot(
            data=plot_df,
            x=a,
            y=b,
            scatter=False,
            ax=ax,
            color="#333333",
            line_kws={"linewidth": 1.1, "alpha": 0.7},
        )
        ax.set_xlabel(METRIC_LABELS[a], fontsize=9)
        ax.set_ylabel(METRIC_LABELS[b], fontsize=9)
        ax.set_title(f"ρ = {row['spearman_rho']:.2f}{_star(row['p_value'])}", fontsize=10)
        ax.legend(fontsize=7, title=None, loc="best")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Strongest metric associations — {dataset} · {target}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def target_label(df: pd.DataFrame) -> str:
    if "target_model" not in df.columns or not df["target_model"].notna().any():
        return "DeepSeek"
    raw = str(df["target_model"].dropna().iloc[0]).lower()
    return "DeepSeek" if "deepseek" in raw else str(df["target_model"].dropna().iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Correlations across persuasion-calibration metrics.")
    ap.add_argument("--expl", type=Path, default=DEFAULT_EXPL)
    ap.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--out-dir", type=Path, default=FEVER_DIR)
    ap.add_argument("--top-pairs", type=int, default=6)
    args = ap.parse_args()

    expl = pd.read_csv(args.expl)
    judge = pd.read_csv(args.judge)
    features = build_dialogue_features(expl, judge)
    rho, pval = correlation_tables(features)
    top_pairs = top_correlation_pairs(rho, pval, top_n=args.top_pairs)

    dataset = "FEVER"
    if "dataset" in expl.columns and expl["dataset"].notna().any():
        dataset = str(expl["dataset"].dropna().iloc[0]).upper()
    target = target_label(expl)

    csv_dir = args.out_dir / "csv"
    png_dir = args.out_dir / "png"
    csv_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    out_features = csv_dir / "metric_dependencies_features.csv"
    out_rho = csv_dir / "metric_dependencies_spearman.csv"
    out_p = csv_dir / "metric_dependencies_pvalues.csv"
    out_top = csv_dir / "metric_dependencies_top_pairs.csv"
    out_heat = png_dir / "metric_dependencies_heatmap.png"
    out_scatter = png_dir / "metric_dependencies_scatter.png"

    features.to_csv(out_features, index=False)
    rho.to_csv(out_rho)
    pval.to_csv(out_p)
    top_pairs.to_csv(out_top, index=False)
    plot_correlation_heatmap(rho, pval, dataset, target, out_heat)
    plot_top_scatter_pairs(features, top_pairs, dataset, target, out_scatter)

    print(f"Saved features: {out_features}")
    print(f"Saved Spearman matrix: {out_rho}")
    print(f"Saved top pairs: {out_top}")
    print(f"Saved heatmap: {out_heat}")
    print(f"Saved scatter panel: {out_scatter}")
    print("\nStrongest associations:")
    print(top_pairs.to_string(index=False))


if __name__ == "__main__":
    main()
