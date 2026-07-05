"""
Argument quality vs verdict flip (dialogue-level and flip-turn analysis).

Example:
  python3 analyze_arg_quality_flip.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from persuasion_arg_quality import TOP_LEVEL_QUALITY_DIMENSIONS

FEVER_DIR = _PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_EXPL = FEVER_DIR / "csv" / "expl.csv"
DEFAULT_JUDGE = (
    FEVER_DIR
    / "arg_quality"
    / "arg_quality_long_fever214_all_turns_v1__top3__openrouter__openai_gpt-5.4-mini.csv"
)
DIM_COLORS = {
    "cogency": "#E45756",
    "effectiveness": "#F58518",
    "reasonableness": "#54A24B",
}


def dialogue_flip_meta(expl: pd.DataFrame) -> pd.DataFrame:
    snap = (
        expl.sort_index()
        .drop_duplicates(subset=["dialogue_id", "turn"], keep="first")
        .copy()
    )
    last = snap.sort_values(["dialogue_id", "turn"]).groupby("dialogue_id", as_index=False).tail(1)
    out = last[["dialogue_id", "flipped_from_initial", "flip_turn"]].copy()
    out["flip_final"] = pd.to_numeric(out["flipped_from_initial"], errors="coerce").fillna(0).astype(int)
    out["flip_turn"] = pd.to_numeric(out["flip_turn"], errors="coerce")
    return out[["dialogue_id", "flip_final", "flip_turn"]]


def build_dialogue_quality(judge: pd.DataFrame, flip_meta: pd.DataFrame) -> pd.DataFrame:
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if d in judge.columns]
    judge = judge.drop_duplicates(subset=["dialogue_id", "turn"], keep="first").copy()
    if "mean_score" not in judge.columns:
        judge["mean_score"] = judge[dims].mean(axis=1)

    agg = {f"mean_{d}": (d, "mean") for d in dims}
    agg["mean_overall"] = ("mean_score", "mean")
    agg["n_scored_turns"] = ("turn", "count")
    dlg = judge.groupby("dialogue_id", as_index=False).agg(**agg)
    dlg = dlg.merge(flip_meta, on="dialogue_id", how="inner")
    dlg["flip_label"] = dlg["flip_final"].map({0: "Not flipped", 1: "Flipped"})
    return dlg


def summarize_flip_turn_quality(judge: pd.DataFrame, dlg: pd.DataFrame) -> pd.DataFrame:
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if d in judge.columns]
    flipped = dlg[dlg["flip_final"] == 1].copy()
    sub = judge.merge(flipped[["dialogue_id", "flip_turn"]], on="dialogue_id", how="inner")
    sub["flip_turn"] = pd.to_numeric(sub["flip_turn"], errors="coerce")
    sub = sub[sub["flip_turn"].notna()].copy()
    sub["at_flip_turn"] = sub["turn"] == sub["flip_turn"].astype(int)
    sub["turn_group"] = np.where(sub["at_flip_turn"], "At flip turn", "Other turns")

    rows = []
    for group, g in sub.groupby("turn_group"):
        row = {"turn_group": group, "n_counterarguments": len(g)}
        for d in dims:
            row[f"mean_{d}"] = float(g[d].mean())
        if "mean_score" in g.columns:
            row["mean_overall"] = float(g["mean_score"].mean())
        else:
            row["mean_overall"] = float(g[dims].mean(axis=1).mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    order = ["At flip turn", "Other turns"]
    summary["turn_group"] = pd.Categorical(summary["turn_group"], categories=order, ordered=True)
    return summary.sort_values("turn_group")


def fit_logistic_flip(dlg: pd.DataFrame) -> pd.DataFrame:
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if f"mean_{d}" in dlg.columns]
    feature_cols = [f"mean_{d}" for d in dims]
    work = dlg.dropna(subset=feature_cols + ["flip_final"]).copy()
    if work.empty or work["flip_final"].nunique() < 2:
        return pd.DataFrame()

    X = work[feature_cols].to_numpy(dtype=float)
    y = work["flip_final"].to_numpy(dtype=int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(Xs, y)

    rows = []
    for i, d in enumerate(dims):
        coef = float(model.coef_[0, i])
        # Wald-style SE via inverse Hessian approximation (bootstrap-free, rough)
        p = model.predict_proba(Xs)[:, 1]
        w = p * (1 - p)
        w = np.clip(w, 1e-6, None)
        xw = Xs * np.sqrt(w)[:, None]
        try:
            cov = np.linalg.inv(xw.T @ xw)
            se = float(np.sqrt(cov[i, i]))
        except np.linalg.LinAlgError:
            se = float("nan")
        z = coef / se if se and not np.isnan(se) else float("nan")
        pval = float(2 * stats.norm.sf(abs(z))) if not np.isnan(z) else float("nan")
        rows.append(
            {
                "dimension": d,
                "coef_std": coef,
                "se_std": se,
                "odds_ratio_per_sd": float(np.exp(coef)),
                "p_value": pval,
            }
        )
    out = pd.DataFrame(rows)
    out["intercept"] = float(model.intercept_[0])
    out["n_dialogues"] = len(work)
    out["n_flipped"] = int(y.sum())
    out["pseudo_r2"] = float(model.score(Xs, y))
    return out


def plot_flip_analysis(
    dlg: pd.DataFrame,
    flip_turn_summary: pd.DataFrame,
    logistic: pd.DataFrame,
    out_png: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 13), gridspec_kw={"height_ratios": [1.0, 1.05, 0.95]})

    # ── (1) Boxplot: mean arg quality × flip_final ───────────────────────
    ax0 = axes[0]
    order = ["Not flipped", "Flipped"]
    palette = {"Not flipped": "#9ECAE1", "Flipped": "#F4A582"}
    sns.boxplot(
        data=dlg,
        x="flip_label",
        y="mean_overall",
        order=order,
        hue="flip_label",
        palette=palette,
        width=0.55,
        linewidth=1.2,
        ax=ax0,
        legend=False,
    )
    sns.stripplot(
        data=dlg,
        x="flip_label",
        y="mean_overall",
        order=order,
        color="#333333",
        alpha=0.35,
        size=3,
        jitter=0.18,
        ax=ax0,
    )
    n_nf = int((dlg["flip_final"] == 0).sum())
    n_f = int((dlg["flip_final"] == 1).sum())
    mu_nf = dlg.loc[dlg["flip_final"] == 0, "mean_overall"].mean()
    mu_f = dlg.loc[dlg["flip_final"] == 1, "mean_overall"].mean()
    ax0.set_xlabel("")
    ax0.set_ylabel("Mean argument quality per dialogue (0–3)")
    ax0.set_title(
        "Mean argument quality by final verdict flip\n"
        f"(not flipped n={n_nf}, μ={mu_nf:.2f}; flipped n={n_f}, μ={mu_f:.2f})",
        pad=8,
    )

    # ── (2) Bar chart: flip turn vs other turns (flipped dialogues only) ─
    ax1 = axes[1]
    dims = [d for d in TOP_LEVEL_QUALITY_DIMENSIONS if f"mean_{d}" in flip_turn_summary.columns]
    x = np.arange(len(dims))
    width = 0.34
    at_flip = flip_turn_summary[flip_turn_summary["turn_group"] == "At flip turn"].iloc[0]
    other = flip_turn_summary[flip_turn_summary["turn_group"] == "Other turns"].iloc[0]
    for i, group_row, offset, label, color in [
        (0, at_flip, -width / 2, "At flip turn", "#E45756"),
        (1, other, width / 2, "Other turns", "#4C78A8"),
    ]:
        vals = [group_row[f"mean_{d}"] for d in dims]
        bars = ax1.bar(x + offset, vals, width=width, label=label, color=color, alpha=0.9)
        for bar, val in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in dims])
    ax1.set_ylim(0, 3.1)
    ax1.set_ylabel("Mean judge score (0–3)")
    n_at = int(at_flip["n_counterarguments"])
    n_other = int(other["n_counterarguments"])
    ax1.set_title(
        "Argument quality at flip turn vs other turns (flipped dialogues only)\n"
        f"(at flip turn n={n_at}; other turns n={n_other})",
        pad=8,
    )
    ax1.legend(loc="upper right", fontsize=9)

    # ── (3) Logistic regression coefficients ─────────────────────────────
    ax2 = axes[2]
    if logistic.empty:
        ax2.set_title("Logistic regression (insufficient data)")
        ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
    else:
        dims_lr = logistic["dimension"].tolist()
        coefs = logistic["coef_std"].to_numpy()
        ses = logistic["se_std"].to_numpy()
        colors = [DIM_COLORS.get(d, "#666666") for d in dims_lr]
        y_pos = np.arange(len(dims_lr))
        ax2.barh(y_pos, coefs, color=colors, alpha=0.85, height=0.55)
        for i, (c, se) in enumerate(zip(coefs, ses)):
            if not np.isnan(se):
                ax2.errorbar(c, i, xerr=1.96 * se, fmt="none", ecolor="#333333", capsize=3)
            ax2.text(
                c + (0.03 if c >= 0 else -0.03),
                i,
                f"{c:+.2f}",
                va="center",
                ha="left" if c >= 0 else "right",
                fontsize=9,
            )
        ax2.axvline(0, color="gray", linestyle="--", linewidth=0.9)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([d.capitalize() for d in dims_lr])
        ax2.set_xlabel("Standardized log-odds coefficient (per 1 SD increase)")
        n_d = int(logistic["n_dialogues"].iloc[0])
        n_fl = int(logistic["n_flipped"].iloc[0])
        ax2.set_title(
            "Logistic regression: P(flip_final) ~ mean cogency + effectiveness + reasonableness\n"
            f"(dialogue-level means; n={n_d}, flipped={n_fl})",
            pad=8,
        )

    fig.suptitle(
        "Argument quality and verdict flip — FEVER / DeepSeek target / GPT persuader",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Argument quality vs verdict flip.")
    ap.add_argument("--expl", type=Path, default=DEFAULT_EXPL)
    ap.add_argument("--judge-csv", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--out-dir", type=Path, default=FEVER_DIR)
    args = ap.parse_args()

    if not args.judge_csv.exists():
        raise SystemExit(f"Judge CSV not found: {args.judge_csv}")
    if not args.expl.exists():
        raise SystemExit(f"expl.csv not found: {args.expl}")

    judge = pd.read_csv(args.judge_csv)
    expl = pd.read_csv(args.expl)
    flip_meta = dialogue_flip_meta(expl)
    dlg = build_dialogue_quality(judge, flip_meta)
    flip_turn_summary = summarize_flip_turn_quality(judge, dlg)
    logistic = fit_logistic_flip(dlg)

    out_dir = args.out_dir / "arg_quality"
    png_dir = args.out_dir / "png"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    dlg.to_csv(out_dir / "arg_quality_flip_dialogues.csv", index=False)
    flip_turn_summary.to_csv(out_dir / "arg_quality_flip_turn_summary.csv", index=False)
    if not logistic.empty:
        logistic.to_csv(out_dir / "arg_quality_flip_logistic.csv", index=False)

    out_png = png_dir / "arg_quality_flip.png"
    plot_flip_analysis(dlg, flip_turn_summary, logistic, out_png)

    print(f"Dialogues: {len(dlg)} (flipped={int(dlg['flip_final'].sum())})")
    print(f"Saved figure: {out_png}")
    print(f"Saved CSV: {out_dir / 'arg_quality_flip_dialogues.csv'}")
    if not logistic.empty:
        print("\nLogistic regression (standardized coefficients):")
        for _, r in logistic.iterrows():
            print(
                f"  {r['dimension']:14s}  coef={r['coef_std']:+.3f}  "
                f"OR/SD={r['odds_ratio_per_sd']:.3f}  p={r['p_value']:.3f}"
            )


if __name__ == "__main__":
    main()
