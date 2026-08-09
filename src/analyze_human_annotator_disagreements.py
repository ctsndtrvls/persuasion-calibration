"""
Confusion matrices, label-skew diagnostics, and review-case selection for human annotators.

Example:
  python3 analyze_human_annotator_disagreements.py \\
    --input-dir ../output_wood/persuasion/human_annotators
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

from analyze_human_annotator_agreement import SCORE_COLUMNS, load_annotator_csv

LABELS = list(range(4))  # scores are 0-3
ANNOTATOR_NAMES = ("annotator_1", "annotator_2", "annotator_3")
PAIR_LABELS = (
    ("annotator_1", "annotator_2", "1 ↔ 2"),
    ("annotator_1", "annotator_3", "1 ↔ 3"),
    ("annotator_2", "annotator_3", "2 ↔ 3"),
)


def annotator_csv_paths(input_dir: Path) -> list[Path]:
    paths = [input_dir / f"human_annotation_annotator_{i}.csv" for i in range(1, 4)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing annotator CSV files: " + ", ".join(p.name for p in missing))
    return paths


def load_merged_annotators(input_dir: Path) -> pd.DataFrame:
    paths = annotator_csv_paths(input_dir)

    base = load_annotator_csv(paths[0])[["item_id", "dataset", "claim", "counterargument"]].copy()
    for i, path in enumerate(paths, start=1):
        df = load_annotator_csv(path)
        for dim in SCORE_COLUMNS:
            base[f"annotator_{i}__{dim}"] = base["item_id"].map(
                df.set_index("item_id")[dim].astype(float)
            )
    return base.sort_values("item_id").reset_index(drop=True)


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    mat = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for x, y in zip(a, b):
        if np.isnan(x) or np.isnan(y):
            continue
        mat[int(x), int(y)] += 1
    return pd.DataFrame(mat, index=LABELS, columns=LABELS)


def label_distribution(df: pd.DataFrame, annotator: str, dimension: str) -> pd.Series:
    values = df[f"{annotator}__{dimension}"].dropna().astype(int)
    counts = values.value_counts().reindex(LABELS, fill_value=0).astype(int)
    return counts / counts.sum()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    eps = 1e-12

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2((a[mask] + eps) / (b[mask] + eps))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def skew_report(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dim in SCORE_COLUMNS:
        pooled = np.zeros(len(LABELS), dtype=float)
        for name in ANNOTATOR_NAMES:
            pooled += label_distribution(df, name, dim).to_numpy()
        pooled /= len(ANNOTATOR_NAMES)

        for name in ANNOTATOR_NAMES:
            dist = label_distribution(df, name, dim)
            values = df[f"{name}__{dim}"].dropna().astype(int)
            rows.append(
                {
                    "dimension": dim,
                    "annotator": name,
                    "mean_score": float(values.mean()),
                    "pct_score_0": float(dist.loc[0]),
                    "pct_score_1": float(dist.loc[1]),
                    "pct_score_2": float(dist.loc[2]),
                    "pct_score_3": float(dist.loc[3]),
                    "js_vs_pooled": js_divergence(dist.to_numpy(), pooled),
                    "dominant_label": int(dist.idxmax()),
                    "dominant_label_pct": float(dist.max()),
                }
            )
    return pd.DataFrame(rows)


def overall_skew_summary(skew: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in ANNOTATOR_NAMES:
        sub = skew[skew["annotator"] == name]
        rows.append(
            {
                "annotator": name,
                "mean_js_vs_pooled": float(sub["js_vs_pooled"].mean()),
                "mean_score_overall": float(sub["mean_score"].mean()),
                "avg_dominant_label_pct": float(sub["dominant_label_pct"].mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["mean_js_vs_pooled", "avg_dominant_label_pct"],
        ascending=False,
    )
    out["rank"] = range(1, len(out) + 1)
    return out


def disagreement_score(row: pd.Series, dimension: str) -> float:
    values = [row[f"{name}__{dimension}"] for name in ANNOTATOR_NAMES]
    arr = np.array(values, dtype=float)
    if np.isnan(arr).any():
        return float("nan")
    return float(arr.max() - arr.min())


def item_disagreement_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        per_dim = {dim: disagreement_score(row, dim) for dim in SCORE_COLUMNS}
        max_dim = max(per_dim, key=lambda d: per_dim[d])
        rows.append(
            {
                "item_id": int(row["item_id"]),
                "dataset": row["dataset"],
                "claim": row["claim"],
                "counterargument": row["counterargument"],
                "max_disagreement": float(max(per_dim.values())),
                "worst_dimension": max_dim,
                **{f"disagreement__{dim}": per_dim[dim] for dim in SCORE_COLUMNS},
                **{
                    f"{name}__{dim}": int(row[f"{name}__{dim}"])
                    for name in ANNOTATOR_NAMES
                    for dim in SCORE_COLUMNS
                },
            }
        )
    out = pd.DataFrame(rows)
    out["score_vector"] = out.apply(
        lambda r: tuple(
            int(r[f"{name}__{r['worst_dimension']}"]) for name in ANNOTATOR_NAMES
        ),
        axis=1,
    )
    return out.sort_values(
        ["max_disagreement", "worst_dimension", "item_id"],
        ascending=[False, True, True],
    )


def select_review_cases(
    disagreements: pd.DataFrame,
    *,
    n_cases: int = 15,
) -> pd.DataFrame:
    per_dim = max(1, n_cases // len(SCORE_COLUMNS))
    remainder = n_cases - per_dim * len(SCORE_COLUMNS)
    dim_limits = {dim: per_dim + (1 if i < remainder else 0) for i, dim in enumerate(SCORE_COLUMNS)}

    selected: list[pd.Series] = []
    seen_patterns: set[tuple[object, ...]] = set()

    for dim in SCORE_COLUMNS:
        dim_items = disagreements[disagreements["worst_dimension"] == dim].copy()
        dim_items["pattern"] = dim_items.apply(
            lambda r: tuple(int(r[f"{name}__{dim}"]) for name in ANNOTATOR_NAMES),
            axis=1,
        )
        dim_items = dim_items.sort_values("max_disagreement", ascending=False)
        picked = 0
        for _, row in dim_items.iterrows():
            pattern = (dim, row["pattern"])
            if pattern in seen_patterns:
                continue
            selected.append(row)
            seen_patterns.add(pattern)
            picked += 1
            if picked >= dim_limits[dim]:
                break

    if len(selected) < n_cases:
        for _, row in disagreements.iterrows():
            pattern = (row["worst_dimension"], row["score_vector"])
            if pattern in seen_patterns:
                continue
            selected.append(row)
            seen_patterns.add(pattern)
            if len(selected) >= n_cases:
                break

    out = pd.DataFrame(selected).head(n_cases).copy()
    out["review_priority"] = range(1, len(out) + 1)
    return out


def plot_confusion_matrices(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(len(SCORE_COLUMNS), len(PAIR_LABELS), figsize=(14, 11))
    fig.suptitle("Human annotator confusion matrices (rows = annotator A, cols = annotator B)", y=0.995)

    for i, dim in enumerate(SCORE_COLUMNS):
        for j, (a_name, b_name, pair_label) in enumerate(PAIR_LABELS):
            ax = axes[i, j]
            a = df[f"{a_name}__{dim}"].to_numpy(dtype=float)
            b = df[f"{b_name}__{dim}"].to_numpy(dtype=float)
            cm = confusion_matrix(a, b)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
                linewidths=0.5,
                linecolor="white",
            )
            if i == 0:
                ax.set_title(pair_label, fontsize=11, pad=8)
            if j == 0:
                ax.set_ylabel(dim.capitalize(), fontsize=11)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Annotator B")
            if j == 0:
                ax.set_ylabel(f"{dim.capitalize()}\nAnnotator A")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_label_distributions(skew: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(SCORE_COLUMNS), figsize=(13, 4.2), sharey=True)
    fig.suptitle("Label distributions by annotator", y=1.02)

    palette = {"annotator_1": "#4C78A8", "annotator_2": "#F58518", "annotator_3": "#54A24B"}
    x = np.arange(len(LABELS))
    width = 0.24

    for ax, dim in zip(axes, SCORE_COLUMNS):
        sub = skew[skew["dimension"] == dim]
        for k, name in enumerate(ANNOTATOR_NAMES):
            row = sub[sub["annotator"] == name].iloc[0]
            heights = [row[f"pct_score_{label}"] for label in LABELS]
            ax.bar(x + (k - 1) * width, heights, width=width, label=name, color=palette[name])
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS)
        ax.set_xlabel("Score")
        ax.set_title(dim.capitalize())
        ax.set_ylim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel("Share of labels")
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(
    out_path: Path,
    skew_summary: pd.DataFrame,
    skew: pd.DataFrame,
    review_cases: pd.DataFrame,
) -> None:
    most_skewed = skew_summary.iloc[0]
    lines = [
        "# Human annotator disagreement diagnostics",
        "",
        "## Most skewed annotator",
        "",
        f"- **{most_skewed['annotator']}** ranks highest on average label skew "
        f"(mean JS divergence vs pooled distribution = {most_skewed['mean_js_vs_pooled']:.3f}).",
        f"- Overall mean score: {most_skewed['mean_score_overall']:.2f} "
        f"(lower = more low-score labels).",
        "",
        "## Per-dimension skew",
        "",
        "| dimension | annotator | mean score | dominant label | dominant % | JS vs pooled |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in skew.sort_values(["dimension", "js_vs_pooled"], ascending=[True, False]).iterrows():
        lines.append(
            f"| {row['dimension']} | {row['annotator']} | {row['mean_score']:.2f} | "
            f"{row['dominant_label']} | {row['dominant_label_pct']:.0%} | {row['js_vs_pooled']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Suggested review cases",
            "",
            "These items combine large disagreement with diverse score patterns across dimensions.",
            "",
            "| priority | item_id | dimension | scores (A1, A2, A3) | claim |",
            "|---:|---:|---|---|---|",
        ]
    )
    for _, row in review_cases.iterrows():
        dim = row["worst_dimension"]
        scores = ", ".join(str(int(row[f"{name}__{dim}"])) for name in ANNOTATOR_NAMES)
        claim = str(row["claim"]).replace("|", "/")[:90]
        lines.append(
            f"| {int(row['review_priority'])} | {int(row['item_id'])} | {dim} | {scores} | {claim} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Confusion matrices and review-case selection.")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=_PROJECT_ROOT / "output_wood" / "persuasion" / "human_annotators",
    )
    ap.add_argument("--n-cases", type=int, default=15)
    args = ap.parse_args()

    out_dir = args.input_dir
    df = load_merged_annotators(out_dir)

    skew = skew_report(df)
    skew_summary = overall_skew_summary(skew)
    disagreements = item_disagreement_summary(df)
    review_cases = select_review_cases(disagreements, n_cases=args.n_cases)

    plot_confusion_matrices(df, out_dir / "human_annotator_confusion_matrices.png")
    plot_label_distributions(skew, out_dir / "human_annotator_label_distributions.png")

    skew.to_csv(out_dir / "human_annotator_label_skew.csv", index=False)
    skew_summary.to_csv(out_dir / "human_annotator_skew_summary.csv", index=False)
    disagreements.to_csv(out_dir / "human_annotator_disagreements_all.csv", index=False)
    review_cases.to_csv(out_dir / "human_annotator_review_cases.csv", index=False)
    write_summary_md(
        out_dir / "human_annotator_disagreement_summary.md",
        skew_summary,
        skew,
        review_cases,
    )

    print("Most skewed annotator:")
    print(skew_summary.to_string(index=False))
    print(f"\nSaved review cases -> {out_dir / 'human_annotator_review_cases.csv'}")
    print(f"Saved confusion matrices -> {out_dir / 'human_annotator_confusion_matrices.png'}")


if __name__ == "__main__":
    main()
