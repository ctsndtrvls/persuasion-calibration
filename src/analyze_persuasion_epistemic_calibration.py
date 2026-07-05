"""
Epistemic markers in the target model's decision_explanation.

Reads column `decision_explanation` and adds two columns:
  - decision_explanation_epistemic_markers   pipe-separated marker strings found in the text
  - decision_explanation_epistemic_marker_n  count of markers

Uses marker lists from linguistic_confidence.py.
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
import pandas as pd
import seaborn as sns

from linguistic_confidence import (
    CERTAINTY_MARKERS,
    UNCERTAINTY_MARKERS,
    extract_epistemic_markers,
)

PROJECT_ROOT = _PROJECT_ROOT
FEVER_DEEPSEEK_DIR = PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_INPUT = FEVER_DEEPSEEK_DIR / "csv" / "expl.csv"
DEFAULT_OUT_DIR = FEVER_DEEPSEEK_DIR

MARKER_LIST_COL = "decision_explanation_epistemic_markers"
MARKER_COUNT_COL = "decision_explanation_epistemic_marker_n"


def persuasion_setup_labels(df: pd.DataFrame) -> tuple[str, str]:
    """Human-readable persuader + dataset labels from the rollout CSV."""
    persuader = "unknown persuader"
    if "persuader_model" in df.columns and df["persuader_model"].notna().any():
        persuader = str(df["persuader_model"].dropna().iloc[0]).replace("openai/", "")
    dataset = "FEVER"
    if "dataset" in df.columns and df["dataset"].notna().any():
        dataset = str(df["dataset"].dropna().iloc[0]).upper()
    return f"Persuader: {persuader} · Dataset: {dataset}", dataset


def add_epistemic_marker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate each row from decision_explanation only."""
    out = df.copy()
    expl = out["decision_explanation"].fillna("").astype(str)
    out[MARKER_LIST_COL] = expl.map(lambda t: "|".join(extract_epistemic_markers(t)))
    out[MARKER_COUNT_COL] = expl.map(lambda t: len(extract_epistemic_markers(t)))
    return out


def build_dialogue_epistemic(df: pd.DataFrame) -> pd.DataFrame:
    """One row per dialogue: any log row with ≥1 marker counts as marked."""
    if "dialogue_id" not in df.columns:
        raise ValueError("Expected column dialogue_id")

    g = df.groupby("dialogue_id", sort=True)
    out = (
        g.agg(
            n_log_rows=(MARKER_COUNT_COL, "size"),
            n_log_rows_with_marker=(MARKER_COUNT_COL, lambda s: int((s > 0).sum())),
            has_any_marker=(MARKER_COUNT_COL, lambda s: bool((s > 0).any())),
            max_marker_n_in_one_row=(MARKER_COUNT_COL, "max"),
            mean_marker_n_per_log_row=(MARKER_COUNT_COL, "mean"),
            total_marker_n=(MARKER_COUNT_COL, "sum"),
        )
        .reset_index()
    )
    if "turn" in df.columns:
        first_turn0 = (
            df[df["turn"] == 0]
            .drop_duplicates(subset="dialogue_id", keep="first")
            .set_index("dialogue_id")
        )
        out["marker_n_at_turn0"] = out["dialogue_id"].map(
            first_turn0[MARKER_COUNT_COL].to_dict()
        ).fillna(0).astype(int)
        out["has_marker_at_turn0"] = out["marker_n_at_turn0"] > 0
    return out


def summarize_markers(df: pd.DataFrame, dialogue: pd.DataFrame | None = None) -> pd.DataFrame:
    n = len(df)
    has = df[MARKER_COUNT_COL] > 0
    rows = [
        ("n_log_rows", float(n)),
        ("log_rows_with_any_marker", float(has.sum())),
        ("pct_log_rows_with_any_marker", 100.0 * has.mean() if n else 0.0),
        ("mean_marker_n_per_log_row", float(df[MARKER_COUNT_COL].mean())),
    ]
    if dialogue is not None:
        n_dlg = len(dialogue)
        dlg_has = dialogue["has_any_marker"]
        rows.extend(
            [
                ("n_dialogues", float(n_dlg)),
                ("dialogues_with_any_marker", float(dlg_has.sum())),
                ("pct_dialogues_with_any_marker", 100.0 * dlg_has.mean() if n_dlg else 0.0),
                ("mean_marker_n_per_dialogue_total", float(dialogue["total_marker_n"].mean())),
            ]
        )
        if "has_marker_at_turn0" in dialogue.columns:
            t0 = dialogue["has_marker_at_turn0"]
            rows.extend(
                [
                    ("dialogues_with_marker_at_turn0", float(t0.sum())),
                    ("pct_dialogues_with_marker_at_turn0", 100.0 * t0.mean() if n_dlg else 0.0),
                ]
            )
    # Backward-compatible aliases for older notebooks/scripts
    rows.extend(
        [
            ("n_rows", float(n)),
            ("rows_with_any_marker", float(has.sum())),
            ("pct_rows_with_any_marker", 100.0 * has.mean() if n else 0.0),
            ("mean_marker_n_per_row", float(df[MARKER_COUNT_COL].mean())),
        ]
    )
    return pd.DataFrame(rows, columns=["metric", "value"])


def summarize_markers_by_turn(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for turn in sorted(df["turn"].unique()):
        sub = df[df["turn"] == turn]
        dlg_has = sub.groupby("dialogue_id")[MARKER_COUNT_COL].max() > 0
        rows.append(
            {
                "turn": int(turn),
                "n_log_rows": len(sub),
                "log_rows_with_marker": int((sub[MARKER_COUNT_COL] > 0).sum()),
                "pct_log_rows_with_marker": 100.0 * (sub[MARKER_COUNT_COL] > 0).mean(),
                "n_dialogues": int(len(dlg_has)),
                "dialogues_with_marker": int(dlg_has.sum()),
                "pct_dialogues_with_marker": 100.0 * dlg_has.mean() if len(dlg_has) else 0.0,
                "mean_marker_n_per_log_row": float(sub[MARKER_COUNT_COL].mean()),
                # legacy column names
                "n_rows": len(sub),
                "rows_with_marker": int((sub[MARKER_COUNT_COL] > 0).sum()),
                "pct_rows_with_marker": 100.0 * (sub[MARKER_COUNT_COL] > 0).mean(),
                "mean_marker_n": float(sub[MARKER_COUNT_COL].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_top_markers(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for raw in df.loc[df[MARKER_COUNT_COL] > 0, MARKER_LIST_COL]:
        for m in str(raw).split("|"):
            m = m.strip()
            if m:
                counts[m] = counts.get(m, 0) + 1
    rows = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
    out = pd.DataFrame(rows, columns=["marker", "count"])
    if not out.empty:
        out["polarity"] = out["marker"].map(
            lambda m: "uncertainty"
            if m in UNCERTAINTY_MARKERS
            else ("certainty" if m in CERTAINTY_MARKERS else "other")
        )
    return out


def summarize_marker_count_dist(df: pd.DataFrame) -> pd.DataFrame:
    vc = df[MARKER_COUNT_COL].value_counts().sort_index()
    return pd.DataFrame({"marker_n": vc.index.astype(int), "count": vc.values})


def plot_markers_by_turn(summary_turn: pd.DataFrame, out_png: Path, *, title: str) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4))
    long = summary_turn.melt(
        id_vars=["turn"],
        value_vars=["pct_log_rows_with_marker", "pct_dialogues_with_marker"],
        var_name="unit",
        value_name="pct",
    )
    long["unit"] = long["unit"].map(
        {
            "pct_log_rows_with_marker": "log rows",
            "pct_dialogues_with_marker": "dialogues",
        }
    )
    sns.lineplot(data=long, x="turn", y="pct", hue="unit", marker="o", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Turn")
    ax.set_ylabel("% with ≥1 epistemic marker in decision_explanation")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_epistemic_summary_pack(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    summary_turn: pd.DataFrame,
    top_markers: pd.DataFrame,
    count_dist: pd.DataFrame,
    out_png: Path,
) -> None:
    del summary_turn  # by-turn chart omitted from summary pack
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 1) Top markers
    ax = axes[0]
    if top_markers.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No markers found", ha="center", va="center")
    else:
        palette = {"uncertainty": "#F58518", "certainty": "#54A24B", "other": "#BAB0AC"}
        sns.barplot(
            data=top_markers,
            y="marker",
            x="count",
            hue="polarity",
            ax=ax,
            palette=palette,
            dodge=False,
        )
        ax.set_title("Most frequent epistemic markers")
        ax.set_xlabel("Occurrences in decision_explanation")
        ax.set_ylabel("")
        ax.legend(title="", loc="lower right", fontsize=8)

    # 2) Distribution: how many markers per row
    ax = axes[1]
    sns.barplot(data=count_dist, x="marker_n", y="count", ax=ax, color="#9ECAE9")
    ax.set_title("Markers per log row (0 = none)")
    ax.set_xlabel("Number of markers in one explanation")
    ax.set_ylabel("Log row count")

    # 3) Key numbers
    ax = axes[2]
    ax.axis("off")

    def _metric(name: str, default: float = 0.0) -> float:
        hit = summary.loc[summary["metric"] == name, "value"]
        return float(hit.iloc[0]) if len(hit) else default

    pct_rows = _metric("pct_log_rows_with_any_marker", _metric("pct_rows_with_any_marker"))
    n_marked_rows = int(_metric("log_rows_with_any_marker", _metric("rows_with_any_marker")))
    n_rows = int(_metric("n_log_rows", _metric("n_rows")))
    pct_dlg = _metric("pct_dialogues_with_any_marker")
    n_marked_dlg = int(_metric("dialogues_with_any_marker"))
    n_dlg = int(_metric("n_dialogues"))
    top3 = ", ".join(top_markers["marker"].head(3).tolist()) if not top_markers.empty else "—"
    setup_footer, dataset = persuasion_setup_labels(df)
    txt = (
        f"Log rows: {n_marked_rows}/{n_rows} with ≥1 marker ({pct_rows:.1f}%)\n"
        f"Dialogues: {n_marked_dlg}/{n_dlg} with ≥1 marker ({pct_dlg:.1f}%)\n"
        f"Top markers: {top3}\n\n"
        "Source: target decision_explanation\n"
        f"{setup_footer}"
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=11)
    ax.set_title("Summary")

    persuader_short = setup_footer.split("Persuader: ", 1)[-1].split(" ·", 1)[0]
    fig.suptitle(
        f"Epistemic markers in target decision_explanation — {dataset} · {persuader_short} persuader",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract epistemic markers from decision_explanation (target model)."
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    enriched = add_epistemic_marker_columns(df)
    dialogue = build_dialogue_epistemic(enriched)
    summary = summarize_markers(enriched, dialogue)
    summary_turn = summarize_markers_by_turn(enriched)
    top_markers = summarize_top_markers(enriched)
    count_dist = summarize_marker_count_dist(enriched)

    csv_dir = args.out_dir / "csv"
    png_dir = args.out_dir / "png"
    csv_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    out_enriched = csv_dir / "epistemic.csv"
    out_dialogues = csv_dir / "epistemic_dialogues.csv"
    out_summary = csv_dir / "epistemic_summary.csv"
    out_by_turn = csv_dir / "epistemic_by_turn.csv"
    out_top = csv_dir / "epistemic_top_markers.csv"
    out_png_turn = png_dir / "epistemic_by_turn.png"
    out_png_pack = png_dir / "epistemic_summary_pack.png"

    enriched.to_csv(out_enriched, index=False)
    dialogue.to_csv(out_dialogues, index=False)
    summary.to_csv(out_summary, index=False)
    summary_turn.to_csv(out_by_turn, index=False)
    top_markers.to_csv(out_top, index=False)
    plot_markers_by_turn(
        summary_turn,
        out_png_turn,
        title="Epistemic markers in target decision_explanation (by turn)",
    )
    plot_epistemic_summary_pack(
        enriched, summary, summary_turn, top_markers, count_dist, out_png_pack
    )

    print(f"Source column: decision_explanation")
    print(f"Added columns: {MARKER_LIST_COL}, {MARKER_COUNT_COL}")
    print(f"Saved enriched CSV: {out_enriched}")
    print(f"Saved dialogue table: {out_dialogues}")
    print(f"Saved summary: {out_summary}")
    print(f"Saved by-turn: {out_by_turn}")
    print(f"Saved top markers: {out_top}")
    print(f"Saved figure (by turn): {out_png_turn}")
    print(f"Saved figure (summary pack): {out_png_pack}")
    print("\nSummary:")
    for _, r in summary.iterrows():
        print(f"  {r['metric']}: {r['value']}")


if __name__ == "__main__":
    main()
