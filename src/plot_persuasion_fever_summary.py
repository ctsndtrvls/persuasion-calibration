from __future__ import annotations

import argparse
import os
from pathlib import Path

_PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT_BOOTSTRAP / ".mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEVER_DEEPSEEK_DIR = PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"
DEFAULT_INPUT = FEVER_DEEPSEEK_DIR / "csv" / "expl.csv"
DEFAULT_OUT_DIR = FEVER_DEEPSEEK_DIR
LABEL_ORDER = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def normalize_label(x: object) -> str:
    s = str(x or "").strip().upper()
    if "NOT ENOUGH" in s or s == "NEI":
        return "NOT ENOUGH INFO"
    if "REFUTE" in s:
        return "REFUTES"
    if "SUPPORT" in s:
        return "SUPPORTS"
    return s


def prepare_dialogue_view(df: pd.DataFrame) -> pd.DataFrame:
    """Per dialogue: compare turn 0 (baseline) vs final turn after persuasion."""
    # Continue runs may append new segments that restart at turn 0; keep the first baseline.
    t0 = (
        df[df["turn"] == 0][["dialogue_id", "gold_label", "answer", "confidence"]]
        .drop_duplicates(subset="dialogue_id", keep="first")
        .copy()
    )
    final = (
        df.sort_values(["dialogue_id", "turn"])
        .groupby("dialogue_id", as_index=False)
        .tail(1)[
            [
                "dialogue_id",
                "answer",
                "confidence",
                "turn",
                "flipped_from_initial",
                "flip_turn",
                "stop_reason",
            ]
        ]
    )
    t0 = t0.rename(columns={"answer": "answer_t0", "confidence": "conf_t0"})
    final = final.rename(
        columns={
            "answer": "answer_final",
            "confidence": "conf_final",
            "turn": "final_turn",
        }
    )
    out = t0.merge(final, on="dialogue_id", how="inner")

    for c in ("gold_label", "answer_t0", "answer_final"):
        out[c] = out[c].map(normalize_label)

    out["flip"] = out["answer_t0"] != out["answer_final"]
    out["correct_t0"] = out["answer_t0"] == out["gold_label"]
    out["correct_final"] = out["answer_final"] == out["gold_label"]
    out["conf_delta"] = out["conf_final"] - out["conf_t0"]
    out["persuasion_turns"] = out["final_turn"].astype(int)
    flip_turn_num = pd.to_numeric(out["flip_turn"], errors="coerce")
    out["flip_turn_num"] = flip_turn_num
    return out


def write_summary_csv(view: pd.DataFrame, out_csv: Path) -> None:
    flipped = view[view["flip"]]
    rows = [
        ("n_dialogues", float(len(view))),
        ("flip_rate_pct", float(view["flip"].mean() * 100.0)),
        ("accuracy_t0_pct", float(view["correct_t0"].mean() * 100.0)),
        ("accuracy_final_pct", float(view["correct_final"].mean() * 100.0)),
        (
            "accuracy_delta_pp",
            float((view["correct_final"].mean() - view["correct_t0"].mean()) * 100.0),
        ),
        ("mean_conf_t0", float(view["conf_t0"].mean())),
        ("mean_conf_final", float(view["conf_final"].mean())),
        ("mean_conf_delta", float(view["conf_delta"].mean())),
        ("mean_persuasion_turns", float(view["persuasion_turns"].mean())),
        ("median_flip_turn", float(flipped["flip_turn_num"].median()) if len(flipped) else float("nan")),
        ("stopped_flipped", float((view["stop_reason"] == "flipped").sum())),
        ("stopped_max_turns", float((view["stop_reason"] == "max_turns").sum())),
    ]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(out_csv, index=False)


def plot_figure_pack(view: pd.DataFrame, out_png: Path, *, title: str) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 1) Overall accuracy: turn 0 vs final
    acc_df = pd.DataFrame(
        {
            "stage": ["Before persuasion (turn 0)", "After persuasion (final)"],
            "accuracy": [view["correct_t0"].mean(), view["correct_final"].mean()],
        }
    )
    sns.barplot(data=acc_df, x="stage", y="accuracy", ax=axes[0, 0], color="#4C78A8")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Overall Accuracy: Before vs After")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].tick_params(axis="x", rotation=12)

    # 2) Accuracy by gold label
    cls_rows = []
    for lbl in LABEL_ORDER:
        sub = view[view["gold_label"] == lbl]
        if sub.empty:
            continue
        cls_rows.append((lbl, "Before", sub["correct_t0"].mean()))
        cls_rows.append((lbl, "After", sub["correct_final"].mean()))
    cls_df = pd.DataFrame(cls_rows, columns=["gold_label", "stage", "accuracy"])
    sns.barplot(data=cls_df, x="gold_label", y="accuracy", hue="stage", ax=axes[0, 1])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Accuracy by FEVER Label")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].legend(title="")

    # 3) Flip rate by gold label (initial -> final)
    flip_cls = (
        view.groupby("gold_label", as_index=False)["flip"]
        .mean()
        .assign(flip_rate=lambda d: d["flip"] * 100.0)
    )
    flip_cls["gold_label"] = pd.Categorical(flip_cls["gold_label"], categories=LABEL_ORDER, ordered=True)
    flip_cls = flip_cls.sort_values("gold_label")
    sns.barplot(data=flip_cls, x="gold_label", y="flip_rate", ax=axes[0, 2], color="#F58518")
    axes[0, 2].set_title("Flip Rate by FEVER Label")
    axes[0, 2].set_xlabel("")
    axes[0, 2].set_ylabel("Flip rate (%)")
    axes[0, 2].tick_params(axis="x", rotation=20)

    # 4) Persuasion turns used (final turn index)
    sns.histplot(view["persuasion_turns"], bins=range(1, 17), ax=axes[1, 0], color="#54A24B")
    axes[1, 0].set_title("Persuasion Turns Used (final turn)")
    axes[1, 0].set_xlabel("Turn")
    axes[1, 0].set_ylabel("Count")

    # 5) Stop reason
    stop_df = view["stop_reason"].value_counts().reset_index()
    stop_df.columns = ["stop_reason", "count"]
    sns.barplot(data=stop_df, x="stop_reason", y="count", ax=axes[1, 1], color="#B279A2")
    axes[1, 1].set_title("Stop Reason")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Count")

    # 6) Key numbers
    axes[1, 2].axis("off")
    acc0 = view["correct_t0"].mean() * 100.0
    accf = view["correct_final"].mean() * 100.0
    flip = view["flip"].mean() * 100.0
    med_flip = view.loc[view["flip"], "flip_turn_num"].median()
    txt = (
        f"N dialogues: {len(view)}\n"
        f"Flip rate: {flip:.1f}%\n"
        f"Accuracy turn 0: {acc0:.1f}%\n"
        f"Accuracy final: {accf:.1f}%\n"
        f"Accuracy delta: {accf - acc0:+.1f} pp\n"
        f"Mean conf delta: {view['conf_delta'].mean():+.3f}\n"
        f"Mean turns: {view['persuasion_turns'].mean():.1f}\n"
        f"Median flip turn: {med_flip:.0f}" if view["flip"].any() else "Median flip turn: n/a"
    )
    axes[1, 2].text(0.02, 0.98, txt, va="top", ha="left", fontsize=10)
    axes[1, 2].set_title("Key Numbers")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot FEVER persuasion summary figure pack.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--title",
        default=(
            "Persuasion Summary: FEVER (no flip @ turn 1), "
            "continue until flip (max 15), DeepSeek <- GPT-5.4-mini"
        ),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    view = prepare_dialogue_view(df)

    png_dir = args.out_dir / "png"
    csv_dir = args.out_dir / "csv"
    png_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    out_png = png_dir / "summary.png"
    out_csv = csv_dir / "summary_metrics.csv"

    plot_figure_pack(view, out_png, title=args.title)
    write_summary_csv(view, out_csv)

    print(f"Saved figure: {out_png}")
    print(f"Saved summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
