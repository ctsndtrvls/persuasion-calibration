"""
Build reliability (ECE-style) figures from token-probability CSVs under
output_wood/token_prob_confidence/<provider>/{csv,png}/.

Layout:
  - openai/csv/*.csv, openai/png/*.png
  - deepseek/csv/*.csv, deepseek/png/*.png
  - qwen-gemma/csv/*.csv, qwen-gemma/png/*.png, qwen-gemma/logs/
  - debateqa/csv/debateqa_tokenprob_four_models*.csv, debateqa/png/*.png

Modes:
  - OpenAI / DeepSeek: fever + ConflictQA + optional DebateQA (standalone CSV or
    debateqa_tokenprob_openai_deepseek_temp*.csv under debateqa/csv/ or
    temperature_experiments/debateqa/csv/).
  - qwen-gemma: CSV in qwen-gemma/csv/ — FEVER, PopQA, DebateQA when present.
  - debateqa: one calibration figure for DebateQA token-prob across OpenAI,
    DeepSeek, Qwen, Gemma (requires debateqa/csv/debateqa_tokenprob_four_models*.csv;
    see scripts/run_tokenprob_debateqa_four_models.sh).

Run:

  ./scripts/plot_token_prob_figures.sh --which all

or:

  MPLBACKEND=Agg python3 src/plot_token_prob_confidence_figures.py --which all
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_ece_calibration as pec

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


def _csv_dir(token_prob_dir: Path, provider: str) -> Path:
    return token_prob_dir / provider / "csv"


def _png_dir(token_prob_dir: Path, provider: str) -> Path:
    return token_prob_dir / provider / "png"


def _normalize_calibration_df(df: pd.DataFrame) -> pd.DataFrame:
    need = {"model", "dataset", "confidence", "correct"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {missing}; have {list(df.columns)}")
    out = df.copy()
    out["model"] = out["model"].astype(str)
    out["dataset"] = out["dataset"].astype(str)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["correct"] = pd.to_numeric(out["correct"], errors="coerce")
    out = out.dropna(subset=["confidence", "correct"])
    out["correct"] = out["correct"].astype(int)
    out = out[(out["confidence"] >= 0) & (out["confidence"] <= 1)]
    return out


def _load_openai_token_prob(token_prob_dir: Path, repo: Path) -> pd.DataFrame:
    base = _csv_dir(token_prob_dir, "openai")
    parts: list[pd.DataFrame] = []
    mapping: list[tuple[str, Path]] = [
        ("fever", base / "fever_ece_openai_token_prob.csv"),
        ("conflictqa_popqa", base / "conflictqa_ece_openai_token_prob_popqa.csv"),
        ("conflictqa_strategyqa", base / "conflictqa_ece_openai_token_prob_strategyqa.csv"),
        ("debateqa", base / "debateqa_ece_openai_token_prob.csv"),
    ]
    for ds_hint, path in mapping:
        if not path.exists():
            continue
        d = pd.read_csv(path)
        parts.append(_normalize_calibration_df(d.assign(dataset=d["dataset"].fillna(ds_hint).astype(str))))

    extra = _debateqa_tokenprob_rows_from_combined(repo, "openai", token_prob_dir)
    if extra is not None and not extra.empty:
        parts.append(extra)

    if not parts:
        raise FileNotFoundError(f"No OpenAI token-prob CSVs under {base}")
    return _normalize_calibration_df(pd.concat(parts, ignore_index=True))


def _debateqa_tokenprob_rows_from_combined(
    repo: Path, which: Literal["openai", "deepseek"], token_prob_dir: Path
) -> pd.DataFrame | None:
    """Rows from debateqa_tokenprob_openai_deepseek_temp*.csv (local or legacy temp-exp dir)."""
    candidates: list[Path] = []
    local = token_prob_dir / "debateqa" / "csv"
    if local.is_dir():
        candidates.extend(sorted(local.glob("debateqa_tokenprob_openai_deepseek_temp*.csv")))
    ddir = repo / "output_wood" / "temperature_experiments" / "debateqa" / "csv"
    if ddir.is_dir():
        candidates.extend(sorted(ddir.glob("debateqa_tokenprob_openai_deepseek_temp*.csv")))
    if not candidates:
        return None
    path = candidates[0]
    for p in candidates:
        if "temp0p0" in p.name or "temp0p00" in p.name:
            path = p
            break
    df = pd.read_csv(path)
    if "confidence" not in df.columns or "correct" not in df.columns:
        return None
    df = df.copy()
    df["model"] = df["model"].astype(str)
    if which == "openai":
        sub = df[df["model"].astype(str).str.startswith("openai/")]
        sub = sub.assign(dataset="debateqa")
    else:
        m = df["model"].astype(str).str.lower()
        sub = df[m.str.contains("deepseek") & ~m.str.contains("openai")]
        sub = sub.assign(model="deepseek/deepseek-chat", dataset="debateqa")
    if sub.empty:
        return None
    return _normalize_calibration_df(sub[["model", "dataset", "confidence", "correct"]])


def _dataset_from_deepseek_filename(stem: str) -> str:
    s = stem.lower()
    if "debateqa" in s:
        return "debateqa"
    if "fever" in s:
        return "fever"
    if "strategyqa" in s:
        return "conflictqa_strategyqa"
    if "popqa" in s:
        return "conflictqa_popqa"
    return "unknown"


def _load_deepseek_token_prob(token_prob_dir: Path, repo: Path) -> pd.DataFrame:
    base = _csv_dir(token_prob_dir, "deepseek")
    if not base.is_dir():
        raise FileNotFoundError(f"Missing directory: {base}")
    paths = sorted(base.glob("deepseek_tokenprob_*.csv"))
    parts: list[pd.DataFrame] = []
    for path in paths:
        d = pd.read_csv(path)
        if "confidence" not in d.columns or "correct" not in d.columns:
            continue
        ds = _dataset_from_deepseek_filename(path.stem)
        parts.append(
            pd.DataFrame(
                {
                    "model": "deepseek/deepseek-chat",
                    "dataset": ds,
                    "confidence": d["confidence"],
                    "correct": d["correct"],
                }
            )
        )
    extra = _debateqa_tokenprob_rows_from_combined(repo, "deepseek", token_prob_dir)
    if extra is not None and not extra.empty:
        parts.append(extra)
    if not parts:
        raise FileNotFoundError(f"No deepseek_tokenprob_*.csv under {base}")
    return _normalize_calibration_df(pd.concat(parts, ignore_index=True))


def _load_qwen_gemma_token_prob(token_prob_dir: Path, csv_name: str = "tokenprob_new_models_v2.csv") -> pd.DataFrame:
    path = _csv_dir(token_prob_dir, "qwen-gemma") / csv_name
    if not path.exists():
        raise FileNotFoundError(path)
    d = pd.read_csv(path)
    return _normalize_calibration_df(d)


def _ensure_deepseek_label() -> None:
    pec.MODEL_LABELS.setdefault("deepseek/deepseek-chat", "DeepSeek Chat")


def _models_in_plot_order(df: pd.DataFrame) -> list[str]:
    seen = set(df["model"].unique())
    ordered = [m for m in pec.MODEL_IDS if m in seen]
    for m in sorted(seen):
        if m not in ordered:
            ordered.append(m)
    return ordered


def _debateqa_four_model_order() -> list[str]:
    """Preferred bar order for the four-model DebateQA token-prob figure."""
    return [
        "openai/gpt-4o-2024-11-20",
        "deepseek/deepseek-chat-v2.5",
        "deepseek/deepseek-chat",
        "qwen/qwen3-14b",
        "google/gemma-4-26b-a4b-it",
    ]


def _write_debateqa_tokenprob_four_models(token_prob_dir: Path) -> None:
    csv_dir = _csv_dir(token_prob_dir, "debateqa")
    png_dir = _png_dir(token_prob_dir, "debateqa")
    png_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(csv_dir.glob("debateqa_tokenprob_four_models*.csv"))
    if not paths:
        print(
            "Skipping DebateQA four-model token-prob figure: no "
            f"debateqa_tokenprob_four_models*.csv in {csv_dir}. "
            "Run: ./scripts/run_tokenprob_debateqa_four_models.sh",
            flush=True,
        )
        return
    path = paths[0]
    for p in paths:
        if "temp0p0" in p.name or "temp0p00" in p.name:
            path = p
            break
    df = _normalize_calibration_df(pd.read_csv(path))
    df = df[df["dataset"].astype(str) == "debateqa"].copy()
    if df.empty:
        print(f"Skipping DebateQA four-model plot: no debateqa rows in {path}", flush=True)
        return
    order = [m for m in _debateqa_four_model_order() if m in set(df["model"].unique())]
    for m in df["model"].unique():
        if m not in order:
            order.append(str(m))
    model_labels_order = [pec._label(m) for m in order]
    df = df[df["model"].isin(order)].copy()
    df["_model_lab"] = df["model"].map(lambda x: pec._label(x))

    edges = pec.bin_edges()
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    pec.plot_grouped_calibration(
        ax,
        df,
        edges,
        "_model_lab",
        model_labels_order,
        title="DebateQA — token probability calibration (four models)",
    )
    out = png_dir / "ece_debateqa_tokenprob_four_models.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("Saved:", out)


def _write_figures(
    df: pd.DataFrame,
    png_dir: Path,
    file_suffix: str,
    *,
    ds_order: list[str],
    ds_labels: dict[str, str],
) -> None:
    png_dir.mkdir(parents=True, exist_ok=True)
    edges = pec.bin_edges()

    models_in_data = _models_in_plot_order(df)
    model_labels_order = [pec._label(m) for m in models_in_data]

    df = df.copy()
    df["_model_lab"] = df["model"].map(lambda x: pec._label(x))

    ds_present = [d for d in ds_order if d in set(df["dataset"].unique())]
    if not ds_present:
        ds_present = sorted(df["dataset"].unique())

    suffix = file_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    n_panels = 1 + len(ds_present)
    fig_w = min(22.0, 3.6 * n_panels + 1.5)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4.8), constrained_layout=True)
    if n_panels == 1:
        axes = np.array([axes])
    pec.plot_grouped_calibration(
        axes[0],
        df,
        edges,
        "_model_lab",
        model_labels_order,
        title="(a) All datasets pooled — by model (token prob)",
    )
    letters = "bcdefghij"
    for i, ds in enumerate(ds_present):
        sub = df[df["dataset"] == ds]
        title_ds = ds_labels.get(ds, ds)
        pec.plot_grouped_calibration(
            axes[i + 1],
            sub,
            edges,
            "_model_lab",
            model_labels_order,
            title=f"({letters[i]}) Dataset: {title_ds}",
        )
    tag = f"{len(ds_present)}ds" if len(ds_present) != 3 else "3ds"
    out3 = png_dir / f"ece_panels_1x{n_panels}_models_token_prob{suffix}_{tag}.png"
    fig.savefig(out3, dpi=200)
    plt.close(fig)
    print("Saved:", out3)

    ds_legend_order = [ds_labels.get(d, d) for d in ds_present]
    df["_ds_lab"] = df["dataset"].map(lambda d: ds_labels.get(d, d))

    for m in models_in_data:
        sub = df[df["model"] == m]
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        pec.plot_grouped_calibration(
            ax,
            sub,
            edges,
            "_ds_lab",
            ds_legend_order,
            title=f"Calibration — {pec._label(m)} (token prob)",
        )
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", m.replace("/", "_").replace(".", "_"))
        outp = png_dir / f"ece_per_model_{safe}_token_prob{suffix}.png"
        fig.savefig(outp, dpi=200)
        plt.close(fig)
        print("Saved:", outp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Root of persuasion-calibration (must contain output_wood/).",
    )
    parser.add_argument(
        "--which",
        nargs="+",
        choices=("openai", "deepseek", "qwen-gemma", "debateqa", "both", "all"),
        default=["all"],
        help="Which figure bundles to build. 'both' = openai+deepseek.",
    )
    parser.add_argument(
        "--qwen-gemma-csv",
        type=str,
        default="tokenprob_new_models_v2.csv",
        help="Filename under output_wood/token_prob_confidence/qwen-gemma/csv/.",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    token_prob_dir = repo / "output_wood" / "token_prob_confidence"

    which: set[str] = set()
    for w in args.which:
        if w == "both":
            which.update(("openai", "deepseek"))
        elif w == "all":
            which.update(("openai", "deepseek", "qwen-gemma", "debateqa"))
        else:
            which.add(w)

    ds_order_openai = ["fever", "conflictqa_popqa", "conflictqa_strategyqa", "debateqa"]
    ds_labels_common = {
        "fever": "FEVER",
        "conflictqa_popqa": "ConflictQA popQA",
        "conflictqa_strategyqa": "ConflictQA strategyQA",
        "debateqa": "DebateQA",
    }

    if "openai" in which:
        df_o = _load_openai_token_prob(token_prob_dir, repo)
        _write_figures(
            df_o,
            _png_dir(token_prob_dir, "openai"),
            "_openai",
            ds_order=ds_order_openai,
            ds_labels=ds_labels_common,
        )

    if "deepseek" in which:
        _ensure_deepseek_label()
        df_d = _load_deepseek_token_prob(token_prob_dir, repo)
        _write_figures(
            df_d,
            _png_dir(token_prob_dir, "deepseek"),
            "_deepseek",
            ds_order=ds_order_openai,
            ds_labels=ds_labels_common,
        )

    if "qwen-gemma" in which:
        df_q = _load_qwen_gemma_token_prob(token_prob_dir, args.qwen_gemma_csv)
        order_q = ["fever", "conflictqa_popqa", "debateqa"]
        labels_q = {
            "fever": "FEVER",
            "conflictqa_popqa": "PopQA (ConflictQA subset)",
            "debateqa": "DebateQA",
        }
        sub = df_q[df_q["dataset"].isin(order_q)].copy()
        if sub.empty:
            raise ValueError("No rows for fever / conflictqa_popqa / debateqa in Qwen+Gemma CSV.")
        _write_figures(
            sub,
            _png_dir(token_prob_dir, "qwen-gemma"),
            "_qwen_gemma_fpd",
            ds_order=order_q,
            ds_labels=labels_q,
        )

    if "debateqa" in which:
        _write_debateqa_tokenprob_four_models(token_prob_dir)


if __name__ == "__main__":
    main()
