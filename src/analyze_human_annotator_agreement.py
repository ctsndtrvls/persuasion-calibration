"""
Convert mixed human-annotator submissions (xlsx/pdf) to CSV and compute inter-rater agreement.

Metrics (per dimension):
  - Krippendorff's alpha (ordinal) across all annotators
  - Quadratic weighted Cohen's kappa for each annotator pair

Thresholds (supervisor):
  - alpha >= 0.5
  - pairwise kappa >= 0.65

Example:
  python3 analyze_human_annotator_agreement.py \\
    --input-dir ../output_wood/persuasion/human_annotators
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
from sklearn.metrics import cohen_kappa_score

from persuasion_arg_quality import TOP_LEVEL_QUALITY_DIMENSIONS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORE_COLUMNS = TOP_LEVEL_QUALITY_DIMENSIONS
CONTEXT_COLUMNS = (
    "item_id",
    "dataset",
    "claim",
    "target_answer_before",
    "counterargument",
    "target_answer_after",
)
ALPHA_THRESHOLD = 0.5
KAPPA_THRESHOLD = 0.65


def _clean_excel_annotator(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "item_id" in out.columns:
        out = out[out["item_id"].notna()].copy()
        out["item_id"] = out["item_id"].astype(int)
    for col in SCORE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    keep = [c for c in CONTEXT_COLUMNS if c in out.columns] + list(SCORE_COLUMNS)
    extra = [c for c in out.columns if c not in keep and c not in ("Comments",)]
    return out[keep + extra].sort_values("item_id").reset_index(drop=True)


def load_annotator_excel(path: Path) -> pd.DataFrame:
    return _clean_excel_annotator(pd.read_excel(path))


def load_annotator_pdf(path: Path) -> pd.DataFrame:
    pdf = pdfium.PdfDocument(str(path))
    text = "".join(page.get_textpage().get_text_range() + "\n" for page in pdf)
    rows: list[dict[str, int | str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not re.match(r"^\d+\s", line):
            continue
        id_match = re.match(r"^(\d+)\s+(\S+)", line)
        score_match = re.search(r"(\d)\s+(\d)\s+(\d)\s*$", line)
        if not id_match or not score_match:
            continue
        rows.append(
            {
                "item_id": int(id_match.group(1)),
                "dataset": id_match.group(2),
                "cogency": int(score_match.group(1)),
                "effectiveness": int(score_match.group(2)),
                "reasonableness": int(score_match.group(3)),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) != 100:
        raise ValueError(f"{path.name}: expected 100 scored rows, found {len(df)}")
    return df.sort_values("item_id").reset_index(drop=True)


def load_annotator_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _clean_excel_annotator(df)


def detect_submission(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return load_annotator_excel(path)
    if suffix == ".pdf":
        return load_annotator_pdf(path)
    if suffix == ".csv":
        return load_annotator_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def enrich_pdf_with_context(scores: pd.DataFrame, context_path: Path | None) -> pd.DataFrame:
    if context_path is None or not context_path.is_file():
        return scores
    ctx = pd.read_csv(context_path)
    cols = [c for c in CONTEXT_COLUMNS if c in ctx.columns]
    merged = ctx[cols].merge(scores[["item_id"] + list(SCORE_COLUMNS)], on="item_id", how="right")
    return merged.sort_values("item_id").reset_index(drop=True)


def convert_submissions(
    paths: list[Path],
    out_dir: Path,
    *,
    context_path: Path | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for i, path in enumerate(paths, start=1):
        df = detect_submission(path)
        if set(SCORE_COLUMNS) - set(df.columns):
            raise ValueError(f"{path.name}: missing score columns {SCORE_COLUMNS}")
        if path.suffix.lower() == ".pdf":
            df = enrich_pdf_with_context(df, context_path)
        out_path = out_dir / f"human_annotation_annotator_{i}.csv"
        df.to_csv(out_path, index=False)
        out_paths.append(out_path)
        print(f"Wrote {out_path} ({len(df)} rows)")
    return out_paths


def quadratic_kappa(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return float("nan")
    return float(
        cohen_kappa_score(
            a[mask].astype(int),
            b[mask].astype(int),
            weights="quadratic",
        )
    )


def krippendorff_alpha(mat: np.ndarray) -> float:
    """mat shape: (n_annotators, n_items)."""
    if mat.shape[1] < 2:
        return float("nan")
    return float(krippendorff.alpha(reliability_data=mat, level_of_measurement="ordinal"))


def agreement_report(paths: list[Path]) -> pd.DataFrame:
    annotators = [load_annotator_csv(p) if p.suffix == ".csv" else detect_submission(p) for p in paths]
    names = [f"annotator_{i}" for i in range(1, len(annotators) + 1)]

    base = annotators[0][["item_id"]].copy()
    for name, df in zip(names, annotators):
        for dim in SCORE_COLUMNS:
            if dim not in df.columns:
                raise ValueError(f"{name}: missing column {dim}")
            base[f"{name}__{dim}"] = base["item_id"].map(
                df.set_index("item_id")[dim].astype(float)
            )

    rows: list[dict[str, object]] = []
    for dim in SCORE_COLUMNS:
        cols = [f"{name}__{dim}" for name in names]
        mat = base[cols].to_numpy(dtype=float).T
        alpha = krippendorff_alpha(mat)
        rows.append(
            {
                "dimension": dim,
                "metric": "krippendorff_alpha",
                "annotator_a": "all",
                "annotator_b": "all",
                "n_items": int(np.sum(~np.isnan(mat).all(axis=0))),
                "value": alpha,
                "threshold": ALPHA_THRESHOLD,
                "passes": alpha >= ALPHA_THRESHOLD if not np.isnan(alpha) else False,
            }
        )
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                kappa = quadratic_kappa(mat[i], mat[j])
                rows.append(
                    {
                        "dimension": dim,
                        "metric": "quadratic_kappa",
                        "annotator_a": names[i],
                        "annotator_b": names[j],
                        "n_items": int(np.sum(~(np.isnan(mat[i]) | np.isnan(mat[j])))),
                        "value": kappa,
                        "threshold": KAPPA_THRESHOLD,
                        "passes": kappa >= KAPPA_THRESHOLD if not np.isnan(kappa) else False,
                    }
                )
    return pd.DataFrame(rows)


def discover_default_inputs(input_dir: Path) -> list[Path]:
    candidates = [
        input_dir / "human_annotation_1.xlsx",
        input_dir / "human_annotation-2.csv.xlsx",
        input_dir / "human_annotation_3.pdf",
    ]
    missing = [p for p in candidates if not p.is_file()]
    if missing:
        csvs = sorted(input_dir.glob("human_annotation_annotator_*.csv"))
        if len(csvs) >= 3:
            return csvs[:3]
        raise FileNotFoundError("Missing inputs: " + ", ".join(p.name for p in missing))
    return candidates


def print_summary(report: pd.DataFrame) -> None:
    print("\nInter-annotator agreement")
    print("=" * 72)
    for dim in SCORE_COLUMNS:
        sub = report[report["dimension"] == dim]
        alpha_row = sub[sub["metric"] == "krippendorff_alpha"].iloc[0]
        flag = "PASS" if alpha_row["passes"] else "FAIL"
        print(f"\n{dim}")
        print(f"  Krippendorff alpha = {alpha_row['value']:.3f}  (threshold >= {ALPHA_THRESHOLD})  [{flag}]")
        for _, row in sub[sub["metric"] == "quadratic_kappa"].iterrows():
            flag = "PASS" if row["passes"] else "FAIL"
            print(
                f"  {row['annotator_a']} vs {row['annotator_b']}: "
                f"kappa = {row['value']:.3f}  (threshold >= {KAPPA_THRESHOLD})  [{flag}]"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert human annotator files and compute agreement.")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "output_wood" / "persuasion" / "human_annotators",
        help="Directory with raw xlsx/pdf or converted CSV files.",
    )
    ap.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit paths to three annotator files (overrides auto-discovery).",
    )
    ap.add_argument(
        "--context",
        type=Path,
        default=PROJECT_ROOT
        / "output_wood"
        / "persuasion"
        / "DeepSeek"
        / "mixed"
        / "human_annotation"
        / "human_annotation.csv",
        help="Template CSV used to add context columns to PDF-derived scores.",
    )
    ap.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert submissions to human_annotation_annotator_*.csv.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Agreement report CSV path (default: <input-dir>/human_annotator_agreement.csv).",
    )
    args = ap.parse_args()

    input_dir = args.input_dir
    raw_paths = args.inputs or discover_default_inputs(input_dir)

    converted = convert_submissions(
        raw_paths,
        input_dir,
        context_path=args.context if args.context.is_file() else None,
    )
    if args.convert_only:
        return

    report = agreement_report(converted)
    out = args.out or (input_dir / "human_annotator_agreement.csv")
    report.to_csv(out, index=False)
    print_summary(report)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
