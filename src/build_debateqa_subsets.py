from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "debateqa" / "dataset"
OUT_DIR = PROJECT_ROOT / "output_wood" / "dataset_subsampling" / "debateqa" / "csv"
SEED = 42


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "dataset", "items", "records", "questions"):
            val = obj.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def _normalize_partial_answers(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            cand = (
                item.get("partial_answer")
                or item.get("answer")
                or item.get("point_of_view")
                or item.get("text")
                or item.get("perspective")
                or item.get("explanation")
            )
            t = _as_text(cand)
            if t:
                out.append(t)
        else:
            t = _as_text(item)
            if t:
                out.append(t)
    return out


def load_debateqa(path: Path) -> list[dict[str, Any]]:
    records = _read_records(path)
    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        question = _as_text(rec.get("question") or rec.get("query") or rec.get("prompt"))
        partial_answers = _normalize_partial_answers(
            rec.get("partial_answers") or rec.get("answers") or rec.get("perspectives")
        )
        if not question or not partial_answers:
            continue
        rows.append(
            {
                "question": question,
                "ground_truth": partial_answers,
                "partial_answers": partial_answers,
                "perspective_count": len(partial_answers),
                "dataset": "debateqa",
                "original_index": i,
            }
        )
    if not rows:
        raise ValueError(f"No usable DebateQA rows found in {path}")
    return rows


def assign_quantile_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Assign 3 quantile-like bins by sorted perspective_count:
    first third -> 1, second third -> 2, last third -> 3.
    """
    ordered = sorted(rows, key=lambda r: (int(r["perspective_count"]), int(r["original_index"])))
    n = len(ordered)
    c1 = n // 3
    c2 = (2 * n) // 3
    for idx, row in enumerate(ordered):
        if idx < c1:
            row["complexity_level"] = 1
        elif idx < c2:
            row["complexity_level"] = 2
        else:
            row["complexity_level"] = 3
    return ordered


def stratified_sample(rows: list[dict[str, Any]], n_per_level: int, seed: int = SEED) -> list[dict[str, Any]]:
    binned = assign_quantile_bins(rows)
    by_level: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    for row in binned:
        by_level[int(row["complexity_level"])].append(row)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for level in (1, 2, 3):
        group = by_level[level]
        if len(group) < n_per_level:
            raise ValueError(
                f"Not enough rows in complexity_level={level}: need {n_per_level}, found {len(group)}"
            )
        sampled.extend(rng.sample(group, n_per_level))
    rng.shuffle(sampled)
    return sampled


def save_subset(rows: list[dict[str, Any]], n_per_level: int, stem: str) -> Path:
    subset = stratified_sample(rows, n_per_level=n_per_level, seed=SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{stem}_{n_per_level * 3}_{n_per_level}x3.csv"
    fieldnames = [
        "question",
        "ground_truth",
        "partial_answers",
        "perspective_count",
        "dataset",
        "original_index",
        "complexity_level",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in subset:
            out_row = dict(row)
            out_row["ground_truth"] = json.dumps(out_row["ground_truth"], ensure_ascii=False)
            out_row["partial_answers"] = json.dumps(out_row["partial_answers"], ensure_ascii=False)
            writer.writerow(out_row)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DebateQA stratified subsets.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional DebateQA file path. If omitted, first JSON/JSONL in data/debateqa/dataset is used.",
    )
    args = parser.parse_args()

    if args.input_file is not None:
        input_path = args.input_file
    else:
        candidates = sorted(DATA_DIR.glob("*.json*"))
        if not candidates:
            raise FileNotFoundError(f"No DebateQA JSON/JSONL files found in {DATA_DIR}")
        input_path = candidates[0]

    rows = load_debateqa(input_path)
    out_200 = save_subset(rows, n_per_level=200, stem="debateqa")
    out_160 = save_subset(rows, n_per_level=160, stem="debateqa")

    print(f"Input rows loaded: {len(rows)} from {input_path}")
    print("Saved:")
    print(f" - {out_200}")
    print(f" - {out_160}")


if __name__ == "__main__":
    main()
