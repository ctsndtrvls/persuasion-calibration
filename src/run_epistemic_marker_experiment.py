from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence

from linguistic_confidence import NO_MARKER, primary_epistemic_marker


@dataclass(frozen=True)
class Row:
    model: str
    dataset: str
    original_index: int
    answer: str
    correct: int


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x: object, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def load_rows(paths: Sequence[Path]) -> list[Row]:
    rows: list[Row] = []
    seen: set[tuple[str, str, int]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                model = str(rec.get("model", "")).strip()
                dataset = str(rec.get("dataset", "")).strip()
                oid = _safe_int(rec.get("original_index"), -1)
                if not model or not dataset or oid < 0:
                    continue
                key = (model, dataset, oid)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    Row(
                        model=model,
                        dataset=dataset,
                        original_index=oid,
                        answer=str(rec.get("answer", "")),
                        correct=1 if _safe_float(rec.get("correct"), 0.0) >= 1 else 0,
                    )
                )
    return rows


def split_train_test(rows: Sequence[Row], train_ratio: float, seed: int) -> tuple[list[Row], list[Row]]:
    grouped: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in rows:
        grouped[(r.model, r.dataset)].append(r)

    train: list[Row] = []
    test: list[Row] = []
    rng = random.Random(seed)
    for key, block in grouped.items():
        block_copy = list(block)
        rng.shuffle(block_copy)
        cut = max(1, int(len(block_copy) * train_ratio))
        cut = min(cut, len(block_copy) - 1) if len(block_copy) > 1 else len(block_copy)
        train.extend(block_copy[:cut])
        test.extend(block_copy[cut:])
    return train, test


def marker_profile(rows: Iterable[Row], min_occurrences: int = 1) -> Dict[str, float]:
    by_marker: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_marker[primary_epistemic_marker(r.answer)].append(r.correct)
    out: Dict[str, float] = {}
    for marker, vals in by_marker.items():
        if len(vals) >= min_occurrences:
            out[marker] = mean(vals)
    return out


def dataset_accuracy(rows: Iterable[Row]) -> float:
    vals = [r.correct for r in rows]
    return mean(vals) if vals else 0.0


def mean_abs_calibration_error(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    if not y_true:
        return float("nan")
    return mean(abs(float(y) - p) for y, p in zip(y_true, y_prob))


def coefficient_of_variation(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    m = mean(values)
    if m == 0:
        return float("nan")
    return pstdev(values) / m


def _rank(values: Sequence[float]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx == 0 or vy == 0:
        return float("nan")
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    return pearson(_rank(x), _rank(y))


def evaluate_model(
    model: str,
    train_rows: Sequence[Row],
    test_rows: Sequence[Row],
    min_occurrences: int,
) -> dict[str, float]:
    tr_by_ds: dict[str, list[Row]] = defaultdict(list)
    te_by_ds: dict[str, list[Row]] = defaultdict(list)
    for r in train_rows:
        if r.model == model:
            tr_by_ds[r.dataset].append(r)
    for r in test_rows:
        if r.model == model:
            te_by_ds[r.dataset].append(r)

    datasets = sorted(set(tr_by_ds) & set(te_by_ds))
    if not datasets:
        return {}

    profiles_all = {d: marker_profile(tr_by_ds[d], min_occurrences=1) for d in datasets}
    profiles_filtered = {d: marker_profile(tr_by_ds[d], min_occurrences=min_occurrences) for d in datasets}
    train_acc = {d: dataset_accuracy(tr_by_ds[d]) for d in datasets}
    test_acc = {d: dataset_accuracy(te_by_ds[d]) for d in datasets}

    i_ece_vals: list[float] = []
    c_ece_vals: list[float] = []
    i_cv_vals: list[float] = []
    mrc_vals: list[float] = []
    mac_vals: list[float] = []
    c_cv_vals: list[float] = []

    for d in datasets:
        prof = profiles_all[d]
        y_true = [r.correct for r in te_by_ds[d]]
        y_prob = [prof.get(primary_epistemic_marker(r.answer), train_acc[d]) for r in te_by_ds[d]]
        i_ece_vals.append(mean_abs_calibration_error(y_true, y_prob))

    for d_src in datasets:
        prof = profiles_all[d_src]
        for d_tgt in datasets:
            if d_src == d_tgt:
                continue
            y_true = [r.correct for r in te_by_ds[d_tgt]]
            y_prob = [prof.get(primary_epistemic_marker(r.answer), train_acc[d_src]) for r in te_by_ds[d_tgt]]
            c_ece_vals.append(mean_abs_calibration_error(y_true, y_prob))

    for d in datasets:
        vals = list(profiles_filtered[d].values())
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            cv = coefficient_of_variation(vals)
            if not math.isnan(cv):
                i_cv_vals.append(cv)

    shared_all = set.intersection(*(set(profiles_filtered[d].keys()) for d in datasets)) if datasets else set()
    shared_all.discard(NO_MARKER)

    if shared_all:
        for marker in shared_all:
            confs = [profiles_filtered[d][marker] for d in datasets]
            cv = coefficient_of_variation(confs)
            if not math.isnan(cv):
                c_cv_vals.append(cv)

            pr = pearson(confs, [test_acc[d] for d in datasets])
            if not math.isnan(pr):
                mac_vals.append(pr)

        for d1, d2 in combinations(datasets, 2):
            shared_pair = set(profiles_filtered[d1]) & set(profiles_filtered[d2])
            shared_pair.discard(NO_MARKER)
            if len(shared_pair) < 2:
                continue
            x = [profiles_filtered[d1][m] for m in sorted(shared_pair)]
            y = [profiles_filtered[d2][m] for m in sorted(shared_pair)]
            sr = spearman(x, y)
            if not math.isnan(sr):
                mrc_vals.append(sr)

    return {
        "model": model,
        "datasets": float(len(datasets)),
        "I-AvgECE": mean(i_ece_vals) if i_ece_vals else float("nan"),
        "C-AvgECE": mean(c_ece_vals) if c_ece_vals else float("nan"),
        "I-AvgCV": mean(i_cv_vals) if i_cv_vals else float("nan"),
        "C-AvgCV": mean(c_cv_vals) if c_cv_vals else float("nan"),
        "MAC": mean(mac_vals) if mac_vals else float("nan"),
        "MRC": mean(mrc_vals) if mrc_vals else float("nan"),
        "avg_test_accuracy": mean(test_acc.values()) if test_acc else float("nan"),
    }


def format_float(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.4f}"


def write_summary(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "datasets", "avg_test_accuracy", "I-AvgECE", "C-AvgECE", "I-AvgCV", "C-AvgCV", "MAC", "MRC"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = dict(r)
            for c in cols:
                if c in out and isinstance(out[c], float):
                    out[c] = format_float(out[c])
            w.writerow(out)


def write_marker_coverage(path: Path, rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buckets: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"n": 0, "with_marker": 0})
    for r in rows:
        key = (r.model, r.dataset)
        buckets[key]["n"] += 1
        if primary_epistemic_marker(r.answer) != NO_MARKER:
            buckets[key]["with_marker"] += 1

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "dataset", "rows", "rows_with_marker", "marker_coverage"],
        )
        w.writeheader()
        for (model, dataset), s in sorted(buckets.items()):
            n = s["n"]
            k = s["with_marker"]
            cov = (k / n) if n else 0.0
            w.writerow(
                {
                    "model": model,
                    "dataset": dataset,
                    "rows": n,
                    "rows_with_marker": k,
                    "marker_coverage": f"{cov:.4f}",
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ACL'25 epistemic marker experiment on local rollout CSVs.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--min-occurrences", type=int, default=10)
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "output_wood/temperature_experiments/conflictqa/csv/conflictqa_ece_openai_conflictqa_temp06_scale10.csv",
            "output_wood/temperature_experiments/conflictqa/csv/conflictqa_ece_claude_conflictqa_temp06_scale10.csv",
            "output_wood/temperature_experiments/conflictqa/csv/conflictqa_ece_deepseek_conflictqa_temp06_scale10.csv",
            "output_wood/temperature_experiments/conflictqa/csv/conflictqa_ece_gemini_conflictqa_temp06_scale10.csv",
            "output_wood/temperature_experiments/fever/csv/conflictqa_ece_openai_fever_temp06_scale10.csv",
            "output_wood/temperature_experiments/fever/csv/conflictqa_ece_claude_fever_temp06_scale10.csv",
            "output_wood/temperature_experiments/fever/csv/conflictqa_ece_deepseek_fever_temp06_scale10.csv",
            "output_wood/temperature_experiments/fever/csv/conflictqa_ece_gemini_fever_temp06_scale10.csv",
        ],
    )
    ap.add_argument(
        "--out",
        default="output_wood/epistemic_markers/acl25_epistemic_marker_summary_temp06_scale10.csv",
    )
    ap.add_argument(
        "--coverage-out",
        default="output_wood/epistemic_markers/acl25_marker_coverage_temp06_scale10.csv",
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[1]
    input_paths = [base / p for p in args.inputs]
    rows = load_rows(input_paths)
    train, test = split_train_test(rows, train_ratio=args.train_ratio, seed=args.seed)

    models = sorted({r.model for r in rows})
    summary: list[dict[str, float]] = []
    for model in models:
        metrics = evaluate_model(model, train, test, min_occurrences=args.min_occurrences)
        if metrics:
            summary.append(metrics)

    write_summary(base / args.out, summary)
    write_marker_coverage(base / args.coverage_out, rows)

    print("Model results:")
    for r in summary:
        print(
            f"- {r['model']}: "
            f"I-AvgECE={format_float(r['I-AvgECE'])}  "
            f"C-AvgECE={format_float(r['C-AvgECE'])}  "
            f"I-AvgCV={format_float(r['I-AvgCV'])}  "
            f"C-AvgCV={format_float(r['C-AvgCV'])}  "
            f"MAC={format_float(r['MAC'])}  "
            f"MRC={format_float(r['MRC'])}"
        )
    print(f"Saved summary: {base / args.out}")
    print(f"Saved marker coverage: {base / args.coverage_out}")


if __name__ == "__main__":
    main()

