from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "conflictqa"
OUT_DIR = PROJECT_ROOT / "output_wood"

POPQA_INPUT = DATA_DIR / "conflictQA-popQA-llama2-7b.json"
STRATEGY_INPUT = DATA_DIR / "conflictQA-strategyQA-llama2-7b.json"

N_LEVELS = 3
N_PER_LEVEL = 200
SEED = 42


def assign_quantile_bins(series: pd.Series, n_bins: int = 3) -> pd.Series:
    # 1..n bins
    return pd.qcut(series, q=n_bins, labels=False, duplicates="drop") + 1


def stratified_sample_by_popularity(
    df: pd.DataFrame,
    n_levels: int = 3,
    n_per_level: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    out = df.copy()
    out["popularity"] = pd.to_numeric(out["popularity"], errors="coerce")
    out = out.dropna(subset=["popularity"]).copy()
    out["complexity_level"] = assign_quantile_bins(out["popularity"], n_levels).astype(int)

    sampled = []
    for level in range(1, n_levels + 1):
        group = out[out["complexity_level"] == level]
        if len(group) < n_per_level:
            raise ValueError(
                f"Not enough rows in complexity level {level}: "
                f"need {n_per_level}, found {len(group)}"
            )
        sampled.append(group.sample(n=n_per_level, random_state=seed))

    return pd.concat(sampled, ignore_index=True)


def build_one_subset(input_path: Path, out_name: str) -> Path:
    df = pd.read_json(input_path, lines=True)
    subset = stratified_sample_by_popularity(
        df,
        n_levels=N_LEVELS,
        n_per_level=N_PER_LEVEL,
        seed=SEED,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    subset.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    popqa_out = build_one_subset(
        POPQA_INPUT,
        "conflictqa_popqa600_200x3_popularity_llama2_7b.csv",
    )
    strategy_out = build_one_subset(
        STRATEGY_INPUT,
        "conflictqa_strategyqa600_200x3_popularity_llama2_7b.csv",
    )

    print("Saved:")
    print(f" - {popqa_out}")
    print(f" - {strategy_out}")


if __name__ == "__main__":
    main()

