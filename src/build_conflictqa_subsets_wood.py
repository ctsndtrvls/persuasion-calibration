from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "conflictqa"
OUT_DIR = PROJECT_ROOT / "output_wood"

POPQA_INPUT = DATA_DIR / "conflictQA-popQA-llama2-7b.json"
STRATEGY_INPUT = DATA_DIR / "conflictQA-strategyQA-llama2-7b.json"

SYSTEM_PROMPT_WOOD = """
You are an expert in Cognitive Science and Task Complexity.
You will assess an NLP task instance using Robert E. Wood's Task Complexity Framework (1986):

- Component Complexity: number of distinct information cues/acts to process.
- Coordinative Complexity: strength of relationships/logic between cues (reasoning depth).
- Dynamic Complexity: ambiguity or changing states (high in ambiguous or under-specified cases).

RETURN JSON ONLY:
{
  "wood_component": (1-10),
  "wood_coordinative": (1-10),
  "wood_dynamic": (1-10),
  "wood_total": (sum of the three),
  "reasoning": "Short justification"
}
""".strip()


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and not os.getenv(key):
            os.environ[key] = val


def call_openai_json(client: OpenAI, model: str, user_prompt: str) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_WOOD},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)


def to_input_text(row: pd.Series) -> str:
    q = str(row.get("question", "")).strip()
    pm = str(row.get("parametric_memory", "")).strip()
    cm = str(row.get("counter_memory", "")).strip()
    gt = row.get("ground_truth", [])
    if isinstance(gt, list):
        gt_text = ", ".join(map(str, gt))
    else:
        gt_text = str(gt)
    return (
        f"Question: {q}\n\n"
        f"Ground truth (reference): {gt_text}\n\n"
        f"Initial answer rationale:\n{pm}\n\n"
        f"Counter-answer rationale:\n{cm}"
    ).strip()


def ensure_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "original_index" not in out.columns:
        out["original_index"] = np.arange(len(out))
    return out


def judge_all(
    df: pd.DataFrame,
    out_long_csv: Path,
    openai_model: str,
    deepseek_model: str,
    use_openai: bool,
    use_deepseek: bool,
    sleep_s: float = 0.15,
    max_items: int | None = None,
) -> pd.DataFrame:
    out_long_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_long_csv.exists():
        done = pd.read_csv(out_long_csv)
        done_keys = set(zip(done["dataset"], done["original_index"], done["judge"]))
    else:
        done = pd.DataFrame()
        done_keys = set()

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if use_openai else None
    deepseek_client = (
        OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        if use_deepseek
        else None
    )

    judges: list[tuple[str, Any]] = []
    if use_deepseek:
        judges.append(
            ("deepseek", lambda prompt: call_openai_json(deepseek_client, deepseek_model, prompt))
        )
    if use_openai:
        judges.append(("openai", lambda prompt: call_openai_json(openai_client, openai_model, prompt)))

    rows = []
    work = df.copy()
    if max_items is not None:
        work = work.head(max_items)

    for _, r in work.iterrows():
        ds = r["dataset"]
        oid = r["original_index"]
        prompt = r["input_text_for_judge"]

        for judge_name, fn in judges:
            if (ds, oid, judge_name) in done_keys:
                continue
            try:
                res = fn(prompt)
                row = {
                    "dataset": ds,
                    "original_index": oid,
                    "judge": judge_name,
                    "wood_component": res.get("wood_component"),
                    "wood_coordinative": res.get("wood_coordinative"),
                    "wood_dynamic": res.get("wood_dynamic"),
                    "wood_total": res.get("wood_total"),
                    "reasoning": res.get("reasoning", ""),
                }
                rows.append(row)
                pd.DataFrame([row]).to_csv(
                    out_long_csv,
                    mode="a",
                    header=not out_long_csv.exists(),
                    index=False,
                )
                time.sleep(sleep_s)
            except Exception as e:
                print(f"[ERROR] {judge_name} failed for ({ds},{oid}): {e}")

    if rows:
        new = pd.DataFrame(rows)
        if len(done) > 0:
            return pd.concat([done, new], ignore_index=True)
        return new
    return done


def pivot_and_aggregate(df_base: pd.DataFrame, df_long: pd.DataFrame) -> pd.DataFrame:
    wide = df_long.pivot_table(
        index=["dataset", "original_index"],
        columns="judge",
        values=["wood_component", "wood_coordinative", "wood_dynamic", "wood_total"],
        aggfunc="first",
    )
    wide.columns = [f"{m}__{j}" for (m, j) in wide.columns]
    wide = wide.reset_index()

    merged = df_base.merge(wide, on=["dataset", "original_index"], how="left")
    totals = [c for c in merged.columns if c.startswith("wood_total__")]
    merged["wood_total_mean"] = merged[totals].mean(axis=1)
    return merged


def assign_quantile_bins(series: pd.Series, n_bins: int = 3) -> pd.Series:
    return pd.qcut(series, q=n_bins, labels=False, duplicates="drop") + 1


def stratified_sample(df: pd.DataFrame, n_per_level: int = 160, seed: int = 42) -> pd.DataFrame:
    out = df.copy()
    out["wood_total_mean"] = pd.to_numeric(out["wood_total_mean"], errors="coerce")
    out = out.dropna(subset=["wood_total_mean"]).copy()
    out["complexity_level"] = assign_quantile_bins(out["wood_total_mean"], n_bins=3).astype(int)

    target_total = n_per_level * 3
    parts = []
    for lvl in (1, 2, 3):
        g = out[out["complexity_level"] == lvl]
        take_n = min(len(g), n_per_level)
        if take_n > 0:
            parts.append(g.sample(n=take_n, random_state=seed))

    sampled = pd.concat(parts) if parts else pd.DataFrame(columns=out.columns)
    if len(sampled) < target_total:
        # Fallback for sparse bins: top up from remaining rows by score to keep total size stable.
        remaining = out.drop(index=sampled.index, errors="ignore")
        top_up_n = min(target_total - len(sampled), len(remaining))
        if top_up_n > 0:
            top_up = remaining.sort_values("wood_total_mean", ascending=False).head(top_up_n)
            sampled = pd.concat([sampled, top_up])

    if len(sampled) < target_total:
        raise ValueError(
            f"Not enough rows after fallback: need {target_total}, found {len(sampled)}"
        )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prefilter_by_popularity(
    df: pd.DataFrame,
    n_levels: int = 3,
    n_per_level: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fast pre-selection stage to cut API volume while preserving diversity.
    We sample a larger balanced candidate pool by popularity quantiles
    (e.g., 400 x 3 = 1200 rows) and run Wood judges only on that pool.
    """
    out = df.copy()
    out["popularity"] = pd.to_numeric(out.get("popularity"), errors="coerce")
    out = out.dropna(subset=["popularity"]).copy()
    out["prefilter_level"] = assign_quantile_bins(out["popularity"], n_bins=n_levels).astype(int)

    picked = []
    for lvl in range(1, n_levels + 1):
        g = out[out["prefilter_level"] == lvl]
        if len(g) < n_per_level:
            # If a bin is small, take all from that bin.
            picked.append(g)
        else:
            picked.append(g.sample(n=n_per_level, random_state=seed))
    return pd.concat(picked, ignore_index=True)


def read_conflictqa(path: Path, dataset_tag: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = ensure_id_columns(df)
    df["dataset"] = dataset_tag
    df["input_text_for_judge"] = df.apply(to_input_text, axis=1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="wood_conflictqa_v1")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--deepseek-model", default="deepseek-chat")
    parser.add_argument("--sleep-s", type=float, default=0.15)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--prefilter-per-level",
        type=int,
        default=400,
        help=(
            "Pre-sample candidates by popularity before Wood judging. "
            "Default 400 means ~1200 candidates per dataset (400x3). "
            "Set 0 to disable prefilter and judge all rows."
        ),
    )
    parser.add_argument("--no-openai", action="store_true")
    parser.add_argument("--no-deepseek", action="store_true")
    args = parser.parse_args()

    # Make CLI behavior consistent with notebook: read keys from project .env.
    load_dotenv(PROJECT_ROOT / ".env")

    use_openai = not args.no_openai
    use_deepseek = not args.no_deepseek
    if not use_openai and not use_deepseek:
        raise ValueError("At least one judge must be enabled.")

    pop_df = read_conflictqa(POPQA_INPUT, "conflictqa_popqa")
    strat_df = read_conflictqa(STRATEGY_INPUT, "conflictqa_strategyqa")

    # Speed/quality trade-off: judge only a large balanced candidate pool first.
    if args.prefilter_per_level and args.prefilter_per_level > 0:
        pop_df = prefilter_by_popularity(
            pop_df,
            n_levels=3,
            n_per_level=args.prefilter_per_level,
            seed=42,
        )
        strat_df = prefilter_by_popularity(
            strat_df,
            n_levels=3,
            n_per_level=args.prefilter_per_level,
            seed=42,
        )
        print(
            f"Prefiltered candidate pool sizes -> popQA: {len(pop_df)}, "
            f"strategyQA: {len(strat_df)}"
        )

    base_df = pd.concat([pop_df, strat_df], ignore_index=True)

    out_long = OUT_DIR / f"wood_judgements_long_{args.run_name}.csv"
    df_long = judge_all(
        base_df,
        out_long_csv=out_long,
        openai_model=args.openai_model,
        deepseek_model=args.deepseek_model,
        use_openai=use_openai,
        use_deepseek=use_deepseek,
        sleep_s=args.sleep_s,
        max_items=args.max_items,
    )
    df_enriched = pivot_and_aggregate(base_df, df_long)

    pop_sub = stratified_sample(df_enriched[df_enriched["dataset"] == "conflictqa_popqa"], n_per_level=160)
    strat_sub = stratified_sample(
        df_enriched[df_enriched["dataset"] == "conflictqa_strategyqa"],
        n_per_level=160,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pop_out = OUT_DIR / f"conflictqa_popqa480_160x3_wood_{args.run_name}.csv"
    strat_out = OUT_DIR / f"conflictqa_strategyqa480_160x3_wood_{args.run_name}.csv"
    pop_sub.to_csv(pop_out, index=False)
    strat_sub.to_csv(strat_out, index=False)

    print("Saved:")
    print(" -", out_long)
    print(" -", pop_out, pop_sub.shape)
    print(" -", strat_out, strat_sub.shape)


if __name__ == "__main__":
    main()

