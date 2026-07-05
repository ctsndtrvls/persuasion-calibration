"""
LLM-as-judge: 14-dimension argument quality on persuader counterarguments.

Reads persuasion rollout CSV (e.g. expl.csv), optionally joins task complexity_level
from the FEVER subsample CSV, scores counterarguments, caches incrementally.

Example (100 items, gpt-5.4-mini via OpenRouter — same as persuader):
  cd src && python3 run_persuasion_arg_quality_judge.py \\
    --input ../output_wood/persuasion/DeepSeek/fever/csv/expl.csv \\
    --fever-meta ../output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv \\
    --max-items 100 --judge openrouter

Requires .env: OPENROUTER_API_KEY (and optionally OPENAI_API_KEY / DEEPSEEK_API_KEY for --judge openai|deepseek).

Human-validation sample (no API):
  python3 run_persuasion_arg_quality_judge.py --prepare-human-sample 100 \\
    --input ../output_wood/persuasion/DeepSeek/fever/csv/expl.csv \\
    --fever-meta ../output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

from build_conflictqa_subsets_wood import load_dotenv
from collect_conflictqa_ece import parse_json_obj
from persuasion_arg_quality import (
    HUMAN_ANNOTATION_GUIDE,
    QUALITY_DIMENSIONS,
    SYSTEM_PROMPT_ARG_QUALITY,
    SYSTEM_PROMPT_ARG_QUALITY_TOP3,
    TOP_LEVEL_QUALITY_DIMENSIONS,
    build_user_prompt,
    validate_judge_scores,
    validate_judge_scores_fine14,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_14DIM = (
    PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever" / "arg_quality" / "archive_14dim"
)

QUALITY_SCHEMES: dict[str, tuple[str, ...]] = {
    "top3": TOP_LEVEL_QUALITY_DIMENSIONS,
    "fine14": QUALITY_DIMENSIONS,
}
DEFAULT_INPUT = PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever" / "csv" / "expl.csv"
DEFAULT_FEVER_META = (
    PROJECT_ROOT
    / "output_wood"
    / "dataset_subsampling"
    / "fever"
    / "csv"
    / "fever600_200x3_complexity_wood_v1_lr_40.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever" / "arg_quality"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.4-mini"


def judge_cache_slug(judge: str, model: str | None = None) -> str:
    if model:
        safe = model.replace("/", "_").replace(":", "_")
        return f"{judge}__{safe}"
    return judge


def call_openai_json(
    client: OpenAI, model: str, user_prompt: str, *, system_prompt: str
) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)


def call_openrouter_json(
    model: str, user_prompt: str, *, system_prompt: str, max_tokens: int = 2048
) -> dict[str, Any]:
    """OpenRouter chat (e.g. openai/gpt-5.4-mini) — same stack as persuader."""
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=120.0,
        max_retries=1,
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    try:
        resp = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
            timeout=120.0,
        )
    except Exception:
        resp = client.chat.completions.create(**kwargs, timeout=120.0)
    raw = resp.choices[0].message.content or "{}"
    return parse_json_obj(raw)


def normalize_label(x: object) -> str:
    s = str(x or "").strip().upper()
    if "NOT ENOUGH" in s or s == "NEI":
        return "NOT ENOUGH INFO"
    if "REFUTE" in s:
        return "REFUTES"
    if "SUPPORT" in s:
        return "SUPPORTS"
    return s


def load_persuasion_rows(
    path: Path,
    *,
    turn: int | None,
    min_turn: int,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["counterargument"].fillna("").astype(str).str.strip() != ""].copy()
    if turn is not None:
        df = df[df["turn"] == turn]
    else:
        df = df[df["turn"] >= min_turn]
    return df


def join_complexity(df: pd.DataFrame, fever_meta: Path | None) -> pd.DataFrame:
    if fever_meta is None or not fever_meta.exists():
        return df
    meta = pd.read_csv(fever_meta)
    if "original_index" not in meta.columns or "complexity_level" not in meta.columns:
        return df
    sub = meta[["original_index", "complexity_level"]].drop_duplicates("original_index")
    return df.merge(sub, on="original_index", how="left")


def add_dialogue_context(df: pd.DataFrame) -> pd.DataFrame:
    """Target verdict before this row's counterargument (previous turn in same dialogue)."""
    df = df.sort_values(["dialogue_id", "turn"]).copy()
    prev_answers: list[str] = []
    for _, g in df.groupby("dialogue_id", sort=False):
        answers = g["answer"].map(normalize_label).tolist()
        for i, idx in enumerate(g.index):
            prev_answers.append(answers[i - 1] if i > 0 else answers[0])
    df["target_answer_before"] = prev_answers
    return df


def make_eval_id(row: pd.Series) -> str:
    return f"{row['dialogue_id']}__t{int(row['turn'])}"


def dialogue_flip_flag(full_df: pd.DataFrame) -> pd.Series:
    """Per-dialogue flip from the full rollout (final), not a single-turn slice."""
    if "flipped_from_initial" not in full_df.columns:
        return pd.Series(0, index=full_df["dialogue_id"].unique())
    return full_df.groupby("dialogue_id")["flipped_from_initial"].max()


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
    *,
    flip_lookup: pd.Series | None = None,
) -> pd.DataFrame:
    """Balance flip vs no-flip when possible (uses final dialogue-level flip)."""
    if len(df) <= n:
        return df.copy()
    rng = np.random.default_rng(seed)
    if flip_lookup is None and "flipped_from_initial" not in df.columns:
        return df.sample(n=n, random_state=seed)

    parts: list[pd.DataFrame] = []
    flips = flip_lookup if flip_lookup is not None else dialogue_flip_flag(df)
    df = df.drop_duplicates(subset="dialogue_id", keep="first").copy()
    df["_dialogue_flipped"] = df["dialogue_id"].map(flips).fillna(0).astype(int)
    for flag in (0, 1):
        g = df[df["_dialogue_flipped"] == flag]
        take = min(len(g), n // 2)
        if take > 0:
            parts.append(g.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    out = pd.concat(parts, ignore_index=True)
    if len(out) < n:
        rest = df[~df.index.isin(out.index)]
        need = n - len(out)
        if len(rest) > 0:
            out = pd.concat(
                [out, rest.sample(n=min(need, len(rest)), random_state=seed)],
                ignore_index=True,
            )
    return out.head(n)


HUMAN_README = """Human annotation instructions (3 top-level dimensions)
======================================================

{guide}

Fill columns: cogency, effectiveness, reasonableness (integers 0-3 only).

Rules:
- Do not change eval_id, claim, counterargument, or context columns.
- Do not use gold labels or web search to verify the claim.
- Judge whether the counterargument would persuade an LLM fact-checker in this context.

Annotator ID: {annotator_id}
"""


def archive_14dim_artifacts(out_dir: Path) -> None:
    """Move/copy prior 14-dimension human + judge files into archive_14dim/."""
    archive = ARCHIVE_14DIM
    archive.mkdir(parents=True, exist_ok=True)
    patterns = (
        "human_annotation*.csv",
        "human_annotation*.readme.txt",
        "human_annotation_sample_manifest.txt",
        "arg_quality_long_*__openrouter__openai_gpt-5.4-mini.csv",
    )
    for pat in patterns:
        for path in out_dir.glob(pat):
            if path.parent.resolve() == archive.resolve():
                continue
            dest = archive / path.name
            if not dest.exists():
                shutil.copy2(path, dest)
    readme = archive / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Archive: 14 fine-grained argument-quality dimensions (legacy experiment).\n"
            "Current human annotation uses top3: cogency, effectiveness, reasonableness.\n",
            encoding="utf-8",
        )


def prepare_human_sample(
    df: pd.DataFrame,
    out_dir: Path,
    n: int,
    seed: int,
    *,
    full_df: pd.DataFrame,
    annotator_ids: tuple[str, ...] = ("annotator_1", "annotator_2", "annotator_3"),
) -> pd.DataFrame:
    """Export one CSV per annotator (context + 3 top-level score columns)."""
    archive_14dim_artifacts(out_dir)
    work = df.copy()
    if (work["turn"] == 1).any():
        work = work[work["turn"] == 1]
    flip_lookup = dialogue_flip_flag(full_df)
    work = stratified_sample(
        work.drop_duplicates("dialogue_id", keep="first"),
        n,
        seed,
        flip_lookup=flip_lookup,
    )
    work["eval_id"] = work.apply(make_eval_id, axis=1)

    export_cols = [
        "eval_id",
        "dialogue_id",
        "turn",
        "original_index",
        "claim",
        "counterargument",
        "target_answer_before",
        "answer",
        "confidence",
        "complexity_level",
        "flipped_from_initial",
    ]
    export_cols = [c for c in export_cols if c in work.columns]
    base = work[export_cols].copy()
    base = base.rename(columns={"answer": "target_answer_after"})
    flip_map = dialogue_flip_flag(full_df)
    base["dialogue_flipped_final"] = base["dialogue_id"].map(flip_map).fillna(0).astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ann_id in annotator_ids:
        ann_df = base.copy()
        for dim in TOP_LEVEL_QUALITY_DIMENSIONS:
            ann_df[dim] = pd.NA
        out_path = out_dir / f"human_annotation_{ann_id}.csv"
        ann_df.to_csv(out_path, index=False)
        paths.append(out_path)
        readme = out_dir / f"human_annotation_{ann_id}.readme.txt"
        readme.write_text(
            HUMAN_README.format(annotator_id=ann_id, guide=HUMAN_ANNOTATION_GUIDE),
            encoding="utf-8",
        )

    manifest = out_dir / "human_annotation_sample_manifest.txt"
    manifest.write_text(
        f"scheme=top3 (cogency, effectiveness, reasonableness)\n"
        f"n_rows={len(base)}\nseed={seed}\nfiles:\n"
        + "\n".join(f"  - {p.name}" for p in paths)
        + f"\narchive_14dim={ARCHIVE_14DIM}\n",
        encoding="utf-8",
    )
    return base


def judge_all(
    df: pd.DataFrame,
    out_long_csv: Path,
    *,
    judge_name: str,
    call_fn: Any,
    sleep_s: float,
    max_items: int | None,
    seed: int,
    flip_lookup: pd.Series | None = None,
    dimensions: tuple[str, ...] = TOP_LEVEL_QUALITY_DIMENSIONS,
) -> pd.DataFrame:
    out_long_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_long_csv.exists():
        done = pd.read_csv(out_long_csv)
        done_keys = set(done["eval_id"].astype(str))
    else:
        done = pd.DataFrame()
        done_keys = set()

    work = df.copy()
    if "eval_id" not in work.columns:
        work["eval_id"] = work.apply(make_eval_id, axis=1)
    work = work.drop_duplicates(subset="eval_id", keep="first")
    if max_items is not None:
        work = stratified_sample(work, max_items, seed, flip_lookup=flip_lookup)

    rows: list[dict[str, Any]] = []
    for _, r in work.iterrows():
        eid = str(r["eval_id"])
        if eid in done_keys:
            continue
        ca = str(r["counterargument"]).strip()
        if not ca:
            continue
        complexity = r.get("complexity_level")
        complexity_i = int(complexity) if pd.notna(complexity) else None
        user = build_user_prompt(
            claim=str(r["claim"]),
            counterargument=ca,
            target_answer=normalize_label(r["answer"]),
            target_confidence=r["confidence"],
            turn=int(r["turn"]),
            target_answer_before=normalize_label(r.get("target_answer_before", r["answer"])),
            persuader_complexity_level=complexity_i,
            dimensions=dimensions,
        )
        try:
            raw = call_fn(user)
            scores = validate_judge_scores(raw, dimensions)
            row: dict[str, Any] = {
                "eval_id": eid,
                "dialogue_id": r["dialogue_id"],
                "turn": int(r["turn"]),
                "original_index": r.get("original_index"),
                "judge": judge_name,
                "reasoning": raw.get("reasoning", ""),
            }
            row.update(scores)
            row["mean_score"] = sum(scores.values()) / len(scores)
            rows.append(row)
            pd.DataFrame([row]).to_csv(
                out_long_csv,
                mode="a",
                header=not out_long_csv.exists(),
                index=False,
            )
            time.sleep(sleep_s)
            print(f"[ok] {judge_name} {eid}")
        except Exception as e:
            print(f"[ERROR] {judge_name} {eid}: {e}")

    if rows:
        new = pd.DataFrame(rows)
        return pd.concat([done, new], ignore_index=True) if len(done) else new
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM judge for persuasion argument quality.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--fever-meta", type=Path, default=DEFAULT_FEVER_META)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--run-name", default="judge_v1")
    ap.add_argument("--turn", type=int, default=1, help="Score only this turn (default: 1; ignored with --all-turns).")
    ap.add_argument("--min-turn", type=int, default=1)
    ap.add_argument(
        "--all-turns",
        action="store_true",
        help="Score every persuasion turn (turn >= --min-turn) in the rollout CSV.",
    )
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument(
        "--human-sample",
        type=Path,
        default=None,
        help="CSV with eval_id column: judge exactly these rows (e.g. human_annotation_sample_100.csv).",
    )
    ap.add_argument(
        "--fever-subsample",
        type=Path,
        default=None,
        help="Restrict to original_index values in this FEVER CSV (e.g. fever600_200x3_....csv).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep-s", type=float, default=0.2)
    ap.add_argument(
        "--prepare-human-sample",
        type=int,
        metavar="N",
        default=None,
        help="Write N-row human annotation CSV and exit (no API).",
    )
    ap.add_argument(
        "--quality-scheme",
        choices=tuple(QUALITY_SCHEMES),
        default="top3",
        help="top3=cogency/effectiveness/reasonableness (default); fine14=legacy 14 sub-dimensions.",
    )
    ap.add_argument(
        "--judge",
        choices=("openrouter", "openai", "deepseek"),
        default="openrouter",
        help="Judge backend (default: openrouter = gpt-5.4-mini via OpenRouter).",
    )
    ap.add_argument(
        "--openrouter-model",
        default=DEFAULT_OPENROUTER_MODEL,
        help="OpenRouter model id (default: openai/gpt-5.4-mini).",
    )
    ap.add_argument("--openai-model", default="gpt-4o-2024-11-20")
    ap.add_argument("--deepseek-model", default="deepseek-chat")
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    if args.judge == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY missing in .env (required for --judge openrouter).")

    full_df = pd.read_csv(args.input)
    turn_filter = None if args.all_turns else args.turn
    df = load_persuasion_rows(args.input, turn=turn_filter, min_turn=args.min_turn)
    if args.all_turns:
        print(f"[all-turns] {len(df)} counterargument rows (turn >= {args.min_turn})")
    df = join_complexity(df, args.fever_meta)
    df = add_dialogue_context(df)
    flip_lookup = dialogue_flip_flag(full_df)

    if args.fever_subsample is not None:
        fmeta = pd.read_csv(args.fever_subsample)
        allowed = set(fmeta["original_index"].astype(int))
        df = df[df["original_index"].astype(int).isin(allowed)]
        print(f"[fever-subsample] {len(allowed)} claims in meta; {len(df)} turn rows to judge")

    use_human_ids = args.human_sample is not None
    if use_human_ids:
        hs = pd.read_csv(args.human_sample)
        if "eval_id" not in hs.columns:
            raise SystemExit(f"--human-sample missing eval_id column: {args.human_sample}")
        target_ids = set(hs["eval_id"].astype(str))
        df = df.copy()
        df["eval_id"] = df.apply(make_eval_id, axis=1)
        df = df[df["eval_id"].isin(target_ids)].drop_duplicates("eval_id", keep="first")
        if len(df) < len(target_ids):
            print(f"[warn] matched {len(df)}/{len(target_ids)} eval_ids from rollout CSV")
        args.max_items = None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare_human_sample is not None:
        sample = prepare_human_sample(
            df, args.out_dir, args.prepare_human_sample, args.seed, full_df=full_df
        )
        print(f"Wrote {len(sample)} rows x 3 annotator CSVs (top3) -> {args.out_dir}")
        print(f"14-dim archive -> {ARCHIVE_14DIM}")
        print(f"Manifest -> {args.out_dir / 'human_annotation_sample_manifest.txt'}")
        return

    dimensions = QUALITY_SCHEMES[args.quality_scheme]
    if args.quality_scheme == "fine14":
        system_prompt = SYSTEM_PROMPT_ARG_QUALITY
    else:
        system_prompt = SYSTEM_PROMPT_ARG_QUALITY_TOP3

    judges: list[tuple[str, Any]] = []
    if args.judge == "openrouter":
        slug = judge_cache_slug("openrouter", args.openrouter_model)
        judges.append(
            (
                slug,
                lambda up, m=args.openrouter_model, sp=system_prompt: call_openrouter_json(
                    m, up, system_prompt=sp
                ),
            )
        )
    elif args.judge == "openai":
        oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        judges.append(
            (
                "openai",
                lambda up, sp=system_prompt: call_openai_json(oa, args.openai_model, up, system_prompt=sp),
            )
        )
    elif args.judge == "deepseek":
        ds = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        judges.append(
            (
                "deepseek",
                lambda up, sp=system_prompt: call_openai_json(ds, args.deepseek_model, up, system_prompt=sp),
            )
        )

    for judge_name, fn in judges:
        scheme_tag = args.quality_scheme
        out_long = args.out_dir / f"arg_quality_long_{args.run_name}__{scheme_tag}__{judge_name}.csv"
        judge_all(
            df,
            out_long,
            judge_name=judge_name,
            call_fn=fn,
            sleep_s=args.sleep_s,
            max_items=args.max_items,
            seed=args.seed,
            flip_lookup=flip_lookup,
            dimensions=dimensions,
        )
        print(f"Cache: {out_long}")


if __name__ == "__main__":
    main()
