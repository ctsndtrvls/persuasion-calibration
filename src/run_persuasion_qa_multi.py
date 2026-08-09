"""
Multi-turn persuasion on QA datasets (ConflictQA PopQA and DebateQA) with a
configurable target model.

Target models supported (via --target-provider):
  - openai      : first-party OpenAI API (e.g. gpt-4o-2024-11-20)
  - openrouter  : OpenRouter routes (e.g. google/gemma-4-26b-a4b-it, qwen/qwen3-14b)
  - deepseek    : DeepSeek API (deepseek-chat; same as FEVER persuasion line)

Persuader is always openai/gpt-5.4-mini via OpenRouter (same as the DeepSeek FEVER runs).
Target confidence is elicited on a 1–10 scale. Default mode: until_flip, hard cap
--max-turns 15.

Datasets use the same stratified 480-item Wood subsets as the calibration line:
  PopQA    : output_wood/dataset_subsampling/conflictqa/csv/conflictqa_popqa480_160x3_wood_v2_resplit.csv
  DebateQA : output_wood/dataset_subsampling/debateqa/csv/debateqa_480_160x3.csv

Example:
  cd src && python3 run_persuasion_qa_multi.py \\
    --dataset conflictqa_popqa \\
    --target-provider openai --target-model gpt-4o-2024-11-20 \\
    --target-label GPT-4o --max-items 5

Requires .env: OPENROUTER_API_KEY (persuader + openrouter target);
OPENAI_API_KEY when --target-provider openai;
DEEPSEEK_API_KEY when --target-provider deepseek.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from collect_conflictqa_ece import (
    DEBATEQA_DEFAULT,
    DEEPSEEK_API_MODEL,
    POPQA_SUBSET_DEFAULT,
    _parse_ground_truth_cell,
    load_dotenv,
    normalize_phrase,
)
from run_persuasion_fever_pilot import (
    PERSUADER_SYSTEM,
    TARGET_SYSTEM_SCALE10,
    call_openrouter_json,
    clip_confidence_scale10,
    parse_json_obj,
)
from run_persuasion_mixed import (
    QA_PERSUADER_USER,
    QA_TARGET_HISTORY,
    answers_differ,
    format_qa_context,
    format_refs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSUASION_ROOT = PROJECT_ROOT / "output_wood" / "persuasion"

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "conflictqa_popqa": {
        "default_input": POPQA_SUBSET_DEFAULT,
        "context_cols": ("parametric_memory", "counter_memory"),
        "out_slug": "popqa",
    },
    "debateqa": {
        "default_input": DEBATEQA_DEFAULT,
        "context_cols": ("partial_answers",),
        "out_slug": "debateqa",
    },
}

CSV_COLUMNS = [
    "dialogue_id",
    "item_id",
    "target_model",
    "target_label",
    "persuader_model",
    "dataset",
    "original_index",
    "turn",
    "claim",
    "gold_label",
    "answer",
    "confidence",
    "confidence_scale",
    "flipped_from_initial",
    "flip_turn",
    "counterargument",
    "decision_explanation",
    "target_prompt",
    "target_raw_json",
    "persuader_prompt",
    "persuader_raw_json",
    "conversation_json",
    "persuasion_mode",
    "stop_reason",
    "error",
]


def _openai_client(provider: str):
    from openai import OpenAI

    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120.0, max_retries=2)
    if provider == "openrouter":
        return OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0,
            max_retries=2,
        )
    if provider == "deepseek":
        return OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=120.0,
            max_retries=2,
        )
    raise ValueError(f"Unsupported target provider: {provider}")


def call_qa_target(
    question: str,
    refs: str,
    history: list[str],
    *,
    provider: str,
    model: str,
    temperature: float,
) -> tuple[str, int, str, str, str]:
    if not history:
        numbered = "(none yet)"
    else:
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(history, start=1))
    user = QA_TARGET_HISTORY.format(question=question, refs=refs, history=numbered)
    client = _openai_client(provider)
    messages = [
        {"role": "system", "content": TARGET_SYSTEM_SCALE10},
        {"role": "user", "content": user},
    ]
    try:
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception:
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    raw = r.choices[0].message.content or "{}"
    data = parse_json_obj(raw)
    return (
        str(data.get("answer", "")),
        clip_confidence_scale10(data.get("confidence", 1)),
        str(data.get("decision_explanation", "")).strip(),
        user,
        raw,
    )


def call_qa_persuader(
    *,
    model: str,
    question: str,
    refs: str,
    target_answer: str,
    target_confidence: int,
    complexity_level: int,
    context_block: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str, str]:
    user = QA_PERSUADER_USER.format(
        question=question,
        refs=refs,
        target_answer=target_answer,
        target_confidence=target_confidence,
        complexity_level=complexity_level,
        context_block=context_block,
    )
    data, raw_json = call_openrouter_json(
        model=model,
        system=PERSUADER_SYSTEM.replace("fact-checking debate", "question-answering debate"),
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return str(data.get("counterargument", "")).strip(), user, raw_json


def load_dataset_rows(dataset: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "question" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError(f"{dataset} CSV must contain 'question' and 'ground_truth': {path}")
    if "original_index" not in df.columns:
        df = df.copy()
        df["original_index"] = range(len(df))
    if "item_id" not in df.columns:
        df = df.copy()
        df["item_id"] = range(1, len(df) + 1)
    df["dataset"] = dataset
    return df


def _row(**kwargs: Any) -> dict[str, Any]:
    return {col: kwargs.get(col, "") for col in CSV_COLUMNS}


def run_dialogue(
    row: pd.Series,
    *,
    dataset: str,
    target_provider: str,
    target_model: str,
    target_label: str,
    persuader_model: str,
    complexity_level: int,
    persuasion_mode: str,
    max_turns: int,
    target_temperature: float,
    persuader_temperature: float,
    persuader_max_tokens: int,
    sleep_s: float,
) -> list[dict[str, Any]]:
    claim = str(row["question"]).strip()
    refs = format_refs(row.get("ground_truth"))
    gold = refs
    item_id = int(row["item_id"])
    oid = int(row["original_index"])
    dialogue_id = str(uuid.uuid4())
    context_block = format_qa_context(row, complexity_level)

    records: list[dict[str, Any]] = []
    history: list[str] = []
    transcript: list[dict[str, Any]] = []
    err = ""

    def base_kwargs() -> dict[str, Any]:
        return dict(
            dialogue_id=dialogue_id,
            item_id=item_id,
            target_model=target_model,
            target_label=target_label,
            persuader_model=persuader_model,
            dataset=dataset,
            original_index=oid,
            claim=claim,
            gold_label=gold,
            confidence_scale="1-10",
            persuasion_mode=persuasion_mode,
        )

    try:
        answer0, conf0, expl0, target_prompt0, target_raw0 = call_qa_target(
            claim, refs, [], provider=target_provider, model=target_model, temperature=target_temperature
        )
    except Exception as e:
        answer0, conf0, expl0, target_prompt0, target_raw0 = "", 1, "", "", ""
        err = f"target_turn0: {e}"

    transcript.append(
        {
            "turn": 0,
            "role": "target",
            "prompt": target_prompt0,
            "response_raw_json": target_raw0,
            "answer": answer0,
            "confidence": conf0,
            "decision_explanation": expl0,
        }
    )
    records.append(
        _row(
            **base_kwargs(),
            turn=0,
            answer=answer0,
            confidence=conf0,
            flipped_from_initial=0,
            flip_turn="",
            counterargument="",
            decision_explanation=expl0,
            target_prompt=target_prompt0,
            target_raw_json=target_raw0,
            conversation_json=json.dumps(transcript, ensure_ascii=False),
            stop_reason="error" if err else "",
            error=err,
        )
    )
    if err:
        return records

    current_answer, current_conf = answer0, conf0
    flip_turn: int | None = None
    until_flip = persuasion_mode == "until_flip"

    for turn in range(1, max_turns + 1):
        time.sleep(sleep_s)
        counter = ""
        persuader_prompt = ""
        persuader_raw = ""
        turn_err = ""
        try:
            counter, persuader_prompt, persuader_raw = call_qa_persuader(
                model=persuader_model,
                question=claim,
                refs=refs,
                target_answer=current_answer,
                target_confidence=current_conf,
                complexity_level=complexity_level,
                context_block=context_block,
                temperature=persuader_temperature,
                max_tokens=persuader_max_tokens,
            )
        except Exception as e:
            turn_err = f"persuader: {e}"

        if counter and not turn_err:
            history.append(counter)
        transcript.append(
            {
                "turn": turn,
                "role": "persuader",
                "prompt": persuader_prompt,
                "response_raw_json": persuader_raw,
                "counterargument": counter,
            }
        )

        time.sleep(sleep_s)
        try:
            if history:
                new_answer, new_conf, new_expl, target_prompt, target_raw = call_qa_target(
                    claim,
                    refs,
                    history,
                    provider=target_provider,
                    model=target_model,
                    temperature=target_temperature,
                )
            else:
                new_answer, new_conf, new_expl, target_prompt, target_raw = (
                    current_answer,
                    current_conf,
                    "",
                    "",
                    "",
                )
                if turn_err:
                    raise RuntimeError(turn_err)
        except Exception as e:
            new_answer, new_conf, new_expl, target_prompt, target_raw = (
                current_answer,
                current_conf,
                "",
                "",
                "",
            )
            turn_err = turn_err or f"target: {e}"

        transcript.append(
            {
                "turn": turn,
                "role": "target",
                "prompt": target_prompt,
                "response_raw_json": target_raw,
                "answer": new_answer,
                "confidence": new_conf,
                "decision_explanation": new_expl,
            }
        )

        flipped = answers_differ(new_answer, answer0)
        if flipped and flip_turn is None:
            flip_turn = turn

        if turn_err:
            stop_reason = "error"
        elif flipped:
            stop_reason = "flipped"
        elif turn >= max_turns:
            stop_reason = "max_turns"
        else:
            stop_reason = ""

        records.append(
            _row(
                **base_kwargs(),
                turn=turn,
                answer=new_answer,
                confidence=new_conf,
                flipped_from_initial=1 if flipped else 0,
                flip_turn=flip_turn if flip_turn is not None else "",
                counterargument=counter,
                decision_explanation=new_expl,
                target_prompt=target_prompt,
                target_raw_json=target_raw,
                persuader_prompt=persuader_prompt,
                persuader_raw_json=persuader_raw,
                conversation_json=json.dumps(transcript, ensure_ascii=False),
                stop_reason=stop_reason,
                error=turn_err,
            )
        )
        current_answer, current_conf = new_answer, new_conf
        if turn_err:
            break
        if until_flip and flipped:
            break

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIGS), required=True)
    parser.add_argument("--input", type=Path, default=None, help="Override dataset subset CSV.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--persuasion-mode", choices=("until_flip", "fixed"), default="until_flip")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--complexity-level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--persuader-model", default="openai/gpt-5.4-mini")
    parser.add_argument(
        "--target-provider",
        choices=("openai", "openrouter", "deepseek"),
        required=True,
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument(
        "--target-label",
        default=None,
        help="Short label used for the output directory (e.g. GPT-4o, Gemma, Qwen, DeepSeek).",
    )
    parser.add_argument("--target-temperature", type=float, default=0.6)
    parser.add_argument("--persuader-temperature", type=float, default=0.8)
    parser.add_argument("--persuader-max-tokens", type=int, default=1024)
    parser.add_argument("--sleep-s", type=float, default=0.35)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    required = ["OPENROUTER_API_KEY"]
    if args.target_provider == "openai":
        required.append("OPENAI_API_KEY")
    elif args.target_provider == "deepseek":
        required.append("DEEPSEEK_API_KEY")
    for key in required:
        if not os.getenv(key):
            raise SystemExit(f"Missing {key} in environment / .env")

    cfg = DATASET_CONFIGS[args.dataset]
    input_path = args.input or cfg["default_input"]
    if not input_path.exists():
        raise SystemExit(f"Dataset CSV not found: {input_path}")

    if args.target_provider == "deepseek" and args.target_model in ("", "deepseek-chat"):
        args.target_model = DEEPSEEK_API_MODEL
    target_label = args.target_label or (
        "DeepSeek" if args.target_provider == "deepseek" else args.target_model.replace("/", "_")
    )

    df = load_dataset_rows(args.dataset, input_path)
    if args.max_items is not None:
        df = df.head(args.max_items)

    out = args.out or (
        PERSUASION_ROOT / target_label / cfg["out_slug"] / "rollout" / "csv" / "expl.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[int] = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out)
        if "item_id" in prev.columns:
            done_ids = {int(x) for x in prev["item_id"].unique()}

    new_file = not (out.exists() and out.stat().st_size > 0)
    with open(out, "a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()

        for _, row in df.iterrows():
            item_id = int(row["item_id"])
            if item_id in done_ids:
                continue
            row_cpl = args.complexity_level
            if "complexity_level" in row.index and pd.notna(row["complexity_level"]):
                row_cpl = int(row["complexity_level"])
            print(
                f"[run] target={target_label} dataset={args.dataset} item_id={item_id} "
                f"original_index={row['original_index']} argument_complexity={row_cpl}"
            )
            recs = run_dialogue(
                row,
                dataset=args.dataset,
                target_provider=args.target_provider,
                target_model=args.target_model,
                target_label=target_label,
                persuader_model=args.persuader_model,
                complexity_level=row_cpl,
                persuasion_mode=args.persuasion_mode,
                max_turns=args.max_turns,
                target_temperature=args.target_temperature,
                persuader_temperature=args.persuader_temperature,
                persuader_max_tokens=args.persuader_max_tokens,
                sleep_s=args.sleep_s,
            )
            for rec in recs:
                writer.writerow(rec)
                fp.flush()
            done_ids.add(item_id)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
