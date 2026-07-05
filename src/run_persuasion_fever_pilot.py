"""
Multi-turn persuasion on FEVER (DeepSeek target, OpenRouter persuader).

Target confidence is elicited on a 1–10 scale (same convention as baseline temp06 runs).
Default persuader: openai/gpt-5.4-mini (OpenRouter).

Persuasion modes:
  - until_flip (default): add counterargument rounds until the target changes its label
    or --max-turns is reached (hard cap, default 15).
  - fixed: run exactly --max-turns persuasion rounds regardless of flip.

Example:
  cd src && python3 run_persuasion_fever_pilot.py \\
    --fever-input ../output_wood/dataset_subsampling/fever/csv/fever600_200x3_complexity_wood_v1_lr_40.csv \\
    --max-items 600 --persuasion-mode until_flip --max-turns 15

Requires .env: DEEPSEEK_API_KEY, OPENROUTER_API_KEY
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from collect_conflictqa_ece import (
    DEEPSEEK_API_MODEL,
    FEVER_DEFAULT,
    call_deepseek,
    load_dotenv,
    parse_json_obj,
    read_fever_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "output_wood" / "persuasion" / "DeepSeek" / "fever"

FEVER_REFS = "SUPPORTS; REFUTES; NOT ENOUGH INFO"

PERSUADER_SYSTEM = (
    "You are a persuader in a fact-checking debate. "
    "You only output a single JSON object, no markdown fences, no extra text. "
    'Schema: {"counterargument": string}.'
)

PERSUADER_USER = """Claim: {claim}

The target model's current verdict: {target_answer}
The target's confidence (1–10 scale): {target_confidence}

Your goal: write one counterargument that could reasonably push the target to change its verdict to a *different* label among: SUPPORTS, REFUTES, NOT ENOUGH INFO.
Do not reveal any hidden gold label. Argue from the claim and public-sounding evidence only.

Argument complexity level: {complexity_level} (1=short thesis, 2=add explicit reasoning chain, 3=add concrete evidence details).
{evidence_block}

Output JSON: {{"counterargument": "<your argument>"}}"""

TARGET_HISTORY_TEMPLATE = """Allowed verdict labels (use exactly one in your answer text): SUPPORTS, REFUTES, NOT ENOUGH INFO

Claim: {claim}

Counterarguments so far:
{history}

Re-evaluate the claim after reading the counterarguments. If evidence is mixed, lower confidence.

Output JSON: {{"answer": "<one of the three labels>", "confidence": <integer from 1 to 10>, "decision_explanation": "<short why for this decision>"}}"""

TARGET_SYSTEM_SCALE10 = (
    "You only output a single JSON object, no markdown fences, no extra text. "
    'Schema: {"answer": string, "confidence": number, "decision_explanation": string} where confidence is an integer from 1 to 10 '
    "and means how sure you are that your verdict is correct."
)

CSV_COLUMNS = [
    "dialogue_id",
    "target_model",
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


def normalize_fever_label(text: str) -> str | None:
    s = re.sub(r"\s+", " ", (text or "").strip().upper())
    if not s:
        return None
    if "NOT ENOUGH" in s or s in {"NEI", "NOT ENOUGH INFO"}:
        return "NOT ENOUGH INFO"
    if "REFUTE" in s:
        return "REFUTES"
    if "SUPPORT" in s:
        return "SUPPORTS"
    return None


def labels_differ(a: str, b: str) -> bool:
    na, nb = normalize_fever_label(a), normalize_fever_label(b)
    if na is None or nb is None:
        return normalize_phrase(a) != normalize_phrase(b)
    return na != nb


def clip_confidence_scale10(value: Any) -> int:
    """Map model confidence to integer 1–10 (baseline verbal scale)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1
    if 0.0 < v <= 1.0:
        v *= 10.0
    return int(max(1, min(10, round(v))))


def normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def format_evidence_block(evi: Any, sentence1: Any) -> str:
    parts: list[str] = []
    if evi is not None and str(evi).strip() and str(evi).strip().lower() not in ("nan", "none"):
        parts.append(f"Evidence snippet (use if level 3): {str(evi)[:2000]}")
    if sentence1 is not None and str(sentence1).strip() and str(sentence1).strip().lower() not in ("nan", "none"):
        parts.append(f"Additional context: {str(sentence1)[:1500]}")
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def call_openrouter_json(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], str]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=120.0,
        max_retries=1,
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
            timeout=120.0,
        )
    except Exception:
        r = client.chat.completions.create(**kwargs, timeout=120.0)
    raw = r.choices[0].message.content or "{}"
    return parse_json_obj(raw), raw


def call_persuader(
    *,
    model: str,
    claim: str,
    target_answer: str,
    target_confidence: int,
    complexity_level: int,
    evi: Any,
    sentence1: Any,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str, str]:
    evidence_block = (
        format_evidence_block(evi, sentence1) if complexity_level >= 3 else ""
    )
    user = PERSUADER_USER.format(
        claim=claim,
        target_answer=target_answer,
        target_confidence=target_confidence,
        complexity_level=complexity_level,
        evidence_block=evidence_block,
    )
    data, raw_json = call_openrouter_json(
        model=model,
        system=PERSUADER_SYSTEM,
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return str(data.get("counterargument", "")).strip(), user, raw_json


def call_target_turn0(
    claim: str,
    *,
    model: str,
    temperature: float,
) -> tuple[str, int, str, str, str]:
    return call_target_with_history(claim, [], model=model, temperature=temperature)


def call_target_with_history(
    claim: str,
    history: list[str],
    *,
    model: str,
    temperature: float,
) -> tuple[str, int, str, str, str]:
    if not history:
        numbered = "(none yet)"
    else:
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(history, start=1))
    user = TARGET_HISTORY_TEMPLATE.format(claim=claim, history=numbered)
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TARGET_SYSTEM_SCALE10},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TARGET_SYSTEM_SCALE10},
                {"role": "user", "content": user},
            ],
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


def load_fever_rows(path: Path, max_items: int | None) -> pd.DataFrame:
    base = read_fever_csv(path)
    raw = pd.read_csv(path)
    if "complexity_level" in raw.columns:
        base["task_complexity_level"] = pd.to_numeric(raw["complexity_level"], errors="coerce")
    for col in ("evi", "sentence1", "label"):
        if col in raw.columns:
            base[col] = raw[col]
    if max_items is not None:
        base = base.head(max_items)
    return base


@dataclass
class ContinueState:
    dialogue_id: str
    original_index: int
    claim: str
    gold_label: str
    answer0: str
    current_answer: str
    current_conf: int
    history: list[str]
    transcript: list[dict[str, Any]]
    prior_records: list[dict[str, Any]]


def load_no_flip_after_turn1(prior_csv: Path, *, after_turn: int = 1) -> dict[int, ContinueState]:
    """Dialogues where the model did not flip by `after_turn` in a prior CSV."""
    df = pd.read_csv(prior_csv)
    out: dict[int, ContinueState] = {}
    for oid, sub in df.groupby("original_index"):
        sub = sub.sort_values("turn")
        if after_turn not in set(sub["turn"].astype(int)):
            continue
        t_end = sub[sub["turn"] == after_turn].iloc[-1]
        if int(t_end.get("flipped_from_initial", 0)) != 0:
            continue
        t0 = sub[sub["turn"] == 0].iloc[0]
        history: list[str] = []
        for _, r in sub[sub["turn"] >= 1].iterrows():
            c = str(r.get("counterargument", "")).strip()
            if c:
                history.append(c)
        transcript: list[dict[str, Any]] = []
        raw_transcript = str(t_end.get("conversation_json", "")).strip()
        if raw_transcript:
            try:
                transcript = json.loads(raw_transcript)
            except json.JSONDecodeError:
                transcript = []
        prior_records = sub[sub["turn"] <= after_turn].to_dict(orient="records")
        out[int(oid)] = ContinueState(
            dialogue_id=str(t0["dialogue_id"]),
            original_index=int(oid),
            claim=str(t0["claim"]),
            gold_label=str(t0["gold_label"]),
            answer0=str(t0["answer"]),
            current_answer=str(t_end["answer"]),
            current_conf=int(t_end["confidence"]),
            history=history,
            transcript=transcript,
            prior_records=prior_records,
        )
    return out


def load_credit_error_retry_states(prior_csv: Path) -> dict[int, ContinueState]:
    """Dialogues whose last row ended with an OpenRouter credit / 402 error."""
    df = pd.read_csv(prior_csv)
    out: dict[int, ContinueState] = {}
    for oid, sub in df.groupby("original_index"):
        sub = sub.sort_values("turn")
        last = sub.iloc[-1]
        if str(last.get("stop_reason", "")) != "error":
            continue
        err = str(last.get("error", "")).lower()
        if "402" not in err and "credit" not in err:
            continue
        good = sub[sub["error"].fillna("").astype(str).str.len() == 0]
        if good.empty:
            continue
        prior_records = good.to_dict(orient="records")
        last_good = good.iloc[-1]
        t0 = good[good["turn"] == 0].iloc[0]
        history: list[str] = []
        for _, r in good[good["turn"] >= 1].iterrows():
            c = str(r.get("counterargument", "")).strip()
            if c:
                history.append(c)
        transcript: list[dict[str, Any]] = []
        raw_transcript = str(last_good.get("conversation_json", "")).strip()
        if raw_transcript:
            try:
                transcript = json.loads(raw_transcript)
            except json.JSONDecodeError:
                transcript = []
        out[int(oid)] = ContinueState(
            dialogue_id=str(t0["dialogue_id"]),
            original_index=int(oid),
            claim=str(t0["claim"]),
            gold_label=str(t0["gold_label"]),
            answer0=str(t0["answer"]),
            current_answer=str(last_good["answer"]),
            current_conf=int(last_good["confidence"]),
            history=history,
            transcript=transcript,
            prior_records=prior_records,
        )
    return out


def strip_dialogue_from_csv(csv_path: Path, original_index: int) -> int:
    """Remove one dialogue's rows before rewriting it; return rows removed."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0
    df = pd.read_csv(csv_path)
    before = len(df)
    keep = df[df["original_index"].astype(int) != original_index]
    keep.to_csv(csv_path, index=False)
    return before - len(keep)


def run_dialogue_continue(
    row: pd.Series,
    state: ContinueState,
    *,
    target_model: str,
    persuader_model: str,
    complexity_level: int,
    persuasion_mode: str,
    max_turns: int,
    start_turn: int,
    target_temperature: float,
    persuader_temperature: float,
    persuader_max_tokens: int,
    sleep_s: float,
) -> list[dict[str, Any]]:
    """Append persuasion rounds after a prior run (e.g. continue when no flip at turn 1)."""
    claim = state.claim
    gold = state.gold_label
    oid = state.original_index
    evi = row.get("evi")
    sentence1 = row.get("sentence1")
    dialogue_id = state.dialogue_id

    records: list[dict[str, Any]] = [dict(r) for r in state.prior_records]
    history = list(state.history)
    transcript = list(state.transcript)
    answer0 = state.answer0
    current_answer, current_conf = state.current_answer, state.current_conf
    flip_turn: int | None = None
    for r in state.prior_records:
        ft = r.get("flip_turn", "")
        if ft not in ("", None):
            try:
                flip_turn = int(float(ft))
                break
            except (TypeError, ValueError):
                pass
    until_flip = persuasion_mode == "until_flip"

    for turn in range(start_turn + 1, max_turns + 1):
        time.sleep(sleep_s)
        counter = ""
        persuader_prompt = ""
        persuader_raw = ""
        turn_err = ""
        try:
            counter, persuader_prompt, persuader_raw = call_persuader(
                model=persuader_model,
                claim=claim,
                target_answer=current_answer,
                target_confidence=current_conf,
                complexity_level=complexity_level,
                evi=evi,
                sentence1=sentence1,
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
                new_answer, new_conf, new_expl, target_prompt, target_raw = call_target_with_history(
                    claim,
                    history,
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

        flipped = labels_differ(new_answer, answer0)
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
                dialogue_id=dialogue_id,
                target_model=target_model,
                persuader_model=persuader_model,
                oid=oid,
                turn=turn,
                claim=claim,
                gold_label=gold,
                answer=new_answer,
                confidence=new_conf,
                initial=answer0,
                flip_turn=flip_turn,
                counterargument=counter,
                decision_explanation=new_expl,
                target_prompt=target_prompt,
                target_raw_json=target_raw,
                persuader_prompt=persuader_prompt,
                persuader_raw_json=persuader_raw,
                conversation_json=json.dumps(transcript, ensure_ascii=False),
                persuasion_mode=persuasion_mode,
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


def run_dialogue(
    row: pd.Series,
    *,
    target_model: str,
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
    gold = str(row.get("label", row["ground_truth"][0] if isinstance(row.get("ground_truth"), list) else ""))
    oid = int(row["original_index"])
    evi = row.get("evi")
    sentence1 = row.get("sentence1")
    dialogue_id = str(uuid.uuid4())

    records: list[dict[str, Any]] = []
    history: list[str] = []
    transcript: list[dict[str, Any]] = []
    err = ""

    try:
        answer0, conf0, expl0, target_prompt0, target_raw0 = call_target_turn0(
            claim, model=target_model, temperature=target_temperature
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
            dialogue_id=dialogue_id,
            target_model=target_model,
            persuader_model=persuader_model,
            oid=oid,
            turn=0,
            claim=claim,
            gold_label=gold,
            answer=answer0,
            confidence=conf0,
            initial=answer0,
            flip_turn=None,
            counterargument="",
            decision_explanation=expl0,
            target_prompt=target_prompt0,
            target_raw_json=target_raw0,
            persuader_prompt="",
            persuader_raw_json="",
            conversation_json=json.dumps(transcript, ensure_ascii=False),
            persuasion_mode=persuasion_mode,
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
            counter, persuader_prompt, persuader_raw = call_persuader(
                model=persuader_model,
                claim=claim,
                target_answer=current_answer,
                target_confidence=current_conf,
                complexity_level=complexity_level,
                evi=evi,
                sentence1=sentence1,
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
                new_answer, new_conf, new_expl, target_prompt, target_raw = call_target_with_history(
                    claim,
                    history,
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

        flipped = labels_differ(new_answer, answer0)
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
                dialogue_id=dialogue_id,
                target_model=target_model,
                persuader_model=persuader_model,
                oid=oid,
                turn=turn,
                claim=claim,
                gold_label=gold,
                answer=new_answer,
                confidence=new_conf,
                initial=answer0,
                flip_turn=flip_turn,
                counterargument=counter,
                decision_explanation=new_expl,
                target_prompt=target_prompt,
                target_raw_json=target_raw,
                persuader_prompt=persuader_prompt,
                persuader_raw_json=persuader_raw,
                conversation_json=json.dumps(transcript, ensure_ascii=False),
                persuasion_mode=persuasion_mode,
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


def _row(
    *,
    dialogue_id: str,
    target_model: str,
    persuader_model: str,
    oid: int,
    turn: int,
    claim: str,
    gold_label: str,
    answer: str,
    confidence: int,
    initial: str,
    flip_turn: int | None,
    counterargument: str,
    decision_explanation: str,
    target_prompt: str,
    target_raw_json: str,
    persuader_prompt: str,
    persuader_raw_json: str,
    conversation_json: str,
    persuasion_mode: str,
    stop_reason: str,
    error: str,
) -> dict[str, Any]:
    return {
        "dialogue_id": dialogue_id,
        "target_model": target_model,
        "persuader_model": persuader_model,
        "dataset": "fever",
        "original_index": oid,
        "turn": turn,
        "claim": claim,
        "gold_label": gold_label,
        "answer": answer,
        "confidence": confidence,
        "confidence_scale": "1-10",
        "flipped_from_initial": 1 if labels_differ(answer, initial) else 0,
        "flip_turn": flip_turn if flip_turn is not None else "",
        "counterargument": counterargument,
        "decision_explanation": decision_explanation,
        "target_prompt": target_prompt,
        "target_raw_json": target_raw_json,
        "persuader_prompt": persuader_prompt,
        "persuader_raw_json": persuader_raw_json,
        "conversation_json": conversation_json,
        "persuasion_mode": persuasion_mode,
        "stop_reason": stop_reason,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fever-input", type=Path, default=FEVER_DEFAULT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument(
        "--persuasion-mode",
        choices=("until_flip", "fixed"),
        default="until_flip",
        help="until_flip: stop when label changes; fixed: always run --max-turns rounds.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Hard cap on persuasion rounds (until_flip) or exact round count (fixed).",
    )
    parser.add_argument("--complexity-level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--persuader-model",
        default="openai/gpt-5.4-mini",
        help=(
            "OpenRouter persuader model (default: openai/gpt-5.4-mini). "
            "Alternatives: openai/gpt-4.1-mini, openai/gpt-4.1, openai/gpt-5.5."
        ),
    )
    parser.add_argument("--target-model", default=DEEPSEEK_API_MODEL)
    parser.add_argument("--target-temperature", type=float, default=0.6)
    parser.add_argument("--persuader-temperature", type=float, default=0.8)
    parser.add_argument(
        "--persuader-max-tokens",
        type=int,
        default=1024,
        help="Cap OpenRouter output tokens (avoids huge default max_tokens / 402 errors).",
    )
    parser.add_argument("--sleep-s", type=float, default=0.3)
    parser.add_argument(
        "--filter-complexity-level",
        type=int,
        default=None,
        choices=(1, 2, 3),
        help="Only run FEVER rows with this Wood task complexity_level 1/2/3 (optional)",
    )
    parser.add_argument(
        "--continue-from-csv",
        type=Path,
        default=None,
        help="Prior rollout CSV; only continue dialogues with no flip by --continue-after-turn.",
    )
    parser.add_argument(
        "--continue-after-turn",
        type=int,
        default=1,
        help="With --continue-from-csv: require no flip through this persuasion turn (default 1).",
    )
    parser.add_argument(
        "--retry-credit-errors-from-csv",
        type=Path,
        default=None,
        help="Retry dialogues that ended with 402/credit errors; strips their rows from --out first.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    for key in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
        if not os.getenv(key):
            raise SystemExit(f"Missing {key} in environment / .env")

    df = load_fever_rows(args.fever_input, None)
    if args.filter_complexity_level is not None and "task_complexity_level" in df.columns:
        df = df[df["task_complexity_level"] == args.filter_complexity_level]
    if args.max_items is not None:
        df = df.head(args.max_items)

    if args.continue_from_csv is not None or args.persuasion_mode == "until_flip":
        default_name = "expl.csv"
    else:
        default_name = f"fixed_t{args.max_turns}_expl.csv"
    out = args.out or (OUT_DIR / "csv" / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[int] = set()
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out)
        if "original_index" in prev.columns:
            done_ids = {int(x) for x in prev["original_index"].unique()}

    continue_states: dict[int, ContinueState] = {}
    start_turn_by_oid: dict[int, int] = {}
    if args.retry_credit_errors_from_csv is not None:
        continue_states = load_credit_error_retry_states(args.retry_credit_errors_from_csv)
        df = df[df["original_index"].astype(int).isin(continue_states.keys())]
        for oid, st in continue_states.items():
            if st.prior_records:
                start_turn_by_oid[oid] = int(max(int(r["turn"]) for r in st.prior_records))
            else:
                start_turn_by_oid[oid] = 0
        print(
            f"[retry] from {args.retry_credit_errors_from_csv.name}: "
            f"{len(continue_states)} dialogues with credit/402 errors"
        )
    elif args.continue_from_csv is not None:
        continue_states = load_no_flip_after_turn1(
            args.continue_from_csv, after_turn=args.continue_after_turn
        )
        df = df[df["original_index"].astype(int).isin(continue_states.keys())]
        for oid in continue_states:
            start_turn_by_oid[oid] = args.continue_after_turn
        print(
            f"[continue] from {args.continue_from_csv.name}: "
            f"{len(continue_states)} dialogues with no flip by turn {args.continue_after_turn}"
        )

    retry_oids = set(continue_states.keys()) if args.retry_credit_errors_from_csv else set()
    if retry_oids:
        done_ids -= retry_oids

    new_file = not (out.exists() and out.stat().st_size > 0)
    with open(out, "a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()

        for _, row in df.iterrows():
            oid = int(row["original_index"])
            if oid in done_ids:
                continue
            row_cpl = args.complexity_level
            if "task_complexity_level" in row.index and pd.notna(row["task_complexity_level"]):
                row_cpl = int(row["task_complexity_level"])
            print(f"[run] fever idx={oid} argument_complexity={row_cpl}")
            if oid in retry_oids and out.exists():
                removed = strip_dialogue_from_csv(out, oid)
                if removed:
                    print(f"[retry] stripped {removed} old rows for idx={oid}")
            if oid in continue_states:
                recs = run_dialogue_continue(
                    row,
                    continue_states[oid],
                    target_model=args.target_model,
                    persuader_model=args.persuader_model,
                    complexity_level=row_cpl,
                    persuasion_mode=args.persuasion_mode,
                    max_turns=args.max_turns,
                    start_turn=start_turn_by_oid.get(oid, args.continue_after_turn),
                    target_temperature=args.target_temperature,
                    persuader_temperature=args.persuader_temperature,
                    persuader_max_tokens=args.persuader_max_tokens,
                    sleep_s=args.sleep_s,
                )
            else:
                recs = run_dialogue(
                    row,
                    target_model=args.target_model,
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
            done_ids.add(oid)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
