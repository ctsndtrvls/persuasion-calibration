"""
Run chat models on ConflictQA (popQA + strategyQA), optional FEVER / DebateQA subsets, collect JSON answers
with self-reported confidence, score correctness vs ground_truth, and write CSV for ECE plots.

Uses API keys from project .env:
  OPENAI_API_KEY, CLAUDE_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY

Example:
  cd src && python3 collect_conflictqa_ece.py --max-items 20 --plot

Full run (cost + time):
  python3 collect_conflictqa_ece.py --plot

Without Gemini (key not required; run later without this flag to append missing rows):
  python3 collect_conflictqa_ece.py --skip-gemini --plot

Without Anthropic (no credits / Claude not needed):
  python3 collect_conflictqa_ece.py --skip-anthropic --skip-gemini

OpenAI-only on ConflictQA subsets (CSV, same 480×2 as Wood resplit), temp 0.6 + 1–10 scale,
separate combined + per-split CSV and a suffixed plot:
  python3 collect_conflictqa_ece.py --datasets both --skip-anthropic --skip-gemini --skip-deepseek \\
    --popqa-input output_wood/conflictqa_popqa480_160x3_wood_v2_resplit.csv \\
    --strategyqa-input output_wood/conflictqa_strategyqa480_160x3_wood_v2_resplit.csv \\
    --out output_wood/self_reported_confidence/csv/conflictqa_ece_openai_conflictqa_temp06_scale10.csv \\
    --also-split-dataset-csv output_wood/self_reported_confidence/csv/conflictqa_ece_openai_temp06_scale10 \\
    --openai-temperature 0.6 --openai-confidence-scale 1-10 \\
    --plot --plot-file-suffix temp06_scale10_conflictqa

Default confidence is the model's stated P(correct) in JSON (verbal calibration).
For OpenAI you can switch to token-probability confidence (softmax/logprob based) with:
  --openai-confidence-source token-prob

Temperature 0.6 + verbal 1–10 scale + default prompt (rollout CSVs under output_wood/self_reported_confidence/csv/…):
  OpenAI + DeepSeek on DebateQA only — see scripts/run_temp06_scale10_debateqa_openai_deepseek.sh
  Qwen + Gemma on FEVER + PopQA + DebateQA (OpenRouter) — see scripts/run_temp06_scale10_qwen_gemma_fever_popqa_debateqa.sh

OpenAI + DeepSeek token-prob on DebateQA only (no Claude/Gemini):
  python3 collect_conflictqa_ece.py --datasets debateqa --skip-anthropic --skip-gemini \\
    --openai-confidence-source token-prob --openai-confidence-scale 0-1 --openai-temperature 0.6 \\
    --deepseek-confidence-source token-prob --deepseek-confidence-scale 0-1 --deepseek-temperature 0.6 \\
    --out output_wood/self_reported_confidence/csv/debateqa_ece_openai_deepseek_temp06_tokenprob.csv

Qwen + Gemma (OpenRouter token-prob) on FEVER + ConflictQA popqa + DebateQA:
  python3 collect_conflictqa_ece.py --datasets fever_popqa_debateqa --use-conflictqa-subset-csv \\
    --skip-openai --skip-anthropic --skip-deepseek --skip-gemini \\
    --extra-openrouter-tokenprob-model qwen/qwen3-14b \\
    --extra-openrouter-tokenprob-model google/gemma-4-26b-a4b-it \\
    --extra-openrouter-temperature 0.6 \\
    --out output_wood/self_reported_confidence/csv/qwen_gemma_fpd_temp06_tokenprob.csv

Anthropic/Gemini: old IDs (claude-3-5-sonnet-20241022, gemini-1.5-flash-8b) return 404.
Defaults are claude-sonnet-4-6 and gemini-2.5-flash; see --anthropic-model / --gemini-model.

After ID migration in the model column, old failed rows do not block new calls.
Use --redo-model when you need to refresh the same model ID.

Each completed row is appended to CSV and flushed immediately. On interruption (Ctrl+C),
only the in-flight API call can be lost; already written rows remain in the file.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from linguistic_confidence import (
    detect_linguistic_markers,
    linguistic_confidence_level,
    linguistic_confidence_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "conflictqa"
OUT_DIR = PROJECT_ROOT / "output_wood"
# All rollout CSVs from this script (self-reported or token-prob) should live here by convention.
SELF_REPORTED_CONFIDENCE_DIR = OUT_DIR / "self_reported_confidence"
POPQA = DATA_DIR / "conflictQA-popQA-llama2-7b.json"
STRATEGY = DATA_DIR / "conflictQA-strategyQA-llama2-7b.json"
FEVER_DEFAULT = OUT_DIR / "dataset_subsampling" / "fever" / "csv" / "fever480_160x3_complexity_wood_v1_lr_40_resplit.csv"
POPQA_SUBSET_DEFAULT = OUT_DIR / "dataset_subsampling" / "conflictqa" / "csv" / "conflictqa_popqa480_160x3_wood_v2_resplit.csv"
STRATEGY_SUBSET_DEFAULT = (
    OUT_DIR / "dataset_subsampling" / "conflictqa" / "csv" / "conflictqa_strategyqa480_160x3_wood_v2_resplit.csv"
)
DEBATEQA_DEFAULT = OUT_DIR / "dataset_subsampling" / "debateqa" / "csv" / "debateqa_480_160x3.csv"

# Canonical ids written to CSV (must match plot_ece_calibration.MODEL_IDS).
# Anthropic: snapshot claude-3-5-sonnet-20241022 was removed from the API (404).
# Gemini: 2.0-flash may be unavailable to new API users; prefer 2.5-flash.
MODELS: list[tuple[str, str]] = [
    ("openai/gpt-4o-2024-11-20", "openai"),
    ("anthropic/claude-sonnet-4-6", "anthropic"),
    ("deepseek/deepseek-chat-v2.5", "deepseek"),
    ("google/gemini-2.5-flash", "google"),
]

# Provider API model names (override with --openai-model / --anthropic-model / --gemini-model)
OPENAI_API_MODEL = "gpt-4o-2024-11-20"
ANTHROPIC_API_MODEL = "claude-sonnet-4-6"
DEEPSEEK_API_MODEL = "deepseek-chat"
GEMINI_API_MODEL = "gemini-2.5-flash"

SYSTEM = (
    "You only output a single JSON object, no markdown fences, no extra text. "
    'Schema: {"answer": string, "confidence": number} where confidence is in [0,1] '
    "and means your subjective probability that your short answer matches one of the reference forms."
)
SYSTEM_TOKENPROB = (
    "You only output a single JSON object, no markdown fences, no extra text. "
    'Schema: {"answer": string}.'
)

USER_TEMPLATE = """Reference answers (any equivalent labeling of the same entity is acceptable): {refs}

Question: {question}

Output JSON: {{"answer": "<short phrase>", "confidence": <number from 0 to 1>}}"""

USER_TEMPLATE_OPENAI_SCALE10 = """Reference answers (any equivalent labeling of the same entity is acceptable): {refs}

Question: {question}

It is okay to be unsure. If evidence is mixed or incomplete, use lower confidence.
Avoid overconfidence when the claim is ambiguous.

Output JSON: {{"answer": "<short phrase>", "confidence": <integer from 1 to 10>}}"""

USER_TEMPLATE_OPENAI_TOKENPROB = """Reference answers (any equivalent labeling of the same entity is acceptable): {refs}

Question: {question}

Output JSON: {{"answer": "<short phrase>"}}"""

USER_TEMPLATE_EPISTEMIC_MARKER = """Reference answers (any equivalent labeling of the same entity is acceptable): {refs}

Question: {question}

When responding, include exactly ONE epistemic marker phrase in your answer text
to express your uncertainty (e.g., "probably", "likely", "might be", "not sure").
Do not include more than one marker.

Output JSON: {{"answer": "<short phrase with exactly one epistemic marker>", "confidence": <number from 0 to 1>}}"""

USER_TEMPLATE_EPISTEMIC_MARKER_SCALE10 = """Reference answers (any equivalent labeling of the same entity is acceptable): {refs}

Question: {question}

When responding, include exactly ONE epistemic marker phrase in your answer text
to express your uncertainty (e.g., "probably", "likely", "might be", "not sure").
Do not include more than one marker.

Output JSON: {{"answer": "<short phrase with exactly one epistemic marker>", "confidence": <integer from 1 to 10>}}"""

# Anthropic: forced tool call avoids empty/non-JSON text (common with Sonnet 4 + thinking).
ANTHROPIC_CALIBRATION_TOOL: dict[str, Any] = {
    "name": "submit_calibration",
    "description": (
        "Submit your short answer and your subjective probability in [0,1] that it "
        "matches one of the reference answer forms."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Short phrase; equivalent labeling of the same entity is acceptable.",
            },
            "confidence": {
                "type": "number",
                "description": "In [0,1]: how likely your answer matches a reference.",
            },
        },
        "required": ["answer", "confidence"],
    },
}

SYSTEM_ANTHROPIC_TOOL = (
    "You must call submit_calibration exactly once with fields answer (string) and "
    "confidence (number in [0,1]). No other text is required."
)
SYSTEM_ANTHROPIC_TOOL_SCALE10 = (
    "You must call submit_calibration exactly once with fields answer (string) and "
    "confidence (integer from 1 to 10). It is okay to be unsure; avoid overconfidence when evidence is mixed. "
    "No other text is required."
)


class MissingLogprobsError(RuntimeError):
    pass


def _clip_confidence(x: Any) -> float:
    try:
        return float(max(0.0, min(1.0, float(x))))
    except (TypeError, ValueError):
        return 0.0


ROLLUP_CSV_COLUMNS = [
    "model",
    "dataset",
    "original_index",
    "question",
    "answer",
    "confidence",
    "correct",
    "error",
    # Linguistic uncertainty features derived from epistemic markers.
    "ling_conf_level",
    "ling_conf_score",
    "ling_marker_count",
    "ling_unique_marker_count",
    "ling_has_marker",
]


class IncrementalCsvWriter:
    """Append rows one-by-one with flush so progress survives Ctrl+C (except in-flight API call)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not (path.exists() and path.stat().st_size > 0)
        self._fp = open(path, "a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._fp,
            fieldnames=ROLLUP_CSV_COLUMNS,
            extrasaction="ignore",
        )
        if new_file:
            self._writer.writeheader()
            self._fp.flush()

    def append_row(self, row: dict[str, Any]) -> None:
        out = {k: row.get(k, "") for k in ROLLUP_CSV_COLUMNS}
        self._writer.writerow(out)
        self._fp.flush()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> IncrementalCsvWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


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
        if key and key not in os.environ:
            os.environ[key] = val


def normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_correct(answer: str, ground_truth: list[Any]) -> int:
    a = normalize_phrase(answer)
    if len(a) < 2:
        return 0
    for gt in ground_truth:
        g = normalize_phrase(str(gt))
        if len(g) < 2:
            continue
        if g in a or a in g:
            return 1
    return 0


def parse_json_obj(text: str) -> dict[str, Any]:
    """Parse a single JSON object from model output (strip fences, tolerate preamble)."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty response text")
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw).strip()
    candidates: list[str] = []
    if raw:
        candidates.append(raw)
    start = raw.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start : i + 1])
                    break
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON object in model text (preview {raw[:280]!r}…)")


def _build_user_prompt(
    question: str,
    refs: str,
    *,
    use_scale10: bool,
    confidence_source: str,
    elicitation_mode: str,
) -> str:
    if confidence_source == "token-prob":
        return USER_TEMPLATE_OPENAI_TOKENPROB.format(question=question, refs=refs)
    if elicitation_mode == "epistemic-marker":
        tmpl = USER_TEMPLATE_EPISTEMIC_MARKER_SCALE10 if use_scale10 else USER_TEMPLATE_EPISTEMIC_MARKER
        return tmpl.format(question=question, refs=refs)
    return (
        USER_TEMPLATE_OPENAI_SCALE10.format(question=question, refs=refs)
        if use_scale10
        else USER_TEMPLATE.format(question=question, refs=refs)
    )


def call_openai(
    question: str,
    refs: str,
    model_name: str,
    *,
    temperature: float,
    use_scale10: bool,
    confidence_source: str,
    elicitation_mode: str,
) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user = _build_user_prompt(
        question,
        refs,
        use_scale10=use_scale10,
        confidence_source=confidence_source,
        elicitation_mode=elicitation_mode,
    )

    request_kwargs: dict[str, Any] = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_TOKENPROB if confidence_source == "token-prob" else SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    if confidence_source == "token-prob":
        request_kwargs["logprobs"] = True
    r = client.chat.completions.create(**request_kwargs)
    raw = r.choices[0].message.content or "{}"
    data = parse_json_obj(raw)
    answer = str(data.get("answer", ""))
    if confidence_source == "token-prob":
        lp = getattr(r.choices[0], "logprobs", None)
        content = getattr(lp, "content", None) if lp is not None else None
        if not content:
            raise ValueError("OpenAI response does not contain logprobs content for token-prob confidence")
        token_logps: list[float] = []
        for tok in content:
            val = getattr(tok, "logprob", None)
            if val is None:
                continue
            try:
                token_logps.append(float(val))
            except (TypeError, ValueError):
                continue
        if not token_logps:
            raise ValueError("No usable token logprobs in OpenAI response")
        conf = math.exp(sum(token_logps) / len(token_logps))
        return answer, _clip_confidence(conf)
    conf = float(data.get("confidence", 0.0))
    if use_scale10:
        conf = conf / 10.0
    return answer, conf


def _tokenprob_confidence_from_chat_choice(choice: Any, provider_name: str) -> float:
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        raise ValueError(f"{provider_name} response does not contain logprobs content for token-prob confidence")
    token_logps: list[float] = []
    for tok in content:
        val = getattr(tok, "logprob", None)
        if val is None:
            continue
        try:
            token_logps.append(float(val))
        except (TypeError, ValueError):
            continue
    if not token_logps:
        raise ValueError(f"No usable token logprobs in {provider_name} response")
    conf = math.exp(sum(token_logps) / len(token_logps))
    return _clip_confidence(conf)


def call_openrouter_tokenprob_chat(
    question: str,
    refs: str,
    model_name: str,
    *,
    temperature: float,
    provider_name: str,
    elicitation_mode: str,
) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=90.0,
        max_retries=1,
    )
    user = _build_user_prompt(
        question,
        refs,
        use_scale10=False,
        confidence_source="token-prob",
        elicitation_mode=elicitation_mode,
    )
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_TOKENPROB},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "logprobs": True,
    }
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            try:
                r = client.chat.completions.create(
                    **kwargs,
                    response_format={"type": "json_object"},
                    timeout=90.0,
                )
            except Exception:
                r = client.chat.completions.create(**kwargs, timeout=90.0)
            raw = r.choices[0].message.content or "{}"
            data = parse_json_obj(raw)
            answer = str(data.get("answer", ""))
            conf = _tokenprob_confidence_from_chat_choice(r.choices[0], provider_name)
            return answer, conf
        except Exception as e:
            last_err = e
            if "does not contain logprobs content" in str(e) and attempt < 2:
                # OpenRouter may route to providers where logprobs are intermittently absent.
                time.sleep(0.6 * (attempt + 1))
                continue
            break
    if last_err is not None and "does not contain logprobs content" in str(last_err):
        raise MissingLogprobsError(str(last_err))
    raise RuntimeError(str(last_err) if last_err is not None else "unknown OpenRouter error")


def call_openrouter_selfreported_chat(
    question: str,
    refs: str,
    model_name: str,
    *,
    temperature: float,
    use_scale10: bool,
    elicitation_mode: str,
) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=90.0,
        max_retries=1,
    )
    user = _build_user_prompt(
        question,
        refs,
        use_scale10=use_scale10,
        confidence_source="self-reported",
        elicitation_mode=elicitation_mode,
    )
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    try:
        r = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
            timeout=90.0,
        )
    except Exception:
        r = client.chat.completions.create(**kwargs, timeout=90.0)
    raw = r.choices[0].message.content or "{}"
    data = parse_json_obj(raw)
    answer = str(data.get("answer", ""))
    conf = float(data.get("confidence", 0.0))
    if use_scale10:
        conf = conf / 10.0
    return answer, _clip_confidence(conf)


def normalize_openrouter_model_id(model_name: str) -> str:
    """
    Normalize common user-entered model IDs to canonical OpenRouter slugs.
    """
    m = model_name.strip()
    low = m.lower()
    alias_map = {
        "qwen/qwen3-14b": "qwen/qwen3-14b",
        "qwen/qwen3-14B": "qwen/qwen3-14b",
        "Qwen/Qwen3-14B": "qwen/qwen3-14b",
        "google/gemma-4-26b-a4b-it": "google/gemma-4-26b-a4b-it",
        "google/gemma-4-26B-A4B-it": "google/gemma-4-26b-a4b-it",
    }
    if m in alias_map:
        return alias_map[m]
    return low


def call_anthropic(
    question: str,
    refs: str,
    model_name: str,
    *,
    temperature: float,
    use_scale10: bool,
    elicitation_mode: str,
) -> tuple[str, float]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
    user = _build_user_prompt(
        question,
        refs,
        use_scale10=use_scale10,
        confidence_source="self-reported",
        elicitation_mode=elicitation_mode,
    )
    msg = client.messages.create(
        model=model_name,
        max_tokens=1024,
        system=SYSTEM_ANTHROPIC_TOOL_SCALE10 if use_scale10 else SYSTEM_ANTHROPIC_TOOL,
        messages=[{"role": "user", "content": user}],
        tools=[ANTHROPIC_CALIBRATION_TOOL],
        tool_choice={"type": "tool", "name": "submit_calibration"},
        temperature=temperature,
    )
    for b in msg.content:
        if getattr(b, "type", None) == "tool_use" and getattr(b, "name", None) == "submit_calibration":
            inp = getattr(b, "input", None)
            if isinstance(inp, dict):
                conf = float(inp.get("confidence", 0.0))
                if use_scale10:
                    conf = conf / 10.0
                return str(inp.get("answer", "")), _clip_confidence(conf)
    text = ""
    for b in msg.content:
        if getattr(b, "type", None) == "text":
            text += getattr(b, "text", "") or ""
    text = text.strip()
    if not text:
        raise ValueError(
            "Anthropic returned no submit_calibration tool_use and no text "
            f"(stop_reason={getattr(msg, 'stop_reason', None)!r})"
        )
    data = parse_json_obj(text)
    conf = float(data.get("confidence", 0.0))
    if use_scale10:
        conf = conf / 10.0
    return str(data.get("answer", "")), _clip_confidence(conf)


def call_deepseek(
    question: str,
    refs: str,
    model_name: str,
    *,
    temperature: float,
    use_scale10: bool,
    confidence_source: str,
    elicitation_mode: str,
) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    user = _build_user_prompt(
        question,
        refs,
        use_scale10=use_scale10,
        confidence_source=confidence_source,
        elicitation_mode=elicitation_mode,
    )
    kwargs: dict[str, Any] = dict(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_TOKENPROB if confidence_source == "token-prob" else SYSTEM,
            },
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    if confidence_source == "token-prob":
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5
    try:
        r = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception:
        r = client.chat.completions.create(**kwargs)
    raw = r.choices[0].message.content or "{}"
    data = parse_json_obj(raw)
    answer = str(data.get("answer", ""))
    if confidence_source == "token-prob":
        conf = _tokenprob_confidence_from_chat_choice(r.choices[0], "DeepSeek")
        return answer, conf
    conf = float(data.get("confidence", 0.0))
    if use_scale10:
        conf = conf / 10.0
    return answer, conf


def call_gemini(
    question: str,
    refs: str,
    model_name: str,
    *,
    temperature: float,
    use_scale10: bool,
    elicitation_mode: str,
) -> tuple[str, float]:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name,
        system_instruction=SYSTEM,
    )
    user = _build_user_prompt(
        question,
        refs,
        use_scale10=use_scale10,
        confidence_source="self-reported",
        elicitation_mode=elicitation_mode,
    )
    try:
        r = model.generate_content(
            user,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
    except Exception:
        r = model.generate_content(
            user,
            generation_config=genai.GenerationConfig(temperature=temperature),
        )
    raw = r.text or "{}"
    data = parse_json_obj(raw)
    conf = float(data.get("confidence", 0.0))
    if use_scale10:
        conf = conf / 10.0
    return str(data.get("answer", "")), conf


def make_callers(
    openai_model: str,
    anthropic_model: str,
    deepseek_model: str,
    gemini_model: str,
    openai_temperature: float,
    openai_fever_temperature: float | None,
    openai_confidence_scale: str,
    openai_confidence_source: str,
    anthropic_temperature: float,
    anthropic_confidence_scale: str,
    anthropic_confidence_source: str,
    deepseek_temperature: float,
    deepseek_confidence_scale: str,
    deepseek_confidence_source: str,
    gemini_temperature: float,
    gemini_confidence_scale: str,
    gemini_confidence_source: str,
    use_openrouter_for_nonopenai: bool,
    elicitation_mode: str,
    extra_openrouter_tokenprob_models: list[str],
    extra_openrouter_selfreported_models: list[str],
    extra_openrouter_temperature: float,
    extra_openrouter_confidence_scale: str,
) -> dict[str, Callable[[str, str], tuple[str, float]]]:
    def _openai_caller(q: str, r: str, dataset_tag: str) -> tuple[str, float]:
        t = openai_fever_temperature if dataset_tag == "fever" and openai_fever_temperature is not None else openai_temperature
        return call_openai(
            q,
            r,
            openai_model,
            temperature=t,
            use_scale10=(openai_confidence_scale == "1-10"),
            confidence_source=openai_confidence_source,
            elicitation_mode=elicitation_mode,
        )

    callers: dict[str, Callable[[str, str], tuple[str, float]]] = {
        "openai": _openai_caller,
        "anthropic": lambda q, r, _d: call_openrouter_tokenprob_chat(
            q,
            r,
            anthropic_model,
            temperature=anthropic_temperature,
            provider_name="OpenRouter/Anthropic",
            elicitation_mode=elicitation_mode,
        )
        if use_openrouter_for_nonopenai and anthropic_confidence_source == "token-prob"
        else call_anthropic(
            q,
            r,
            anthropic_model,
            temperature=anthropic_temperature,
            use_scale10=(anthropic_confidence_scale == "1-10"),
            elicitation_mode=elicitation_mode,
        ),
        "deepseek": lambda q, r, _d: call_openrouter_tokenprob_chat(
            q,
            r,
            deepseek_model,
            temperature=deepseek_temperature,
            provider_name="OpenRouter/DeepSeek",
            elicitation_mode=elicitation_mode,
        )
        if use_openrouter_for_nonopenai and deepseek_confidence_source == "token-prob"
        else call_deepseek(
            q,
            r,
            deepseek_model,
            temperature=deepseek_temperature,
            use_scale10=(deepseek_confidence_scale == "1-10"),
            confidence_source=deepseek_confidence_source,
            elicitation_mode=elicitation_mode,
        ),
        "google": lambda q, r, _d: call_openrouter_tokenprob_chat(
            q,
            r,
            gemini_model,
            temperature=gemini_temperature,
            provider_name="OpenRouter/Gemini",
            elicitation_mode=elicitation_mode,
        )
        if use_openrouter_for_nonopenai and gemini_confidence_source == "token-prob"
        else call_gemini(
            q,
            r,
            gemini_model,
            temperature=gemini_temperature,
            use_scale10=(gemini_confidence_scale == "1-10"),
            elicitation_mode=elicitation_mode,
        ),
    }

    for raw_model_name in extra_openrouter_tokenprob_models:
        model_name = normalize_openrouter_model_id(raw_model_name)
        provider_key = f"openrouter_extra::{model_name}"
        callers[provider_key] = lambda q, r, _d, m=model_name: call_openrouter_tokenprob_chat(
            q,
            r,
            m,
            temperature=extra_openrouter_temperature,
            provider_name=f"OpenRouter/{m}",
            elicitation_mode=elicitation_mode,
        )
    for raw_model_name in extra_openrouter_selfreported_models:
        model_name = normalize_openrouter_model_id(raw_model_name)
        provider_key = f"openrouter_selfreported::{model_name}"
        callers[provider_key] = lambda q, r, _d, m=model_name: call_openrouter_selfreported_chat(
            q,
            r,
            m,
            temperature=extra_openrouter_temperature,
            use_scale10=(extra_openrouter_confidence_scale == "1-10"),
            elicitation_mode=elicitation_mode,
        )
    return callers


def _parse_ground_truth_cell(val: Any) -> list[Any]:
    if isinstance(val, list):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]
    except (ValueError, SyntaxError):
        pass
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return list(v)
        if isinstance(v, str):
            return [v]
    except (json.JSONDecodeError, TypeError):
        pass
    return [s]


def read_conflictqa_csv(path: Path) -> pd.DataFrame:
    """Subset CSV with question + ground_truth list (string or list) + original_index."""
    df = pd.read_csv(path)
    if "question" not in df.columns:
        raise ValueError(f"ConflictQA CSV must contain 'question': {path}")
    if "ground_truth" not in df.columns:
        raise ValueError(f"ConflictQA CSV must contain 'ground_truth': {path}")
    if "original_index" not in df.columns:
        df = df.copy()
        df["original_index"] = range(len(df))
    out = pd.DataFrame(
        {
            "question": df["question"].fillna("").astype(str),
            "ground_truth": df["ground_truth"].map(_parse_ground_truth_cell),
            "original_index": pd.to_numeric(df["original_index"], errors="coerce").fillna(0).astype(int),
        }
    )
    return out


SPLIT_CSV_TAGS: dict[str, str] = {
    "conflictqa_popqa": "popqa",
    "conflictqa_strategyqa": "strategyqa",
    "fever": "fever",
    "debateqa": "debateqa",
}


def conflictqa_split_csv_path(stem: Path, dataset_tag: str) -> Path | None:
    suffix = SPLIT_CSV_TAGS.get(dataset_tag)
    if suffix is None:
        return None
    base = stem.with_suffix("") if stem.suffix.lower() == ".csv" else stem
    return base.parent / f"{base.name}_{suffix}.csv"


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["original_index"] = range(len(df))
    return df


def read_fever_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"FEVER CSV must contain 'label' column: {path}")
    if "original_index" not in df.columns:
        df = df.copy()
        df["original_index"] = range(len(df))
    claim_col = "claim" if "claim" in df.columns else "question"
    if claim_col not in df.columns:
        raise ValueError(f"FEVER CSV must contain 'claim' or 'question' column: {path}")
    out = pd.DataFrame(
        {
            "question": df[claim_col].fillna("").astype(str),
            "ground_truth": df["label"].fillna("").astype(str).map(lambda x: [x]),
            "original_index": pd.to_numeric(df["original_index"], errors="coerce").fillna(0).astype(int),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=SELF_REPORTED_CONFIDENCE_DIR / "conflictqa_ece_rollout.csv",
        help=f"Output CSV (must be under {SELF_REPORTED_CONFIDENCE_DIR.relative_to(PROJECT_ROOT)} unless --allow-any-out-path).",
    )
    parser.add_argument(
        "--allow-any-out-path",
        action="store_true",
        help="Disable check that --out lies under output_wood/self_reported_confidence/ (for rare legacy paths).",
    )
    parser.add_argument("--sleep-s", type=float, default=0.25, help="Pause between API calls")
    parser.add_argument("--max-items", type=int, default=None, help="Cap rows per dataset (for tests)")
    parser.add_argument(
        "--datasets",
        choices=("both", "popqa", "strategyqa", "fever", "all", "debateqa", "fever_popqa_debateqa"),
        default="both",
        help=(
            "Dataset selection: both=ConflictQA popqa+strategyqa; all=both+FEVER; "
            "debateqa=DebateQA CSV only; fever_popqa_debateqa=FEVER + ConflictQA popqa + DebateQA (no strategyqa)."
        ),
    )
    parser.add_argument(
        "--fever-input",
        type=Path,
        default=FEVER_DEFAULT,
        help="Path to FEVER CSV (default: output_wood/fever480_160x3_complexity_wood_v1_lr_40_resplit.csv)",
    )
    parser.add_argument(
        "--popqa-input",
        type=Path,
        default=None,
        help="Optional CSV for ConflictQA popQA (question, ground_truth, original_index). Default: full JSONL.",
    )
    parser.add_argument(
        "--strategyqa-input",
        type=Path,
        default=None,
        help="Optional CSV for ConflictQA strategyQA (same columns). Default: full JSONL.",
    )
    parser.add_argument(
        "--debateqa-input",
        type=Path,
        default=DEBATEQA_DEFAULT,
        help="DebateQA subset CSV (question, ground_truth, original_index).",
    )
    parser.add_argument(
        "--use-conflictqa-subset-csv",
        action="store_true",
        help=(
            "Load popQA/strategyQA from the Wood 480×3 resplit CSVs in output_wood "
            f"({POPQA_SUBSET_DEFAULT.name} / {STRATEGY_SUBSET_DEFAULT.name}) instead of full JSONL."
        ),
    )
    parser.add_argument(
        "--also-split-dataset-csv",
        type=Path,
        metavar="STEM",
        default=None,
        help=(
            "Mirror each appended row to STEM_<tag>.csv for known dataset tags "
            "(popqa, strategyqa, fever, debateqa; no extension; parent must exist). "
            "Combined rows still go to --out."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After CSV, run plot_ece_calibration.py on this file",
    )
    parser.add_argument(
        "--plot-file-suffix",
        default="",
        help="Passed to plot_ece_calibration.py --file-suffix when --plot is set.",
    )
    parser.add_argument(
        "--openai-model",
        default=OPENAI_API_MODEL,
        help="OpenAI API model id",
    )
    parser.add_argument(
        "--openai-temperature",
        type=float,
        default=0.0,
        help="OpenAI temperature for all datasets (default: 0.0)",
    )
    parser.add_argument(
        "--openai-fever-temperature",
        type=float,
        default=None,
        help="Optional OpenAI temperature override only for FEVER rows (e.g., 0.5)",
    )
    parser.add_argument(
        "--openai-confidence-scale",
        choices=("0-1", "1-10"),
        default="0-1",
        help="OpenAI confidence prompt scale. 1-10 is normalized back to [0,1] in CSV.",
    )
    parser.add_argument(
        "--openai-confidence-source",
        choices=("self-reported", "token-prob"),
        default="self-reported",
        help=(
            "OpenAI confidence source: self-reported reads JSON confidence from the model; "
            "token-prob computes confidence from completion token logprobs."
        ),
    )
    parser.add_argument(
        "--anthropic-model",
        default=ANTHROPIC_API_MODEL,
        help="Anthropic Messages API model id (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--anthropic-temperature",
        type=float,
        default=0.0,
        help="Anthropic temperature (default: 0.0)",
    )
    parser.add_argument(
        "--anthropic-confidence-scale",
        choices=("0-1", "1-10"),
        default="0-1",
        help="Anthropic confidence scale. 1-10 is normalized back to [0,1] in CSV.",
    )
    parser.add_argument(
        "--anthropic-confidence-source",
        choices=("self-reported", "token-prob"),
        default="self-reported",
        help=(
            "Anthropic confidence source: self-reported uses direct Anthropic API tool call; "
            "token-prob requires --use-openrouter-for-nonopenai and computes confidence from OpenRouter logprobs."
        ),
    )
    parser.add_argument(
        "--anthropic-model-id",
        default=MODELS[1][0],
        help="CSV model id for Anthropic rows (default: anthropic/claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--deepseek-model",
        default=DEEPSEEK_API_MODEL,
        help="DeepSeek API model id",
    )
    parser.add_argument(
        "--deepseek-model-id",
        default=MODELS[2][0],
        help="CSV model id for DeepSeek rows (default: deepseek/deepseek-chat-v2.5)",
    )
    parser.add_argument(
        "--deepseek-temperature",
        type=float,
        default=0.0,
        help="DeepSeek temperature (default: 0.0)",
    )
    parser.add_argument(
        "--deepseek-confidence-scale",
        choices=("0-1", "1-10"),
        default="0-1",
        help="DeepSeek confidence prompt scale. 1-10 is normalized back to [0,1] in CSV.",
    )
    parser.add_argument(
        "--deepseek-confidence-source",
        choices=("self-reported", "token-prob"),
        default="self-reported",
        help=(
            "DeepSeek confidence source: self-reported uses DeepSeek API JSON confidence; "
            "token-prob requires --use-openrouter-for-nonopenai and computes confidence from OpenRouter logprobs."
        ),
    )
    parser.add_argument(
        "--gemini-model",
        default=GEMINI_API_MODEL,
        help="Gemini API model id (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--gemini-temperature",
        type=float,
        default=0.0,
        help="Gemini temperature (default: 0.0)",
    )
    parser.add_argument(
        "--gemini-confidence-scale",
        choices=("0-1", "1-10"),
        default="0-1",
        help="Gemini confidence scale. 1-10 is normalized back to [0,1] in CSV.",
    )
    parser.add_argument(
        "--gemini-confidence-source",
        choices=("self-reported", "token-prob"),
        default="self-reported",
        help=(
            "Gemini confidence source: self-reported uses Gemini API JSON confidence; "
            "token-prob requires --use-openrouter-for-nonopenai and computes confidence from OpenRouter logprobs."
        ),
    )
    parser.add_argument(
        "--gemini-model-id",
        default=MODELS[3][0],
        help="CSV model id for Gemini rows (default: google/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--skip-openai",
        action="store_true",
        help="Do not call OpenAI; OPENAI_API_KEY not required. Use for DeepSeek-only runs.",
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Do not call Gemini; GEMINI_API_KEY not required. Rerun later without this flag to append Gemini rows.",
    )
    parser.add_argument(
        "--skip-anthropic",
        action="store_true",
        help="Do not call Anthropic; CLAUDE_API_KEY not required. Use when credits are depleted; rerun later to append Claude rows.",
    )
    parser.add_argument(
        "--skip-deepseek",
        action="store_true",
        help="Do not call DeepSeek; DEEPSEEK_API_KEY not required. Use for OpenAI-only runs.",
    )
    parser.add_argument(
        "--use-openrouter-for-nonopenai",
        action="store_true",
        help=(
            "Route Anthropic/DeepSeek/Gemini calls via OpenRouter chat/completions. "
            "Required for token-prob on non-OpenAI models."
        ),
    )
    parser.add_argument(
        "--redo-model",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help="Re-run rows for this CSV model id even if already present (repeatable). "
        "Use after fixing API ids, e.g. --redo-model anthropic/claude-sonnet-4-6",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        metavar="N",
        help="Print progress every N completed API calls (default: 25). Use 1 for every call, 0 for quiet.",
    )
    parser.add_argument(
        "--elicitation-mode",
        choices=("default", "epistemic-marker"),
        default="default",
        help=(
            "Prompt mode for confidence elicitation. "
            "epistemic-marker asks model to include exactly one epistemic marker in answer text "
            "(aligned with ACL'25 marker setup)."
        ),
    )
    parser.add_argument(
        "--extra-openrouter-tokenprob-model",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help=(
            "Additional OpenRouter model id to run in token-prob mode "
            "(repeatable), e.g. google/gemma-4-26b-it."
        ),
    )
    parser.add_argument(
        "--extra-openrouter-selfreported-model",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help=(
            "Additional OpenRouter model id to run in self-reported mode "
            "(repeatable), e.g. qwen/qwen3-14b."
        ),
    )
    parser.add_argument(
        "--extra-openrouter-temperature",
        type=float,
        default=0.0,
        help="Temperature for --extra-openrouter-tokenprob-model calls (default: 0.0).",
    )
    parser.add_argument(
        "--extra-openrouter-confidence-scale",
        choices=("0-1", "1-10"),
        default="0-1",
        help="Confidence scale for --extra-openrouter-selfreported-model prompts.",
    )
    args = parser.parse_args()

    if not args.allow_any_out_path:
        out_raw = args.out.expanduser()
        out_res = out_raw.resolve() if out_raw.is_absolute() else (PROJECT_ROOT / out_raw).resolve()
        base_res = (PROJECT_ROOT / "output_wood" / "self_reported_confidence").resolve()
        try:
            out_res.relative_to(base_res)
        except ValueError:
            print(
                f"--out must be inside {base_res} (resolved: {out_res}). "
                "Put rollout CSVs under output_wood/self_reported_confidence/, or pass --allow-any-out-path.",
                file=sys.stderr,
            )
            sys.exit(1)

    load_dotenv(PROJECT_ROOT / ".env")

    required_keys: list[str] = []
    if not args.skip_openai:
        required_keys.append("OPENAI_API_KEY")
    if not args.skip_deepseek:
        required_keys.append("DEEPSEEK_API_KEY")
    if not args.skip_anthropic:
        required_keys.append("CLAUDE_API_KEY")
    if not args.skip_gemini:
        required_keys.append("GEMINI_API_KEY")
    if args.use_openrouter_for_nonopenai and (
        (not args.skip_anthropic and args.anthropic_confidence_source == "token-prob")
        or (not args.skip_deepseek and args.deepseek_confidence_source == "token-prob")
        or (not args.skip_gemini and args.gemini_confidence_source == "token-prob")
    ):
        required_keys.append("OPENROUTER_API_KEY")
    if args.extra_openrouter_tokenprob_model or args.extra_openrouter_selfreported_model:
        required_keys.append("OPENROUTER_API_KEY")
    for key in required_keys:
        if not os.getenv(key):
            print(f"Missing {key} in environment or .env", file=sys.stderr)
            sys.exit(1)

    models_to_run = [
        (MODELS[0][0], "openai"),
        (args.anthropic_model_id, "anthropic"),
        (args.deepseek_model_id, "deepseek"),
        (args.gemini_model_id, "google"),
    ]
    if args.skip_gemini:
        models_to_run = [m for m in models_to_run if m[1] != "google"]
    if args.skip_anthropic:
        models_to_run = [m for m in models_to_run if m[1] != "anthropic"]
    if args.skip_openai:
        models_to_run = [m for m in models_to_run if m[1] != "openai"]
    if args.skip_deepseek:
        models_to_run = [m for m in models_to_run if m[1] != "deepseek"]
    for raw_model_name in args.extra_openrouter_tokenprob_model:
        model_name = normalize_openrouter_model_id(raw_model_name)
        models_to_run.append((model_name, f"openrouter_extra::{model_name}"))
    for raw_model_name in args.extra_openrouter_selfreported_model:
        model_name = normalize_openrouter_model_id(raw_model_name)
        models_to_run.append((model_name, f"openrouter_selfreported::{model_name}"))
    if not models_to_run:
        print(
            "No models left to run (adjust --skip-openai / --skip-gemini / --skip-anthropic / --skip-deepseek).",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.skip_openai:
        print("Skipping OpenAI (--skip-openai).", file=sys.stderr)
    if args.skip_gemini:
        print("Skipping Gemini (--skip-gemini).", file=sys.stderr)
    if args.skip_anthropic:
        print("Skipping Anthropic Claude (--skip-anthropic).", file=sys.stderr)
    if args.skip_deepseek:
        print("Skipping DeepSeek (--skip-deepseek).", file=sys.stderr)

    callers = make_callers(
        args.openai_model,
        args.anthropic_model,
        args.deepseek_model,
        args.gemini_model,
        args.openai_temperature,
        args.openai_fever_temperature,
        args.openai_confidence_scale,
        args.openai_confidence_source,
        args.anthropic_temperature,
        args.anthropic_confidence_scale,
        args.anthropic_confidence_source,
        args.deepseek_temperature,
        args.deepseek_confidence_scale,
        args.deepseek_confidence_source,
        args.gemini_temperature,
        args.gemini_confidence_scale,
        args.gemini_confidence_source,
        args.use_openrouter_for_nonopenai,
        args.elicitation_mode,
        args.extra_openrouter_tokenprob_model,
        args.extra_openrouter_selfreported_model,
        args.extra_openrouter_temperature,
        args.extra_openrouter_confidence_scale,
    )
    if args.openai_confidence_source == "token-prob" and args.openai_confidence_scale != "0-1":
        raise ValueError("--openai-confidence-source token-prob requires --openai-confidence-scale 0-1")
    if args.anthropic_confidence_source == "token-prob" and args.anthropic_confidence_scale != "0-1":
        raise ValueError("--anthropic-confidence-source token-prob requires --anthropic-confidence-scale 0-1")
    if args.deepseek_confidence_source == "token-prob" and args.deepseek_confidence_scale != "0-1":
        raise ValueError("--deepseek-confidence-source token-prob requires --deepseek-confidence-scale 0-1")
    if args.gemini_confidence_source == "token-prob" and args.gemini_confidence_scale != "0-1":
        raise ValueError("--gemini-confidence-source token-prob requires --gemini-confidence-scale 0-1")
    if (
        (not args.skip_anthropic and args.anthropic_confidence_source == "token-prob")
        or (not args.skip_gemini and args.gemini_confidence_source == "token-prob")
    ) and not args.use_openrouter_for_nonopenai:
        raise ValueError(
            "token-prob for Anthropic/Gemini requires --use-openrouter-for-nonopenai "
            "(OpenRouter key must be available as OPENROUTER_API_KEY). "
            "DeepSeek token-prob uses the native DeepSeek API when this flag is off."
        )

    popqa_path = args.popqa_input
    strategy_path = args.strategyqa_input
    if args.use_conflictqa_subset_csv:
        popqa_path = args.popqa_input or POPQA_SUBSET_DEFAULT
        strategy_path = args.strategyqa_input or STRATEGY_SUBSET_DEFAULT

    frames = []
    if args.datasets == "fever_popqa_debateqa":
        if not args.fever_input.exists():
            raise FileNotFoundError(args.fever_input)
        frames.append((read_fever_csv(args.fever_input), "fever"))
        popqa_f = args.popqa_input or (POPQA_SUBSET_DEFAULT if args.use_conflictqa_subset_csv else None)
        if popqa_f is not None:
            if not popqa_f.exists():
                raise FileNotFoundError(popqa_f)
            frames.append((read_conflictqa_csv(popqa_f), "conflictqa_popqa"))
        else:
            frames.append((read_jsonl(POPQA), "conflictqa_popqa"))
        if not args.debateqa_input.exists():
            raise FileNotFoundError(args.debateqa_input)
        frames.append((read_conflictqa_csv(args.debateqa_input), "debateqa"))
    elif args.datasets == "debateqa":
        if not args.debateqa_input.exists():
            raise FileNotFoundError(args.debateqa_input)
        frames.append((read_conflictqa_csv(args.debateqa_input), "debateqa"))
    else:
        if args.datasets in ("both", "all", "popqa"):
            if popqa_path is not None:
                if not popqa_path.exists():
                    raise FileNotFoundError(popqa_path)
                frames.append((read_conflictqa_csv(popqa_path), "conflictqa_popqa"))
            else:
                frames.append((read_jsonl(POPQA), "conflictqa_popqa"))
        if args.datasets in ("both", "all", "strategyqa"):
            if strategy_path is not None:
                if not strategy_path.exists():
                    raise FileNotFoundError(strategy_path)
                frames.append((read_conflictqa_csv(strategy_path), "conflictqa_strategyqa"))
            else:
                frames.append((read_jsonl(STRATEGY), "conflictqa_strategyqa"))
        if args.datasets in ("all", "fever"):
            if not args.fever_input.exists():
                raise FileNotFoundError(args.fever_input)
            frames.append((read_fever_csv(args.fever_input), "fever"))

    done: set[tuple[str, str, int]] = set()
    initial_row_count = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.stat().st_size > 0:
        old = pd.read_csv(args.out)
        initial_row_count = len(old)
        for _, r in old.iterrows():
            done.add((str(r["model"]), str(r["dataset"]), int(r["original_index"])))

    for rm in args.redo_model:
        done = {k for k in done if k[0] != rm}

    def count_pending() -> int:
        n = 0
        for base_df, dataset_tag in frames:
            sub = base_df if args.max_items is None else base_df.head(args.max_items)
            for _, row in sub.iterrows():
                oid = int(row["original_index"])
                for model_id, _prov in models_to_run:
                    if (model_id, dataset_tag, oid) not in done:
                        n += 1
        return n

    pending = count_pending()
    ds_tags = [t for _, t in frames]
    print(
        f"collect_conflictqa_ece: {pending} API call(s) queued "
        f"({len(models_to_run)} model(s), datasets={ds_tags}). "
        "First response can take 10–60s.",
        flush=True,
    )

    completed = 0
    t0 = time.monotonic()
    csv_writer: IncrementalCsvWriter | None = None
    split_stem = args.also_split_dataset_csv
    split_writers: dict[str, IncrementalCsvWriter] = {}

    try:
        for base_df, dataset_tag in frames:
            sub = base_df
            if args.max_items is not None:
                sub = sub.head(args.max_items)
            for _, row in sub.iterrows():
                q = str(row.get("question", "")).strip()
                gt = row.get("ground_truth", [])
                if not isinstance(gt, list):
                    gt = [gt]
                refs = "; ".join(map(str, gt))
                oid = int(row["original_index"])

                for model_id, provider in models_to_run:
                    key = (model_id, dataset_tag, oid)
                    if key in done:
                        continue
                    fn = callers[provider]
                    try:
                        answer, conf = fn(q, refs, dataset_tag)
                        conf = float(max(0.0, min(1.0, conf)))
                        cor = is_correct(answer, gt)
                    except MissingLogprobsError as e:
                        # Skip writing a failed row with zero confidence; keep it rerunnable.
                        print(f"[WARN] {model_id} row {oid}: {e}")
                        continue
                    except Exception as e:
                        answer = ""
                        conf = 0.0
                        cor = 0
                        err = str(e)
                        print(f"[ERROR] {model_id} row {oid}: {err}")
                    else:
                        err = ""

                    # Derive linguistic confidence features from epistemic markers in the answer text.
                    ling_counts = detect_linguistic_markers(answer)
                    ling_level = linguistic_confidence_level(answer)
                    ling_score = linguistic_confidence_score(answer)

                    rec = {
                        "model": model_id,
                        "dataset": dataset_tag,
                        "original_index": oid,
                        "question": q,
                        "answer": answer,
                        "confidence": conf,
                        "correct": cor,
                        "error": err,
                        "ling_conf_level": ling_level,
                        "ling_conf_score": ling_score,
                        "ling_marker_count": ling_counts["marker_count"],
                        "ling_unique_marker_count": ling_counts["unique_marker_count"],
                        "ling_has_marker": ling_counts["has_marker"],
                    }
                    if csv_writer is None:
                        csv_writer = IncrementalCsvWriter(args.out)
                    csv_writer.append_row(rec)
                    if split_stem is not None:
                        ds_tag = str(rec.get("dataset", ""))
                        split_path = conflictqa_split_csv_path(Path(split_stem), ds_tag)
                        if split_path is not None:
                            if ds_tag not in split_writers:
                                split_path.parent.mkdir(parents=True, exist_ok=True)
                                split_writers[ds_tag] = IncrementalCsvWriter(split_path)
                            split_writers[ds_tag].append_row(rec)
                    done.add(key)
                    completed += 1
                    pe = args.progress_every
                    if pe > 0 and (completed == 1 or completed % pe == 0):
                        elapsed = time.monotonic() - t0
                        rate = completed / elapsed if elapsed > 0 else 0.0
                        eta = (pending - completed) / rate if rate > 0 and pending > completed else float("nan")
                        eta_s = f" ETA ~{eta / 60:.1f}m" if eta == eta and eta > 0 else ""
                        print(
                            f"[progress] {completed}/{pending}  {model_id}  {dataset_tag}  row={oid}  "
                            f"{elapsed:.0f}s elapsed{eta_s}",
                            flush=True,
                        )
                    time.sleep(args.sleep_s)
    finally:
        if csv_writer is not None:
            csv_writer.close()
        for sw in split_writers.values():
            sw.close()

    if completed > 0:
        print(
            f"Saved incrementally: +{completed} row(s) → {args.out} "
            f"(total rows {initial_row_count + completed})",
            flush=True,
        )
    else:
        print("No new rows; everything already in", args.out, flush=True)

    if args.plot:
        import subprocess

        plot_script = Path(__file__).resolve().parent / "plot_ece_calibration.py"
        cmd = [
            sys.executable,
            str(plot_script),
            "--input",
            str(args.out),
            "--out-dir",
            str(OUT_DIR),
            "--dataset-b",
            "conflictqa_popqa",
            "--dataset-c",
            "conflictqa_strategyqa",
        ]
        ps = (args.plot_file_suffix or "").strip()
        if ps:
            cmd.extend(["--file-suffix", ps])
        print("Running:", " ".join(cmd))
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
