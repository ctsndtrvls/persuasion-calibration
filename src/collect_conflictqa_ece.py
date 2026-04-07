"""
Run chat models on ConflictQA (popQA + strategyQA) and optional FEVER, collect JSON answers
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

Confidence is the model's stated P(correct) in JSON (verbal calibration), not token logprobs.

Anthropic/Gemini: old IDs (claude-3-5-sonnet-20241022, gemini-1.5-flash-8b) return 404.
Defaults are claude-sonnet-4-6 and gemini-2.5-flash; see --anthropic-model / --gemini-model.

After ID migration in the model column, old failed rows do not block new calls.
Use --redo-model when you need to refresh the same model ID.

Each completed row is appended to CSV and flushed immediately. On interruption (Ctrl+C),
only the in-flight API call can be lost; already written rows remain in the file.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "conflictqa"
OUT_DIR = PROJECT_ROOT / "output_wood"
POPQA = DATA_DIR / "conflictQA-popQA-llama2-7b.json"
STRATEGY = DATA_DIR / "conflictQA-strategyQA-llama2-7b.json"
FEVER_DEFAULT = OUT_DIR / "fever480_160x3_complexity_wood_v1_lr_40_resplit.csv"

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

USER_TEMPLATE = """Reference answers (any equivalent labeling of the same entity is acceptable): {refs}

Question: {question}

Output JSON: {{"answer": "<short phrase>", "confidence": <number from 0 to 1>}}"""

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


def call_openai(question: str, refs: str, model_name: str) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user = USER_TEMPLATE.format(question=question, refs=refs)
    r = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = r.choices[0].message.content or "{}"
    data = parse_json_obj(raw)
    return str(data.get("answer", "")), float(data.get("confidence", 0.0))


def call_anthropic(question: str, refs: str, model_name: str) -> tuple[str, float]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
    user = USER_TEMPLATE.format(question=question, refs=refs)
    msg = client.messages.create(
        model=model_name,
        max_tokens=1024,
        system=SYSTEM_ANTHROPIC_TOOL,
        messages=[{"role": "user", "content": user}],
        tools=[ANTHROPIC_CALIBRATION_TOOL],
        tool_choice={"type": "tool", "name": "submit_calibration"},
        temperature=0.0,
    )
    for b in msg.content:
        if getattr(b, "type", None) == "tool_use" and getattr(b, "name", None) == "submit_calibration":
            inp = getattr(b, "input", None)
            if isinstance(inp, dict):
                return str(inp.get("answer", "")), _clip_confidence(inp.get("confidence", 0.0))
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
    return str(data.get("answer", "")), _clip_confidence(data.get("confidence", 0.0))


def call_deepseek(question: str, refs: str, model_name: str) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    user = USER_TEMPLATE.format(question=question, refs=refs)
    kwargs: dict[str, Any] = dict(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    try:
        r = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception:
        r = client.chat.completions.create(**kwargs)
    raw = r.choices[0].message.content or "{}"
    data = parse_json_obj(raw)
    return str(data.get("answer", "")), float(data.get("confidence", 0.0))


def call_gemini(question: str, refs: str, model_name: str) -> tuple[str, float]:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name,
        system_instruction=SYSTEM,
    )
    user = USER_TEMPLATE.format(question=question, refs=refs)
    try:
        r = model.generate_content(
            user,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
    except Exception:
        r = model.generate_content(
            user,
            generation_config=genai.GenerationConfig(temperature=0.0),
        )
    raw = r.text or "{}"
    data = parse_json_obj(raw)
    return str(data.get("answer", "")), float(data.get("confidence", 0.0))


def make_callers(
    openai_model: str,
    anthropic_model: str,
    deepseek_model: str,
    gemini_model: str,
) -> dict[str, Callable[[str, str], tuple[str, float]]]:
    return {
        "openai": lambda q, r: call_openai(q, r, openai_model),
        "anthropic": lambda q, r: call_anthropic(q, r, anthropic_model),
        "deepseek": lambda q, r: call_deepseek(q, r, deepseek_model),
        "google": lambda q, r: call_gemini(q, r, gemini_model),
    }


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
    parser.add_argument("--out", type=Path, default=OUT_DIR / "conflictqa_ece_rollout.csv")
    parser.add_argument("--sleep-s", type=float, default=0.25, help="Pause between API calls")
    parser.add_argument("--max-items", type=int, default=None, help="Cap rows per dataset (for tests)")
    parser.add_argument(
        "--datasets",
        choices=("both", "popqa", "strategyqa", "fever", "all"),
        default="both",
        help="Dataset selection: both=ConflictQA only (popqa+strategyqa), all=ConflictQA+FEVER.",
    )
    parser.add_argument(
        "--fever-input",
        type=Path,
        default=FEVER_DEFAULT,
        help="Path to FEVER CSV (default: output_wood/fever480_160x3_complexity_wood_v1_lr_40_resplit.csv)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After CSV, run plot_ece_calibration.py on this file",
    )
    parser.add_argument(
        "--openai-model",
        default=OPENAI_API_MODEL,
        help="OpenAI API model id",
    )
    parser.add_argument(
        "--anthropic-model",
        default=ANTHROPIC_API_MODEL,
        help="Anthropic Messages API model id (default: claude-sonnet-4-6)",
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
        "--gemini-model",
        default=GEMINI_API_MODEL,
        help="Gemini API model id (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--gemini-model-id",
        default=MODELS[3][0],
        help="CSV model id for Gemini rows (default: google/gemini-2.5-flash)",
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
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    required_keys = ["OPENAI_API_KEY", "DEEPSEEK_API_KEY"]
    if not args.skip_anthropic:
        required_keys.append("CLAUDE_API_KEY")
    if not args.skip_gemini:
        required_keys.append("GEMINI_API_KEY")
    for key in required_keys:
        if not os.getenv(key):
            print(f"Missing {key} in environment or .env", file=sys.stderr)
            sys.exit(1)

    models_to_run = [
        (MODELS[0][0], "openai"),
        (args.anthropic_model_id, "anthropic"),
        (MODELS[2][0], "deepseek"),
        (args.gemini_model_id, "google"),
    ]
    if args.skip_gemini:
        models_to_run = [m for m in models_to_run if m[1] != "google"]
    if args.skip_anthropic:
        models_to_run = [m for m in models_to_run if m[1] != "anthropic"]
    if not models_to_run:
        print("No models left to run (adjust --skip-gemini / --skip-anthropic).", file=sys.stderr)
        sys.exit(1)
    if args.skip_gemini:
        print("Skipping Gemini (--skip-gemini).", file=sys.stderr)
    if args.skip_anthropic:
        print("Skipping Anthropic Claude (--skip-anthropic).", file=sys.stderr)

    callers = make_callers(
        args.openai_model,
        args.anthropic_model,
        args.deepseek_model,
        args.gemini_model,
    )

    frames = []
    if args.datasets in ("both", "all", "popqa"):
        frames.append((read_jsonl(POPQA), "conflictqa_popqa"))
    if args.datasets in ("both", "all", "strategyqa"):
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
                        answer, conf = fn(q, refs)
                        conf = float(max(0.0, min(1.0, conf)))
                        cor = is_correct(answer, gt)
                    except Exception as e:
                        answer = ""
                        conf = 0.0
                        cor = 0
                        err = str(e)
                        print(f"[ERROR] {model_id} row {oid}: {err}")
                    else:
                        err = ""

                    rec = {
                        "model": model_id,
                        "dataset": dataset_tag,
                        "original_index": oid,
                        "question": q,
                        "answer": answer,
                        "confidence": conf,
                        "correct": cor,
                        "error": err,
                    }
                    if csv_writer is None:
                        csv_writer = IncrementalCsvWriter(args.out)
                    csv_writer.append_row(rec)
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
        print("Running:", " ".join(cmd))
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
