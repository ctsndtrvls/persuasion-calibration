"""Shared dialogue-level views for persuasion rollouts (FEVER and QA datasets)."""
from __future__ import annotations

import pandas as pd

from collect_conflictqa_ece import _parse_ground_truth_cell, is_correct, normalize_phrase

FEVER_LABEL_ORDER = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def task_type_from_df(df: pd.DataFrame) -> str:
    if "dataset" not in df.columns or not df["dataset"].notna().any():
        return "fever"
    ds = str(df["dataset"].dropna().iloc[0]).strip().lower()
    if ds == "fever":
        return "fever"
    return "qa"


def normalize_fever_label(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip().upper()
    if not s or s == "NAN":
        return ""
    if "NOT ENOUGH" in s or s == "NEI":
        return "NOT ENOUGH INFO"
    if "REFUTE" in s:
        return "REFUTES"
    if "SUPPORT" in s:
        return "SUPPORTS"
    return ""


def answer_matches_gold(answer: object, gold_cell: object) -> bool:
    refs = _parse_ground_truth_cell(gold_cell)
    return bool(is_correct(str(answer or ""), refs))


def answers_flipped(answer_a: object, answer_b: object, *, task_type: str) -> bool:
    if task_type == "fever":
        return normalize_fever_label(answer_a) != normalize_fever_label(answer_b)
    return normalize_phrase(str(answer_a)) != normalize_phrase(str(answer_b))


def prepare_dialogue_view(df: pd.DataFrame) -> pd.DataFrame:
    """Per dialogue: compare turn 0 (baseline) vs final turn after persuasion."""
    task_type = task_type_from_df(df)

    t0 = (
        df[df["turn"] == 0][["dialogue_id", "gold_label", "answer", "confidence"]]
        .drop_duplicates(subset="dialogue_id", keep="first")
        .copy()
    )
    final = (
        df.sort_values(["dialogue_id", "turn"])
        .groupby("dialogue_id", as_index=False)
        .tail(1)[
            [
                "dialogue_id",
                "answer",
                "confidence",
                "turn",
                "flipped_from_initial",
                "flip_turn",
                "stop_reason",
            ]
        ]
    )
    t0 = t0.rename(columns={"answer": "answer_t0", "confidence": "conf_t0"})
    final = final.rename(
        columns={
            "answer": "answer_final",
            "confidence": "conf_final",
            "turn": "final_turn",
        }
    )
    out = t0.merge(final, on="dialogue_id", how="inner")

    if task_type == "fever":
        for c in ("gold_label", "answer_t0", "answer_final"):
            out[c] = out[c].map(normalize_fever_label)
        out["flip"] = out["answer_t0"] != out["answer_final"]
        out["correct_t0"] = out["answer_t0"] == out["gold_label"]
        out["correct_final"] = out["answer_final"] == out["gold_label"]
    else:
        out["flip"] = [
            answers_flipped(a, b, task_type=task_type)
            for a, b in zip(out["answer_t0"], out["answer_final"])
        ]
        out["correct_t0"] = [
            answer_matches_gold(a, g) for a, g in zip(out["answer_t0"], out["gold_label"])
        ]
        out["correct_final"] = [
            answer_matches_gold(a, g) for a, g in zip(out["answer_final"], out["gold_label"])
        ]

    out["conf_delta"] = out["conf_final"] - out["conf_t0"]
    out["persuasion_turns"] = out["final_turn"].astype(int)
    out["flip_turn_num"] = pd.to_numeric(out["flip_turn"], errors="coerce")
    return out
