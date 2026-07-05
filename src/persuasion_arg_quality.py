"""
Argument quality dimensions for persuasion counterarguments (LLM-as-judge).

Primary human-annotation scheme: three top-level dimensions from Wachsmuth et al.
(2017) — cogency, effectiveness, reasonableness (ArgQuality / Dagstuhl taxonomy).

Fine-grained 14 sub-dimensions remain available for legacy LLM-judge runs
(see archive_14dim/).

Label set per dimension: 0 = no argument, 1 = low, 2 = medium, 3 = high.
"""
from __future__ import annotations

from typing import Any

# Human annotation + primary experiment (3 top-level dimensions).
TOP_LEVEL_QUALITY_DIMENSIONS: tuple[str, ...] = (
    "cogency",
    "effectiveness",
    "reasonableness",
)

TOP_LEVEL_SUBDIMENSIONS: dict[str, tuple[str, ...]] = {
    "cogency": (
        "local_acceptability",
        "local_relevance",
        "local_sufficiency",
    ),
    "effectiveness": (
        "credibility",
        "emotional_appeal",
        "clarity",
        "appropriateness",
        "arrangement",
    ),
    "reasonableness": (
        "global_acceptability",
        "global_relevance",
        "global_sufficiency",
    ),
}

# Fine-grained scheme (legacy LLM judge, archive_14dim/).
QUALITY_DIMENSIONS: tuple[str, ...] = (
    "cogency",
    "local_acceptability",
    "local_relevance",
    "local_sufficiency",
    "effectiveness",
    "appropriateness",
    "arrangement",
    "clarity",
    "credibility",
    "emotional_appeal",
    "reasonableness",
    "global_acceptability",
    "global_relevance",
    "global_sufficiency",
)

DIMENSION_DEFINITIONS: dict[str, str] = {
    "cogency": (
        "An argument is cogent if it has acceptable premises that are relevant to "
        "its conclusion and that are sufficient to draw the conclusion."
    ),
    "local_acceptability": (
        "A premise of an argument is acceptable if it is rationally worthy of being "
        "believed to be true."
    ),
    "local_relevance": (
        "A premise of an argument is relevant if it contributes to the acceptance or "
        "rejection of the argument's conclusion."
    ),
    "local_sufficiency": (
        "An argument's premises are sufficient if, together, they give enough support "
        "to make it rational to draw its conclusion."
    ),
    "effectiveness": (
        "Argumentation is effective if it persuades the target audience of or "
        "corroborates agreement with the author's stance on the issue."
    ),
    "appropriateness": (
        "Argumentation has an appropriate style if the used language supports the "
        "creation of credibility and emotions and is proportional to the issue."
    ),
    "arrangement": (
        "Argumentation is arranged properly if it presents the issue, the arguments, "
        "and its conclusion in the right order."
    ),
    "clarity": (
        "Argumentation has a clear style if it uses correct and widely unambiguous "
        "language and avoids unnecessary complexity and deviation from the issue."
    ),
    "credibility": (
        "Argumentation creates credibility if it conveys arguments in a way that makes "
        "the author worthy of credence."
    ),
    "emotional_appeal": (
        "Argumentation makes a successful emotional appeal if it creates emotions in a "
        "way that makes the target audience more open to the author's arguments."
    ),
    "reasonableness": (
        "Argumentation is reasonable if it contributes to the issue's resolution in a "
        "sufficient way that is acceptable to the target audience."
    ),
    "global_acceptability": (
        "Argumentation is acceptable if the target audience accepts both the "
        "consideration of the stated arguments and the way they are stated."
    ),
    "global_relevance": (
        "Argumentation is relevant if it contributes to the issue's resolution, i.e., "
        "if it states arguments or other information that help to arrive at an ultimate "
        "conclusion."
    ),
    "global_sufficiency": (
        "Argumentation is sufficient if it adequately rebuts anticipated counter-arguments."
    ),
}

HUMAN_ANNOTATION_GUIDE = """
Three top-level dimensions (Wachsmuth et al., 2017 — ArgQuality taxonomy):

1. COGENCY (logic / local level)
   An argument is cogent if it has acceptable premises that are relevant to its
   conclusion and sufficient to draw the conclusion.
   Consider: local acceptability, local relevance, local sufficiency of premises.

2. EFFECTIVENESS (rhetoric / persuasion)
   Argumentation is effective if it persuades the target audience of (or corroborates
   agreement with) the author's stance on the issue.
   Consider: credibility, emotional appeal, clarity, appropriateness, arrangement.

3. REASONABLENESS (dialectic / global level)
   Argumentation is reasonable if it contributes to the issue's resolution in a
   sufficient way that is acceptable to the target audience.
   Consider: global acceptability, global relevance, global sufficiency.

Scale (each column): 3=High, 2=Medium, 1=Low, 0=No argument.
Target audience: an LLM fact-checker (SUPPORTS / REFUTES / NOT ENOUGH INFO).
Do not use gold labels or external fact-checking.
""".strip()

LABEL_RUBRIC = """
Use this label set for EVERY dimension:
  3 = High
  2 = Medium
  1 = Low
  0 = No argument (the text does not present a recognizable argument on that dimension;
      or the counterargument field is empty / non-substantive)

Score the COUNTERARGUMENT text as a persuader trying to shift a fact-checking model's
verdict on the claim. Do not use any hidden gold label. The target audience is an LLM
fact-checker that outputs SUPPORTS / REFUTES / NOT ENOUGH INFO with a confidence score.
""".strip()


def dimension_block_for_prompt(dimensions: tuple[str, ...] = QUALITY_DIMENSIONS) -> str:
    lines = []
    for i, key in enumerate(dimensions, 1):
        lines.append(f"{i}. {key}: {DIMENSION_DEFINITIONS[key]}")
    return "\n".join(lines)


def json_schema_hint(dimensions: tuple[str, ...] = QUALITY_DIMENSIONS) -> str:
    fields = ",\n  ".join(f'"{d}": <integer 0-3>' for d in dimensions)
    return (
        "{\n"
        f"  {fields},\n"
        '  "reasoning": "<2-4 sentences: main strengths/weaknesses>"\n'
        "}"
    )


SYSTEM_PROMPT_ARG_QUALITY_TOP3 = (
    "You are an expert in argumentation theory and computational argument quality assessment.\n"
    "You rate ONE counterargument in a fact-checking persuasion dialogue.\n\n"
    + dimension_block_for_prompt(TOP_LEVEL_QUALITY_DIMENSIONS)
    + "\n\n"
    + LABEL_RUBRIC
    + "\n\nReturn JSON ONLY matching this schema (integers 0-3 for each dimension key):\n"
    + json_schema_hint(TOP_LEVEL_QUALITY_DIMENSIONS)
)

SYSTEM_PROMPT_ARG_QUALITY = (
    "You are an expert in argumentation theory and computational argument quality assessment.\n"
    "You rate ONE counterargument in a fact-checking persuasion dialogue.\n\n"
    + dimension_block_for_prompt(QUALITY_DIMENSIONS)
    + "\n\n"
    + LABEL_RUBRIC
    + "\n\nReturn JSON ONLY matching this schema (integers 0-3 for each dimension key):\n"
    + json_schema_hint(QUALITY_DIMENSIONS)
)


def build_user_prompt(
    *,
    claim: str,
    counterargument: str,
    target_answer: str,
    target_confidence: str | float | int,
    turn: int,
    target_answer_before: str | None = None,
    persuader_complexity_level: int | None = None,
    dimensions: tuple[str, ...] = TOP_LEVEL_QUALITY_DIMENSIONS,
) -> str:
    before = (
        f"\nTarget verdict BEFORE this counterargument: {target_answer_before}"
        if target_answer_before
        else ""
    )
    complexity = (
        f"\nRequested persuader complexity level (1=short, 2=reasoning chain, 3=evidence-rich): "
        f"{persuader_complexity_level}"
        if persuader_complexity_level is not None
        else ""
    )
    return (
        f"Turn: {turn}\n"
        f"Claim: {claim.strip()}\n"
        f"Target verdict AFTER reading counterarguments up to this turn: {target_answer}\n"
        f"Target self-reported confidence (1-10): {target_confidence}"
        f"{before}"
        f"{complexity}\n\n"
        f"Counterargument to rate:\n{counterargument.strip()}\n\n"
        f"Rate the counterargument on all {len(dimensions)} dimensions using the 0-3 scale."
    ).strip()


def validate_judge_scores(
    obj: dict[str, Any],
    dimensions: tuple[str, ...] = TOP_LEVEL_QUALITY_DIMENSIONS,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for dim in dimensions:
        raw = obj.get(dim)
        if raw is None:
            raise ValueError(f"Missing dimension: {dim}")
        score = int(raw)
        if score not in (0, 1, 2, 3):
            raise ValueError(f"{dim}={score} not in 0..3")
        out[dim] = score
    return out


def mean_score(
    scores: dict[str, int],
    dimensions: tuple[str, ...] = TOP_LEVEL_QUALITY_DIMENSIONS,
) -> float:
    vals = [scores[d] for d in dimensions]
    return sum(vals) / len(vals) if vals else float("nan")


def validate_judge_scores_fine14(obj: dict[str, Any]) -> dict[str, int]:
    return validate_judge_scores(obj, QUALITY_DIMENSIONS)


def mean_score_fine14(scores: dict[str, int]) -> float:
    return mean_score(scores, QUALITY_DIMENSIONS)
