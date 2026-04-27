from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, Mapping, TypedDict

# This module follows two papers:
# - Xiong et al. (ICLR 2024): confidence is elicited from verbalized uncertainty.
# - Liu et al. (ACL 2025): marker confidence = empirical accuracy when a marker is used.
#
# In practice we keep a deterministic marker extractor + profile-based confidence estimation.
# Heuristic "strong/medium/weak" buckets are retained only for backward compatibility.


NO_MARKER = "__no_marker__"

# Common epistemic markers used in LLM confidence phrasing (papers + close variants).
EPISTEMIC_MARKERS: list[str] = [
    "not sure",
    "unsure",
    "uncertain",
    "might be",
    "might",
    "may",
    "possibly",
    "possible",
    "perhaps",
    "seems",
    "appears",
    "likely",
    "probably",
    "presumably",
    "i think",
    "i believe",
    "fairly certain",
    "pretty sure",
    "quite sure",
    "quite certain",
    "confident",
    "very confident",
    "undoubtedly",
    "certainly",
    "definitely",
]

# Tier mapping is only a compatibility layer for existing CSV columns.
STRONG_UNCERTAINTY = {
    "not sure",
    "unsure",
    "uncertain",
    "might be",
    "might",
    "may",
    "possibly",
    "possible",
    "perhaps",
}
MEDIUM_UNCERTAINTY = {
    "seems",
    "appears",
    "likely",
    "probably",
    "presumably",
    "i think",
    "i believe",
}
WEAK_UNCERTAINTY = {
    "fairly certain",
    "pretty sure",
    "quite sure",
    "quite certain",
    "confident",
    "very confident",
    "undoubtedly",
    "certainly",
    "definitely",
}


class MarkerStats(TypedDict):
    occurrences: int
    correct: int
    confidence: float


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.lower()
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def extract_epistemic_markers(text: str) -> list[str]:
    """Extract all known epistemic markers appearing in text."""
    t = (text or "").lower()
    found: list[str] = []
    for marker in EPISTEMIC_MARKERS:
        if _contains_phrase(t, marker):
            found.append(marker)
    return found


def primary_epistemic_marker(text: str) -> str:
    """
    Return one marker for confidence mapping.

    ACL'25 uses a single marker token W when computing marker confidence.
    If none found, return a dedicated NO_MARKER bucket.
    """
    markers = extract_epistemic_markers(text)
    return markers[0] if markers else NO_MARKER


def build_marker_confidence_profile(
    samples: Iterable[tuple[str, int]],
    min_occurrences: int = 10,
) -> Dict[str, MarkerStats]:
    """
    Build marker confidence profile from (answer_text, is_correct) pairs.

    Marker confidence is defined as:
      Conf(W) = #correct answers containing W / #answers containing W
    as in Liu et al. (ACL 2025). Markers below min_occurrences are filtered out.
    """
    agg: dict[str, dict[str, int]] = defaultdict(lambda: {"occurrences": 0, "correct": 0})
    for answer_text, is_correct in samples:
        marker = primary_epistemic_marker(answer_text)
        agg[marker]["occurrences"] += 1
        agg[marker]["correct"] += int(bool(is_correct))

    profile: Dict[str, MarkerStats] = {}
    for marker, stats in agg.items():
        occ = stats["occurrences"]
        if occ < min_occurrences:
            continue
        cor = stats["correct"]
        profile[marker] = {
            "occurrences": occ,
            "correct": cor,
            "confidence": cor / occ if occ else 0.0,
        }
    return profile


def marker_confidence_from_profile(
    text: str,
    profile: Mapping[str, MarkerStats],
    default_confidence: float = 0.5,
) -> float:
    """Estimate confidence for text using a precomputed marker profile."""
    marker = primary_epistemic_marker(text)
    stats = profile.get(marker)
    if stats is None:
        return default_confidence
    return float(stats["confidence"])


def detect_linguistic_markers(text: str) -> Dict[str, int]:
    """
    Backward-compatible marker buckets used by the existing data pipeline.

    strong  -> high uncertainty markers
    medium  -> moderate uncertainty markers
    weak    -> low uncertainty / high-confidence verbal markers
    """
    counts: Dict[str, int] = {"strong": 0, "medium": 0, "weak": 0}
    for marker in extract_epistemic_markers(text):
        if marker in STRONG_UNCERTAINTY:
            counts["strong"] += 1
        elif marker in MEDIUM_UNCERTAINTY:
            counts["medium"] += 1
        elif marker in WEAK_UNCERTAINTY:
            counts["weak"] += 1
    return counts


def linguistic_confidence_level(text: str) -> str:
    counts = detect_linguistic_markers(text)
    if counts["strong"] > 0:
        return "low"
    if counts["medium"] > 0:
        return "medium"
    if counts["weak"] > 0:
        return "high"
    return "medium"


def linguistic_confidence_score(text: str) -> float:
    """
    Compatibility score in [0, 1] for current pipeline.

    Prefer marker_confidence_from_profile(...) for paper-faithful experiments.
    """
    counts = detect_linguistic_markers(text)
    score = 0.5
    score -= 0.25 * counts["strong"]
    score -= 0.10 * counts["medium"]
    score += 0.10 * counts["weak"]
    return max(0.0, min(1.0, score))

