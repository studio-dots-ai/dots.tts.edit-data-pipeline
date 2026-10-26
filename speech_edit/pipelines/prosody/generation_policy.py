"""Risk filtering for prosody edit generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from .tag_utils import RATE_TAG, parse_tags

logger = logging.getLogger(__name__)

_STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "can",
    "could",
    "i",
    "me",
    "my",
    "we",
    "us",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "it",
    "its",
    "they",
    "them",
    "their",
    "this",
    "that",
    "these",
    "those",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "if",
    "or",
    "and",
    "but",
    "not",
    "no",
    "so",
    "up",
}


@dataclass
class RiskFilterResult:
    """Result of a risk filter evaluation."""

    passed: bool
    risk_type: str
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def short_rate_risk(
    tagged_text: str,
    *,
    min_words_for_rate: int = 1,
    min_chars_for_rate: int = 2,
) -> RiskFilterResult:
    """Reject rate edits that are too short to resynthesize reliably."""
    try:
        segments = parse_tags(tagged_text)
    except ValueError as exc:
        return RiskFilterResult(False, "parse_error", str(exc))

    use_cjk_policy = _contains_cjk(tagged_text)
    for seg in segments:
        if seg.tag_type != RATE_TAG:
            continue
        words = [word for word in seg.text.strip().split() if word]
        clean_chars = sum(1 for ch in seg.text if ch.isalnum())
        if use_cjk_policy:
            too_short = clean_chars < min_chars_for_rate
        else:
            too_short = len(words) < min_words_for_rate
        if too_short:
            return RiskFilterResult(
                passed=False,
                risk_type="short_rate_risk",
                reason=f"Rate edit span too short: {seg.text!r}",
                metadata={
                    "text": seg.text,
                    "word_count": len(words),
                    "char_count": clean_chars,
                    "policy": "cjk" if use_cjk_policy else "word",
                },
            )
    return RiskFilterResult(True, "short_rate_risk")


def tiny_stopword_rate_risk(tagged_text: str) -> RiskFilterResult:
    """Reject rate edits that target only tiny stopword spans."""
    try:
        segments = parse_tags(tagged_text)
    except ValueError as exc:
        return RiskFilterResult(False, "parse_error", str(exc))

    for seg in segments:
        if seg.tag_type != RATE_TAG:
            continue
        words = [
            word.strip(".,!?;:\"'()[]{}").lower()
            for word in seg.text.strip().split()
            if word.strip(".,!?;:\"'()[]{}")
        ]
        if words and len(words) <= 2 and all(word in _STOPWORDS for word in words):
            return RiskFilterResult(
                passed=False,
                risk_type="tiny_stopword_rate_risk",
                reason=f"Rate edit targets only stopwords: {seg.text!r}",
                metadata={"words": words},
            )
    return RiskFilterResult(True, "tiny_stopword_rate_risk")


def run_all_filters(tagged_text: str) -> List[RiskFilterResult]:
    """Run all risk filters and return only failures."""
    failures: List[RiskFilterResult] = []
    for result in [
        short_rate_risk(tagged_text),
        tiny_stopword_rate_risk(tagged_text),
    ]:
        if not result.passed:
            failures.append(result)
    return failures
