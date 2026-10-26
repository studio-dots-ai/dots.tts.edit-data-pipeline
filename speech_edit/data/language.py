"""Language helpers for data-agent sampling and service calls."""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Optional


SUPPORTED_LANGUAGES = ("zh", "en")
OTHER_LANGUAGE = "others"

_ALLOWED_EN_PUNCTUATION = frozenset(
    set(string.punctuation)
    | {
        "“",
        "”",
        "‘",
        "’",
        "«",
        "»",
        "–",
        "—",
        "…",
    }
)
_ZH_CODES = {
    "zh",
    "zho",
    "chi",
    "cmn",
    "cn",
    "chinese",
    "mandarin",
    "zh-cn",
    "zh_hans",
    "zh-hans",
}
_EN_CODES = {
    "en",
    "eng",
    "english",
    "en-us",
    "en-gb",
}
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _contains_han_character(char: str) -> bool:
    codepoint = ord(char)
    return 0x4E00 <= codepoint <= 0x9FFF


def _is_roman_character_or_punctuation(char: str) -> bool:
    if char.isspace():
        return True
    if char in _ALLOWED_EN_PUNCTUATION:
        return True
    if char.isascii():
        return char.isalnum()

    unicode_name = unicodedata.name(char, "")
    return "LATIN" in unicode_name and unicodedata.category(char)[0] in {"L", "N"}


def classify_text_language(text: str) -> str:
    """Classify text into zh / en / others using the pipeline sampling rule.

    The `en` bucket intentionally means Roman-script text, matching the user
    requirement that all Roman characters plus related punctuation be treated as
    English for sampling purposes.
    """
    normalized = (text or "").strip()
    if not normalized:
        return OTHER_LANGUAGE

    if any(_contains_han_character(char) for char in normalized):
        return "zh"

    if all(_is_roman_character_or_punctuation(char) for char in normalized):
        return "en"

    return OTHER_LANGUAGE


def has_cjk(text: Optional[str]) -> bool:
    return bool(text and _CJK_RE.search(text))


def normalize_language_code(
    language: Optional[str],
    *,
    text: Optional[str] = None,
    default: str = "eng",
) -> str:
    """Return the pipeline language code, normally ``zho`` or ``eng``."""
    if language:
        normalized = str(language).strip().lower().replace("_", "-")
        if normalized in _ZH_CODES:
            return "zho"
        if normalized in _EN_CODES:
            return "eng"
        return normalized
    return "zho" if has_cjk(text) else default


def alignment_language_code(language: Optional[str], *, text: Optional[str] = None) -> str:
    return normalize_language_code(language, text=text, default="eng")


def wer_language_code(
    language: Optional[str],
    *,
    reference_text: Optional[str] = None,
    hypothesis_text: Optional[str] = None,
    prefer_auto: bool = False,
) -> str:
    if prefer_auto:
        return "auto"
    text = " ".join(part for part in (reference_text, hypothesis_text) if part)
    code = normalize_language_code(language, text=text or None, default="eng")
    return "zh" if code == "zho" else "en" if code == "eng" else code


def asr_language_name(
    language: Optional[str],
    *,
    text: Optional[str] = None,
    prefer_auto: bool = False,
) -> Optional[str]:
    if prefer_auto:
        return None
    code = normalize_language_code(language, text=text, default="eng")
    if code == "zho":
        return "Chinese"
    if code == "eng":
        return "English"
    return None
