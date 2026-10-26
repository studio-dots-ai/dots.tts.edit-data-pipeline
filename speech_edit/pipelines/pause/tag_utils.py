"""
Tag utilities for the pause editing pipeline.

Pause tag format (inline, self-closing at non-terminal boundaries):
    <pause act="ins" level="1"/>  — insert a short pause (~200ms)
    <pause act="ins" level="2"/>  — insert a medium pause (~500ms)
    <pause act="ins" level="3"/>  — insert a long pause (~800ms)
    <pause act="red" level="N"/>  — reduce a pause by the level duration

Tags are point markers placed at interior word boundaries. They do NOT wrap
text.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS = {"ins"}
INSTRUCTION_ACTIONS = {"ins", "red"}

PAUSE_LEVEL_TO_DURATION_MS = {
    1: 200,
    2: 500,
    3: 800,
}
VALID_LEVELS = set(PAUSE_LEVEL_TO_DURATION_MS)

# Punctuation that can surround valid pause boundaries.
CN_PAUSE_PUNCTUATION = set("，。、；")
EN_PAUSE_PUNCTUATION = set(".,;")
ALL_PAUSE_PUNCTUATION = CN_PAUSE_PUNCTUATION | EN_PAUSE_PUNCTUATION
DASH_BOUNDARY_CHARS = set("-—–")

# Sentence-final punctuation.
CN_FINAL_PUNCTUATION = set("。！？")
EN_FINAL_PUNCTUATION = set(".!?")
ALL_FINAL_PUNCTUATION = CN_FINAL_PUNCTUATION | EN_FINAL_PUNCTUATION
ALL_COMMON_PUNCTUATION = ALL_PAUSE_PUNCTUATION | ALL_FINAL_PUNCTUATION | set(
    '"\'()[]{}:：、，。！？；!?,.;'
)

# ---------------------------------------------------------------------------
# Tag regex
# ---------------------------------------------------------------------------

# Matches the canonical profile operation tag: <pause act="ins" level="1"/>
PAUSE_TAG_PATTERN = re.compile(
    r'<pause\s+'
    r'act="(ins)"'
    r'\s+level="([1-3])"'
    r'\s*/>'
)

# Matches structured instruction tags used in both directions.
INSTRUCTION_PAUSE_TAG_PATTERN = re.compile(
    r'<pause\s+'
    r'act="(ins|red)"'
    r'\s+level="([1-3])"'
    r'\s*/>'
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PauseTag:
    """A parsed pause tag with its position and context."""

    char_idx: int  # Position in normed_text where the tag sits (between chars)
    act: str  # "ins" for generated profiles; service instructions also allow "red"
    level: int  # Discrete pause level in [1, 2, 3]
    delta_dur_ms: int  # Resolved duration in ms from the pause level
    left_context: str  # Text immediately before the tag (for context)
    right_context: str  # Text immediately after the tag (for context)
    punctuation: Optional[str]  # The punctuation char immediately before the tag


@dataclass
class PauseEditProfile:
    """A single pause edit profile from LLM output."""

    profile_id: int
    operation: str  # normed_text with punctuation-aligned insertion tags
    st_inst: Dict[str, List[str]]  # {"increase_pause": [...], "decrease_pause": [...]}
    nl_inst: Dict[str, List[str]]  # {"increase_pause": [...], "decrease_pause": [...]}
    tags: List[PauseTag]  # Parsed tag info


@dataclass
class PauseEditGroup:
    """Output from Stage I for one batch."""

    normed_text: str
    profiles: List[PauseEditProfile]
    duration_pool: List[int]  # Fixed level-to-duration mapping values [200, 500, 800]
    num_operations: int  # Requested number of operations per profile
    original_audio: bytes  # WAV bytes paired with normed_text for F5 editing
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------


def _pause_text_coordinates(text: str) -> Tuple[str, List[int]]:
    """Normalize formatting whitespace while retaining original character ends.

    Whitespace between Latin words remains a separator. Around punctuation or
    CJK characters it is optional; letter case and all non-space characters
    are preserved. The returned offsets always address the original text.
    """
    characters = []
    ends = []
    punctuation = ALL_COMMON_PUNCTUATION | DASH_BOUNDARY_CHARS
    for match in re.finditer(r"\s+|\S", text):
        value = match.group()
        if value.isspace():
            left = text[match.start() - 1] if match.start() else ""
            right = text[match.end()] if match.end() < len(text) else ""
            if (
                not left or not right
                or left in punctuation or right in punctuation
                or _is_chinese_char(left) or _is_chinese_char(right)
            ):
                continue
            value = " "
        characters.append(value)
        ends.append(match.end())
    return "".join(characters), ends


def _parse_tags_with_pattern(
    tagged_text: str, normed_text: str, pattern: re.Pattern
) -> List[PauseTag]:
    plain_text = strip_pause_tags(tagged_text)
    normalized, plain_ends = _pause_text_coordinates(plain_text)
    expected, reference_ends = _pause_text_coordinates(normed_text)
    if normalized != expected:
        raise ValueError(
            f"Text mismatch after stripping tags.\n"
            f"  Expected: {expected!r}\n"
            f"  Got:      {normalized!r}"
        )
    tags = []
    for match in pattern.finditer(tagged_text):
        plain_before = strip_pause_tags(tagged_text[:match.start()]).rstrip()
        character_count = bisect_right(plain_ends, len(plain_before))
        char_idx = reference_ends[character_count - 1] if character_count else 0
        left_context = normed_text[:char_idx].rstrip()
        right_context = normed_text[char_idx:].lstrip()
        punctuation = (
            left_context[-1]
            if left_context and left_context[-1] in ALL_PAUSE_PUNCTUATION
            else None
        )
        level = int(match.group(2))
        tags.append(
            PauseTag(
                char_idx=char_idx,
                act=match.group(1),
                level=level,
                delta_dur_ms=PAUSE_LEVEL_TO_DURATION_MS[level],
                left_context=left_context[-30:],
                right_context=right_context[:30],
                punctuation=punctuation,
            )
        )
    return tags


def parse_pause_tags(tagged_text: str, normed_text: str) -> List[PauseTag]:
    """Extract insertion tags at exact coordinates in the supplied plain text.

    Formatting whitespace may differ, but words, punctuation and letter case
    must match. Coordinates point immediately after the preceding non-space
    character in normed_text, including when it contains repeated whitespace.
    """
    return _parse_tags_with_pattern(tagged_text, normed_text, PAUSE_TAG_PATTERN)


def parse_instruction_pause_tags(tagged_text: str, normed_text: str) -> List[PauseTag]:
    """Extract either instruction action at coordinates in the supplied text."""
    return _parse_tags_with_pattern(tagged_text, normed_text, INSTRUCTION_PAUSE_TAG_PATTERN)


def strip_pause_tags(tagged_text: str) -> str:
    """Remove all pause tags from text, returning plain text."""
    text = PAUSE_TAG_PATTERN.sub("", tagged_text)
    return INSTRUCTION_PAUSE_TAG_PATTERN.sub("", text)


# ---------------------------------------------------------------------------
# Tag validation
# ---------------------------------------------------------------------------


def _is_at_word_boundary(text: str, char_idx: int) -> bool:
    """Use one position per spoken boundary, after any attached punctuation."""
    if char_idx <= 0 or char_idx >= len(text):
        return True  # The validator reports start/end positions separately.
    left = text[char_idx - 1]
    right_text = text[char_idx:].lstrip()
    if not right_text:
        return True
    right = right_text[0]
    punctuation = ALL_COMMON_PUNCTUATION | DASH_BOUNDARY_CHARS
    # A punctuation run belongs to the preceding boundary, not two positions.
    if left.isspace() or right in punctuation:
        return False
    # Apostrophes inside words do not create spoken boundaries.
    if (
        left in {"'", "’"} and char_idx >= 2
        and text[char_idx - 2].isalnum() and text[char_idx].isalnum()
    ):
        return False
    if not any(c.isalnum() for c in text[:char_idx]):
        return False
    if not any(c.isalnum() for c in right_text):
        return False
    return (
        text[char_idx].isspace()
        or left in punctuation
        or _is_chinese_char(left)
        or _is_chinese_char(right)
    )


def pause_boundaries(text: str) -> List[Tuple[int, str]]:
    """Enumerate distinct feasible interior positions and their tag types."""
    return [
        (index, "punct" if text[index - 1] in ALL_PAUSE_PUNCTUATION else "nonpunct")
        for index in range(1, len(text))
        if not text[index - 1].isspace()
        and text[index:].strip()
        and _is_at_word_boundary(text, index)
    ]


def _is_chinese_char(c: str) -> bool:
    """Check if a character is Chinese/CJK."""
    cp = ord(c)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def _contains_cjk(text: str) -> bool:
    return any(_is_chinese_char(char) for char in text)


def _is_sentence_final_punctuation(text: str, char_idx: int) -> bool:
    """Check if the punctuation at char_idx-1 is sentence-final.

    Sentence-final means it's the last punctuation mark in the text
    (ignoring trailing whitespace).
    """
    if char_idx <= 0 or char_idx - 1 >= len(text):
        return False

    punc_char = text[char_idx - 1]
    if punc_char not in ALL_PAUSE_PUNCTUATION:
        return False

    # Check if there's any non-whitespace content after this position
    remaining = text[char_idx:].strip() if char_idx < len(text) else ""
    return len(remaining) == 0


def validate_pause_tags(
    tagged_text: str,
    normed_text: str,
    *,
    allowed_actions: Optional[set[str]] = None,
) -> Tuple[bool, str]:
    """Validate that pause tags in tagged_text are well-formed and consistent.

    Checks:
    1. Stripped text matches normed_text
    2. Tags are at interior word boundaries
    3. Tags are not at the start or end of the text
    4. All tag actions are valid
    5. All tags use one of the supported discrete levels

    Args:
        tagged_text: Text with <pause .../> tags.
        normed_text: The expected plain text without tags.

    Returns:
        (is_valid, error_message)
    """
    actions = VALID_ACTIONS if allowed_actions is None else set(allowed_actions)
    try:
        tags = _parse_tags_with_pattern(
            tagged_text, normed_text, INSTRUCTION_PAUSE_TAG_PATTERN
        )
    except ValueError as exc:
        return False, str(exc)
    if not tags:
        return False, "No pause tags found in tagged text"

    char_positions = [tag.char_idx for tag in tags]
    if len(set(char_positions)) != len(char_positions):
        return False, (
            "Duplicate pause insertion positions are not allowed within one profile: "
            f"{char_positions}"
        )

    for tag in tags:
        # Validate action
        if tag.act not in actions:
            return False, f"Invalid action: {tag.act}"

        # Validate interior word boundary
        if not _is_at_word_boundary(normed_text, tag.char_idx):
            return False, (
                f"Tag at position {tag.char_idx} is not at a word boundary. "
                f"Context: ...{normed_text[max(0, tag.char_idx - 5):tag.char_idx + 5]}..."
            )

        if tag.char_idx <= 0:
            return False, (
                f"{tag.act} tag at position {tag.char_idx} must not be placed at the start "
                f"of the text. Left context: {tag.left_context!r}"
            )
        if not normed_text[tag.char_idx:].strip():
            return False, (
                f"{tag.act} tag at position {tag.char_idx} must not be placed at the end "
                f"of the text."
            )
        if tag.level not in VALID_LEVELS:
            return False, f"Unsupported pause level: {tag.level}"
        expected_dur = PAUSE_LEVEL_TO_DURATION_MS[tag.level]
        if tag.delta_dur_ms != expected_dur:
            return False, (
                f"Pause level {tag.level} must map to {expected_dur}ms, "
                f"got {tag.delta_dur_ms}ms"
            )

    return True, ""


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison (whitespace + contractions + case)."""
    # Collapse multiple spaces to single
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*--\s*", "--", text)
    punctuation_chars = "".join(re.escape(ch) for ch in sorted(ALL_COMMON_PUNCTUATION))
    text = re.sub(rf"\s*([{punctuation_chars}])\s*", r"\1", text)
    # Normalize common contractions to expanded form (spoken-form)
    # This matches LLM normalization behavior
    _CONTRACTIONS = {
        "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
        "it's": "it is", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
        "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
        "we'd": "we would", "they'd": "they would",
        "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
        "we'll": "we will", "they'll": "they will",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "wouldn't": "would not",
        "can't": "cannot", "couldn't": "could not",
        "shouldn't": "should not", "mightn't": "might not", "mustn't": "must not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "let's": "let us", "that's": "that is", "who's": "who is", "what's": "what is",
        "here's": "here is", "there's": "there is", "where's": "where is",
        "how's": "how is", "ain't": "is not",
    }
    text_lower = text.lower()
    for contraction, expansion in _CONTRACTIONS.items():
        text_lower = re.sub(rf"\b{re.escape(contraction)}\b", expansion, text_lower)
    if _contains_cjk(text_lower):
        text_lower = re.sub(r"\s+", "", text_lower)
    return text_lower


# ---------------------------------------------------------------------------
# Tag conversion utilities
# ---------------------------------------------------------------------------


def compute_generated_text(normed_text: str, tags: List[PauseTag]) -> str:
    """Apply pause operations to produce the generated transcript.

    For the current insertion-only pause-edit task, the human-facing generated
    transcript should stay identical to the original normalized text. The
    comma-like proxy marker is only used inside the F5 edit operations, not in
    the saved/evaluated transcript string.

    Args:
        normed_text: Original normalized text.
        tags: Parsed pause tags.

    Returns:
        Unchanged normalized text.
    """
    return normed_text


def _select_comma(text: str) -> str:
    """Select the appropriate comma character based on text language.

    If text contains CJK characters, use Chinese comma; otherwise English.
    """
    cjk_count = sum(1 for c in text if _is_chinese_char(c))
    total_alpha = sum(1 for c in text if c.isalpha())
    if cjk_count > total_alpha * 0.3:
        return "，"
    return ","


def make_text_versions(
    normed_text: str, tags: List[PauseTag]
) -> Dict[str, str]:
    """Produce dual-version text for data saving.

    Pause-insert-only strategy keeps punctuation unchanged, so a single text
    version is always sufficient.

    Args:
        normed_text: Original normalized text.
        tags: Parsed pause tags.

    Returns:
        Dict with a single "text" key.
    """
    return {"text": normed_text}


def reverse_operation(tagged_text: str) -> str:
    """Reverse pause direction for metadata/instruction use.

    - `ins` -> `red`
    - `red` -> `ins`

    This helper is used for reverse-direction metadata and instructions.
    """

    def _reverse_tag(m: re.Match) -> str:
        act = m.group(1)
        level = m.group(2)
        new_act = "red" if act == "ins" else "ins"
        return f'<pause act="{new_act}" level="{level}"/>'

    return INSTRUCTION_PAUSE_TAG_PATTERN.sub(_reverse_tag, tagged_text)


def _pause_level_label(level: int) -> Tuple[str, str]:
    if level == 1:
        return "short", "短"
    if level == 2:
        return "medium", "中等"
    return "long", "长"


def build_pause_nl_instructions(tagged_text: str) -> List[str]:
    """Build deterministic absolute pause-edit instructions from structured tags."""
    tags = parse_instruction_pause_tags(tagged_text, strip_pause_tags(tagged_text))
    if not tags:
        return []
    english_parts: List[str] = []
    chinese_parts: List[str] = []
    for tag in tags:
        en_level, zh_level = _pause_level_label(tag.level)
        left = tag.left_context.strip()
        right = tag.right_context.strip()
        if tag.act == "red":
            english_parts.append(
                f"reduce the {en_level} pause between '{left}' and '{right}'"
            )
            chinese_parts.append(f"缩短'{left}'和'{right}'之间的{zh_level}停顿")
        else:
            english_parts.append(
                f"add a {en_level} pause between '{left}' and '{right}'"
            )
            chinese_parts.append(f"在'{left}'和'{right}'之间增加一个{zh_level}停顿")
    english_summary = "; ".join(english_parts)
    chinese_summary = "；".join(chinese_parts)
    actions = {tag.act for tag in tags}
    if actions == {"red"}:
        return [
            f"Reduce the pause timing: {english_summary}.",
            f"Make the marked pause shorter: {english_summary}.",
            f"减少停顿：{chinese_summary}。",
            f"把标记处的停顿缩短：{chinese_summary}。",
        ]
    if len(actions) > 1:
        return [
            f"Edit the pause timing: {english_summary}.",
            f"Apply the marked pause changes: {english_summary}.",
            f"编辑停顿：{chinese_summary}。",
            f"按标记调整停顿：{chinese_summary}。",
        ]
    return [
        f"Increase the pause timing: {english_summary}.",
        f"Make the marked pause longer: {english_summary}.",
        f"增加停顿：{chinese_summary}。",
        f"把标记处的停顿加长：{chinese_summary}。",
    ]


def _level_from_duration_ms(delta_dur_ms: int) -> int:
    """Map a duration in ms to the nearest supported pause level."""
    return min(
        PAUSE_LEVEL_TO_DURATION_MS,
        key=lambda level: abs(PAUSE_LEVEL_TO_DURATION_MS[level] - delta_dur_ms),
    )


def inject_duration(tagged_text: str, delta_dur_ms: int) -> str:
    """Add a discrete pause level to insertion tags that don't have one.

    The provided duration is mapped to the nearest supported level.
    """

    level = _level_from_duration_ms(delta_dur_ms)
    tagged_text = re.sub(
        r'<pause\s+act="ins"\s+delta_dur="\d+ms"\s*/>',
        f'<pause act="ins" level="{level}"/>',
        tagged_text,
    )
    tagged_text = re.sub(
        r'<pause\s+act="ins"\s*/>',
        f'<pause act="ins" level="{level}"/>',
        tagged_text,
    )
    return tagged_text


def tags_to_aligner_queries(
    normed_text: str, tags: List[PauseTag]
) -> List[Dict[str, str]]:
    """Convert pause tag positions to aligner queries.

    For each tag, produces queries for the left and right anchor words
    around the pause position.  Each query is a *full-text* string with
    ``<q>...</q>`` tags wrapped around the anchor word so that the
    aligner can map it back to the exact character positions in
    ``normed_text``.

    Args:
        normed_text: Plain text without tags.
        tags: Parsed pause tags.

    Returns:
        List of dicts with:
        - ``left_query`` / ``right_query`` — raw anchor words (backward-compat).
        - ``left_query_full`` / ``right_query_full`` — full text strings with
          ``<q>`` tags for the aligner.
    """
    queries = []
    for tag in tags:
        left_word = _extract_anchor_word(normed_text, tag.char_idx, direction="left")
        right_word = _extract_anchor_word(normed_text, tag.char_idx, direction="right")
        queries.append(
            {
                "left_query": left_word,
                "right_query": right_word,
                "left_query_full": _build_aligner_query(normed_text, left_word, tag.char_idx, direction="left"),
                "right_query_full": _build_aligner_query(normed_text, right_word, tag.char_idx, direction="right"),
                "char_idx": tag.char_idx,
                "act": tag.act,
                "delta_dur_ms": tag.delta_dur_ms,
            }
        )
    return queries


def tags_to_boundary_queries(
    normed_text: str, tags: List[PauseTag]
) -> List[Dict[str, str]]:
    """Convert pause tag positions to boundary-point aligner queries.

    Each returned query inserts exactly one empty `<q></q>` marker at the pause
    boundary. This allows the aligner backend to return the timestamp of the
    insertion point directly.
    """
    queries = []
    for tag in tags:
        queries.append(
            {
                "boundary_query_full": _build_boundary_query(normed_text, tag.char_idx),
                "char_idx": tag.char_idx,
                "act": tag.act,
                "delta_dur_ms": tag.delta_dur_ms,
            }
        )
    return queries


def _build_boundary_query(text: str, char_idx: int) -> str:
    """Build a full-text `<q></q>` boundary query at char_idx."""
    return text[:char_idx] + "<q></q>" + text[char_idx:]


def _build_aligner_query(
    text: str, anchor_word: str, char_idx: int, direction: str
) -> str:
    """Build a full-text aligner query with ``<q>`` tags around *anchor_word*.

    The aligner requires that removing ``<q>`` / ``</q>`` tags recovers the
    exact ``text`` used for alignment.  We therefore embed the anchor word
    inside the complete text rather than returning the word alone.
    """
    if not anchor_word:
        return text

    if direction == "left":
        # Search backwards from char_idx for the anchor word.
        search_end = char_idx
        while search_end > 0 and (text[search_end - 1].isspace() or text[search_end - 1] in ALL_PAUSE_PUNCTUATION):
            search_end -= 1
        idx = text.rfind(anchor_word, max(0, search_end - len(anchor_word)), search_end)
        if idx < 0:
            idx = text.rfind(anchor_word, 0, char_idx)
        if idx < 0:
            return text
        return text[:idx] + f"<q>{anchor_word}</q>" + text[idx + len(anchor_word):]
    else:  # right
        search_start = char_idx
        while search_start < len(text) and (text[search_start].isspace() or text[search_start] in ALL_PAUSE_PUNCTUATION):
            search_start += 1
        idx = text.find(anchor_word, search_start)
        if idx < 0:
            return text
        return text[:idx] + f"<q>{anchor_word}</q>" + text[idx + len(anchor_word):]


def _extract_anchor_word(
    text: str, char_idx: int, direction: str
) -> str:
    """Extract the anchor word at a position in a given direction.

    Args:
        text: Plain text.
        char_idx: Position between characters.
        direction: "left" or "right".

    Returns:
        The word adjacent to the position.
    """
    if direction == "left":
        # Walk backwards from char_idx to find the word
        end = char_idx
        # Skip trailing spaces/punctuation
        while end > 0 and (text[end - 1].isspace() or text[end - 1] in ALL_PAUSE_PUNCTUATION):
            end -= 1
        if end == 0:
            return ""
        # Find word start
        start = end
        while start > 0 and not text[start - 1].isspace() and text[start - 1] not in ALL_PAUSE_PUNCTUATION:
            # For CJK, each char is a word
            if _is_chinese_char(text[start - 1]):
                start -= 1
                break
            start -= 1
        return text[start:end]
    else:  # right
        # Walk forward from char_idx to find the word
        start = char_idx
        # Skip leading spaces/punctuation
        while start < len(text) and (text[start].isspace() or text[start] in ALL_PAUSE_PUNCTUATION):
            start += 1
        if start >= len(text):
            return ""
        # Find word end
        end = start
        while end < len(text) and not text[end].isspace() and text[end] not in ALL_PAUSE_PUNCTUATION:
            # For CJK, each char is a word
            if _is_chinese_char(text[end]):
                end += 1
                break
            end += 1
        return text[start:end]
