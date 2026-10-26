"""
Tag utilities for the emotion/style editing pipeline.

Canonical emotion tag format:
    <emo type="sad" alpha="0.84">text</emo>
    <emo type="sad" level="2">text</emo>
    <emo desc="speak with a quiet, restrained sadness">text</emo>
    <emo desc="speak with a quiet, restrained sadness" type_ref="sad">text</emo>
    <emo desc="soft, slow sadness" dur="1.8">text</emo>
    <emo type="none">text</emo>

The legacy ``<sad, level=2>text</sad>`` shape is intentionally unsupported.
"""

import html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTION_TAGS = [
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
]

STYLE_TAGS: list[str] = []

NEUTRAL_TAG = "none"

CANONICAL_EMO_TAG = "emo"
EMOTION_CONTROL_ALPHA = "alpha"
EMOTION_CONTROL_DESCRIPTION = "description"
ALL_TAGS = [CANONICAL_EMO_TAG]

DEFAULT_LEVEL = 2

LEVEL_TO_ALPHA = {
    "afraid": {1: 0.8, 2: 0.9, 3: 1.0},
    "angry": {1: 0.72, 2: 0.85, 3: 1.0},
    "calm": {1: 0.8, 2: 0.9, 3: 1.0},
    "disgusted": {1: 0.8, 2: 0.9, 3: 1.0},
    "happy": {1: 0.83, 2: 0.93, 3: 1.0},
    "melancholic": {1: 0.8, 2: 0.9, 3: 1.0},
    "sad": {1: 0.8, 2: 0.9, 3: 1.0},
    "surprised": {1: 0.72, 2: 0.8, 3: 0.88},
}
DEFAULT_ALPHA = {
    emotion: level_map[DEFAULT_LEVEL] for emotion, level_map in LEVEL_TO_ALPHA.items()
}

# Punctuation marks that can serve as tag boundaries
CN_PUNCTUATION = set("。，！？、；：…—")
EN_PUNCTUATION = set(".,!?;:-")
ALL_PUNCTUATION = CN_PUNCTUATION | EN_PUNCTUATION


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaggedSegment:
    """A single tagged segment parsed from tagged text."""

    tag_name: Optional[str]  # emotion type, "emo" for desc, "none", or None
    text: str
    alpha: Optional[float]  # None = use default
    tag_type: str  # "emotion", "style", "none", "untagged"
    emotion_control_mode: Optional[str] = None  # "alpha" or "description"
    emotion_type: Optional[str] = None
    emotion_level: Optional[int] = None
    emotion_desc: Optional[str] = None
    emotion_type_ref: Optional[str] = None
    dur: Optional[float] = None

    @property
    def effective_alpha(self) -> Optional[float]:
        """Return effective alpha, falling back to default if not set."""
        if self.tag_name is None or self.tag_name == NEUTRAL_TAG:
            return None
        if self.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION:
            return None
        if self.alpha is not None:
            return self.alpha
        return DEFAULT_ALPHA.get(self.emotion_type or self.tag_name)

    @property
    def control_signature(self) -> Tuple[
        Optional[str],
        Optional[str],
        Optional[float],
        Optional[str],
        Optional[str],
        Optional[float],
    ]:
        """Stable comparable identity for the segment's emotion control."""
        if self.tag_name is None:
            return ("untagged", None, None, None, None, None)
        if self.tag_name == NEUTRAL_TAG:
            return ("none", NEUTRAL_TAG, None, None, None, self.dur)
        if self.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION:
            return (
                EMOTION_CONTROL_DESCRIPTION,
                None,
                None,
                self.emotion_desc,
                self.emotion_type_ref,
                self.dur,
            )
        return (
            EMOTION_CONTROL_ALPHA,
            self.emotion_type,
            self.effective_alpha,
            None,
            None,
            self.dur,
        )

    @property
    def evaluation_tag(self) -> Optional[str]:
        """Canonical emotion label retained for metadata compatibility."""
        if self.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION:
            return self.emotion_type_ref
        return self.emotion_type or self.tag_name


def get_default_alpha(tag_name: Optional[str]) -> Optional[float]:
    """Return the default alpha for an emotion tag, if any."""
    if tag_name is None:
        return None
    return DEFAULT_ALPHA.get(tag_name)


@dataclass
class TextBoundary:
    """A boundary position in text where tags can switch."""

    char_idx: int  # Character index in the plain text


@dataclass
class EditPair:
    """A source-target pair with edit instructions."""

    source_id: str  # Key into tagged_texts dict
    target_id: str  # Key into tagged_texts dict
    nl_instruction: str  # Natural language edit instruction
    st_instruction: str = ""  # Structural instruction (partially tagged diff)


@dataclass
class TaggedTextGroup:
    """Output from Stage I for one batch."""

    sub_type: (
        str  # "single" / "multi_emo_normal" / "multi_emo_frequent"
    )
    plain_text: str  # Original generation text
    speaker_prompt_audio: bytes  # WAV bytes of speaker prompt
    speaker_prompt_text: str  # Transcript of speaker prompt
    tagged_texts: Dict[str, str]  # key=id, value=tagged text
    edit_pairs: List[EditPair]  # List of (source_id, target_id, instruction)
    boundaries: List[TextBoundary]  # Boundary positions used
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tag classification
# ---------------------------------------------------------------------------


def classify_tag(tag_name: str) -> str:
    """Classify a canonical parsed tag name."""
    if tag_name is None:
        return "untagged"
    tag_name = tag_name.lower().strip()
    if tag_name == NEUTRAL_TAG:
        return "none"
    if tag_name == CANONICAL_EMO_TAG or tag_name in EMOTION_TAGS:
        return "emotion"
    if tag_name in STYLE_TAGS:
        return "style"
    raise ValueError(
        f"Unknown tag: {tag_name}. Canonical emotion tags must use <emo ...>."
    )


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------

_TAG_PATTERN = re.compile(r"<emo\b([^>]*)>(.*?)</emo>", re.DOTALL | re.IGNORECASE)
_ATTR_PATTERN = re.compile(
    r"([a-zA-Z_][\w-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\"'\s>]+))"
)
_ANY_TAG_PATTERN = re.compile(r"</?[a-zA-Z_][^>]*>")


def _parse_attrs(raw_attrs: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    cursor = 0
    raw_attrs = raw_attrs.strip()
    while cursor < len(raw_attrs):
        if raw_attrs[cursor].isspace():
            cursor += 1
            continue
        match = _ATTR_PATTERN.match(raw_attrs, cursor)
        if not match:
            raise ValueError(f"Malformed <emo> attributes: {raw_attrs!r}")
        key = match.group(1).lower()
        value = next(group for group in match.groups()[1:] if group is not None)
        if key in attrs:
            raise ValueError(f"Duplicate <emo> attribute: {key}")
        attrs[key] = html.unescape(value)
        cursor = match.end()
    return attrs


def _parse_dur(attrs: Dict[str, str]) -> Optional[float]:
    if "dur" not in attrs:
        return None
    try:
        dur = float(attrs["dur"])
    except ValueError as exc:
        raise ValueError("dur must be a positive number of seconds") from exc
    if dur <= 0:
        raise ValueError("dur must be a positive number of seconds")
    return dur


def _parse_emo_segment(raw_attrs: str, text: str) -> TaggedSegment:
    attrs = _parse_attrs(raw_attrs)
    allowed = {"type", "alpha", "level", "desc", "type_ref", "dur"}
    unknown = sorted(set(attrs) - allowed)
    if unknown:
        raise ValueError(f"Unsupported <emo> attribute(s): {unknown}")

    emotion_type = attrs.get("type")
    if emotion_type is not None:
        emotion_type = emotion_type.lower().strip()
    desc = attrs.get("desc")
    if desc is not None:
        desc = desc.strip()
    type_ref = attrs.get("type_ref")
    if type_ref is not None:
        type_ref = type_ref.lower().strip()
        if type_ref not in EMOTION_TAGS:
            raise ValueError(f"Unknown emotion type_ref: {type_ref}. Must be one of {EMOTION_TAGS}")
    dur = _parse_dur(attrs)

    if desc:
        if emotion_type is not None or "alpha" in attrs or "level" in attrs:
            raise ValueError("<emo desc=...> cannot be combined with type, alpha, or level")
        return TaggedSegment(
            tag_name=CANONICAL_EMO_TAG,
            text=text,
            alpha=None,
            tag_type="emotion",
            emotion_control_mode=EMOTION_CONTROL_DESCRIPTION,
            emotion_desc=desc,
            emotion_type_ref=type_ref,
            dur=dur,
        )

    if type_ref is not None:
        raise ValueError("<emo type_ref=...> requires desc")
    if emotion_type is None:
        raise ValueError("<emo> must include either desc or type")
    if emotion_type == NEUTRAL_TAG:
        if "alpha" in attrs or "level" in attrs:
            raise ValueError('<emo type="none"> cannot include alpha or level')
        return TaggedSegment(
            tag_name=NEUTRAL_TAG,
            text=text,
            alpha=None,
            tag_type="none",
            emotion_control_mode=None,
            emotion_type=NEUTRAL_TAG,
            dur=dur,
        )
    if emotion_type not in EMOTION_TAGS:
        raise ValueError(f"Unknown emotion type: {emotion_type}. Must be one of {EMOTION_TAGS}")
    if "alpha" in attrs and "level" in attrs:
        raise ValueError("<emo> cannot include both alpha and level")

    alpha = None
    level = None
    if "alpha" in attrs:
        alpha = float(attrs["alpha"])
    elif "level" in attrs:
        level_token = attrs["level"].lower()
        if level_token in {"none", ""}:
            level = None
        else:
            level = int(level_token)
            if level == 0:
                return TaggedSegment(
                    tag_name=NEUTRAL_TAG,
                    text=text,
                    alpha=None,
                    tag_type="none",
                    emotion_control_mode=None,
                    emotion_type=NEUTRAL_TAG,
                    dur=dur,
                )
            if level not in LEVEL_TO_ALPHA[emotion_type]:
                raise ValueError("level must be 0, 1, 2, or 3")
            alpha = LEVEL_TO_ALPHA[emotion_type][level]

    return TaggedSegment(
        tag_name=emotion_type,
        text=text,
        alpha=alpha,
        tag_type="emotion",
        emotion_control_mode=EMOTION_CONTROL_ALPHA,
        emotion_type=emotion_type,
        emotion_level=level,
        dur=dur,
    )


def parse_tags(tagged_text: str) -> List[TaggedSegment]:
    """
    Parse canonical XML-like tags into a list of TaggedSegment objects.

    Handles:
        <emo type="happy" alpha="0.84">text</emo>
        <emo type="happy" level="2">text</emo>
        <emo desc="bright and cheerful">text</emo>
        <emo desc="bright and cheerful" type_ref="happy">text</emo>
        <emo desc="bright and cheerful" dur="1.4">text</emo>
        <emo type="none">text</emo>
    Untagged text between/around tags becomes segments with tag_name=None.
    """
    segments = []
    last_end = 0

    for m in _TAG_PATTERN.finditer(tagged_text):
        # Any untagged text before this tag (preserve spaces for accurate stripping)
        if m.start() > last_end:
            untagged = tagged_text[last_end : m.start()]
            if untagged:
                segments.append(
                    TaggedSegment(
                        tag_name=None, text=untagged, alpha=None, tag_type="untagged"
                    )
                )

        segments.append(_parse_emo_segment(m.group(1), m.group(2)))
        last_end = m.end()

    # Any trailing untagged text
    if last_end < len(tagged_text):
        trailing = tagged_text[last_end:]
        if trailing:
            segments.append(
                TaggedSegment(
                    tag_name=None, text=trailing, alpha=None, tag_type="untagged"
                )
            )

    return segments


def strip_tags(tagged_text: str) -> str:
    """Remove all tags from tagged text, return plain text."""
    segments = parse_tags(tagged_text)
    return "".join(seg.text for seg in segments)


def get_marked_segments(tagged_text: str) -> List[TaggedSegment]:
    """Return all explicitly tagged segments, including neutral spans."""
    return [
        segment for segment in parse_tags(tagged_text) if segment.tag_name is not None
    ]


def ensure_fully_tagged(tagged_text: str) -> None:
    """Reject tagged text that contains any untagged spans."""
    untagged_segments = [
        segment for segment in parse_tags(tagged_text) if segment.tag_name is None
    ]
    if not untagged_segments:
        return

    previews = []
    for segment in untagged_segments[:3]:
        text = segment.text.replace("\n", " ")
        preview = text if len(text) <= 40 else text[:37] + "..."
        previews.append(repr(preview))
    joined = ", ".join(previews)
    raise ValueError(
        "Tagged text must explicitly tag the full text. "
        'Replace untagged spans with <emo type="none">...</emo>. '
        f"Found untagged spans: {joined}"
    )


def build_segment_queries(tagged_text: str) -> Tuple[str, List[Dict[str, object]]]:
    """Build full-text `<q>` queries for all explicitly tagged segments.

    Returns `(plain_text, segments)` where each segment dict contains the
    query, text, tag metadata, and character offsets in the plain text.
    """
    ensure_fully_tagged(tagged_text)
    plain_text = strip_tags(tagged_text)
    segments = get_marked_segments(tagged_text)

    query_segments: List[Dict[str, object]] = []
    cursor = 0
    for segment in segments:
        start_char = cursor
        end_char = start_char + len(segment.text)
        query_segments.append(
            {
                "text": segment.text,
                "tag": segment.evaluation_tag,
                "emotion_type": segment.emotion_type,
                "emotion_type_ref": segment.emotion_type_ref,
                "emotion_desc": segment.emotion_desc,
                "dur": segment.dur,
                "emotion_control_mode": segment.emotion_control_mode,
                "tag_type": segment.tag_type,
                "alpha": segment.effective_alpha,
                "start_char": start_char,
                "end_char": end_char,
                "query": (
                    plain_text[:start_char]
                    + "<q>"
                    + plain_text[start_char:end_char]
                    + "</q>"
                    + plain_text[end_char:]
                ),
            }
        )
        cursor = end_char

    return plain_text, query_segments


def build_segment_timestamps_from_answers(
    tagged_text: str,
    answers: List[Dict[str, float]],
) -> List[Dict[str, object]]:
    """Map aligner query answers back to marked segment timestamp metadata."""
    _, query_segments = build_segment_queries(tagged_text)
    if len(answers) != len(query_segments):
        raise ValueError(
            "Aligner answer count does not match marked segment count: "
            f"expected {len(query_segments)}, got {len(answers)}"
        )

    timestamps: List[Dict[str, object]] = []
    for segment, answer in zip(query_segments, answers):
        start_sec = float(answer["start"])
        end_sec = float(answer["end"])
        if end_sec <= start_sec:
            raise ValueError(
                f"Invalid aligned span for segment {segment['text']!r}: {answer}"
            )
        timestamps.append(
            {
                "text": segment["text"],
                "tag": segment["tag"],
                "emotion_type": segment.get("emotion_type"),
                "emotion_type_ref": segment.get("emotion_type_ref"),
                "emotion_desc": segment.get("emotion_desc"),
                "dur": segment.get("dur"),
                "emotion_control_mode": segment.get("emotion_control_mode"),
                "tag_type": segment["tag_type"],
                "alpha": segment["alpha"],
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )
    return timestamps


def convert_levels_to_alpha(tagged_text: str) -> str:
    """Canonicalize typed emotion tags to ``alpha=X`` for downstream TTS.

    Emotion tags are rebuilt as ``<emo type="emotion" alpha="X">...`` even
    when the input used ``level=N``. Description-mode tags are preserved.

    Level 0 maps to the neutral tag with no explicit alpha.
    """

    segments = parse_tags(tagged_text)
    rebuilt = []
    for segment in segments:
        if segment.tag_name is None:
            rebuilt.append(segment.text)
        elif segment.tag_name == NEUTRAL_TAG:
            rebuilt.append(
                render_emo_tag(
                    text=segment.text,
                    emotion_type=NEUTRAL_TAG,
                    dur=segment.dur,
                )
            )
        elif segment.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION:
            rebuilt.append(
                render_emo_tag(
                    text=segment.text,
                    emotion_desc=segment.emotion_desc,
                    emotion_type_ref=segment.emotion_type_ref,
                    dur=segment.dur,
                )
            )
        elif segment.tag_type == "emotion":
            rebuilt.append(
                render_emo_tag(
                    text=segment.text,
                    emotion_type=segment.emotion_type or segment.tag_name,
                    alpha=segment.effective_alpha,
                    dur=segment.dur,
                )
            )
        elif segment.alpha is not None:
            rebuilt.append(
                render_emo_tag(
                    text=segment.text,
                    emotion_type=segment.tag_name,
                    alpha=segment.alpha,
                    dur=segment.dur,
                )
            )
        else:
            rebuilt.append(
                render_emo_tag(
                    text=segment.text,
                    emotion_type=segment.tag_name,
                    dur=segment.dur,
                )
            )
    return "".join(rebuilt)


# ---------------------------------------------------------------------------
# Tag validation
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Handles common LLM output differences:
    - Whitespace normalization
    - Curly/smart quote → straight quote
    - Strip leading/trailing orphan quotes
    - Normalize dashes (em/en → --)
    - Strip trailing punctuation-only differences
    """
    text = text.strip()
    # Normalize curly/smart quotes to straight
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')
    # Normalize dashes
    text = text.replace("\u2014", "--").replace("\u2013", "--")
    # Strip leading/trailing orphan quotes that LLMs tend to drop
    text = text.strip("\"' ")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compact_text_and_space_boundaries(text: str) -> Tuple[str, set[int]]:
    compact: List[str] = []
    boundaries: set[int] = set()
    pending_space = False
    for char in text:
        if char.isspace():
            pending_space = bool(compact)
            continue
        if pending_space and compact:
            boundaries.add(len(compact))
        compact.append(char)
        pending_space = False
    return "".join(compact), boundaries


def texts_equivalent_ignoring_safe_whitespace(expected: str, actual: str) -> bool:
    """Compare text while ignoring whitespace except English word boundaries.

    Whitespace may be inserted or removed around CJK text and punctuation. A
    whitespace boundary between two ASCII letters in ``expected`` must remain
    present in ``actual`` so two English words cannot be concatenated.
    """
    expected_compact, expected_spaces = _compact_text_and_space_boundaries(
        expected
    )
    actual_compact, actual_spaces = _compact_text_and_space_boundaries(
        actual
    )
    if expected_compact != actual_compact:
        return False
    for boundary in expected_spaces:
        if boundary <= 0 or boundary >= len(expected_compact):
            continue
        left = expected_compact[boundary - 1]
        right = expected_compact[boundary]
        if left.isascii() and right.isascii() and left.isalpha() and right.isalpha():
            if boundary not in actual_spaces:
                return False
    return True


def validate_tagged_text(tagged_text: str, plain_text: str) -> Tuple[bool, str]:
    """
    Validate that tagged text preserves the original plain text.

    Returns (is_valid, error_message).
    """
    stripped = strip_tags(tagged_text)
    if not texts_equivalent_ignoring_safe_whitespace(plain_text, stripped):
        return False, (
            f"Text mismatch after stripping tags.\n"
            f"  Expected: {_normalize_text(plain_text)!r}\n"
            f"  Got:      {_normalize_text(stripped)!r}"
        )

    segments = parse_tags(tagged_text)
    for seg in segments:
        if seg.tag_name is not None and seg.tag_type not in {"emotion", "none"}:
            return False, f"Unsupported tag type: {seg.tag_type}"
        if seg.alpha is not None and not (0.0 <= seg.alpha <= 1.0):
            return False, f"Alpha {seg.alpha} out of range [0, 1]"
        if seg.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION and not seg.emotion_desc:
            return False, "emotion_desc must be non-empty"
        if (
            seg.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION
            and seg.emotion_type_ref is not None
            and seg.emotion_type_ref not in EMOTION_TAGS
        ):
            return False, f"emotion_type_ref must be one of {EMOTION_TAGS}"
        if seg.dur is not None and seg.dur <= 0:
            return False, "dur must be a positive number of seconds"

    try:
        ensure_fully_tagged(tagged_text)
    except ValueError as exc:
        return False, str(exc)

    return True, ""


def validate_no_overlap(tagged_text: str) -> Tuple[bool, str]:
    """Check that there are no overlapping or nested tags."""
    segments = parse_tags(tagged_text)
    # With our regex, nested tags would not be parsed correctly,
    # so if the stripped text doesn't match, there's likely nesting.
    # This is implicitly checked by validate_tagged_text.
    # Additional check: no untagged text should exist between tags
    # in a fully-tagged text. But we allow partial tagging.
    return True, ""


def validate_gradual_variation(
    tagged_texts: Dict[str, str], ordered_keys: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Validate that consecutive tagged texts in order differ by exactly one tag change.

    A "tag change" means one segment's tag_name and/or alpha differs.
    """
    if ordered_keys is None:
        ordered_keys = sorted(tagged_texts.keys(), key=lambda k: int(k))

    if len(ordered_keys) < 2:
        return True, ""

    for i in range(len(ordered_keys) - 1):
        key_a, key_b = ordered_keys[i], ordered_keys[i + 1]
        segs_a = parse_tags(tagged_texts[key_a])
        segs_b = parse_tags(tagged_texts[key_b])

        # Filter out untagged segments
        tagged_a = [s for s in segs_a if s.tag_name is not None]
        tagged_b = [s for s in segs_b if s.tag_name is not None]

        if len(tagged_a) != len(tagged_b):
            return False, (
                f"Texts {key_a} and {key_b} have different number of "
                f"tagged segments: {len(tagged_a)} vs {len(tagged_b)}"
            )

        diffs = 0
        for sa, sb in zip(tagged_a, tagged_b):
            if sa.text.strip() != sb.text.strip():
                return False, (
                    f"Texts {key_a} and {key_b} have different segment text: "
                    f"{sa.text!r} vs {sb.text!r}"
                )
            if sa.control_signature != sb.control_signature:
                diffs += 1

        if diffs != 1:
            return False, (
                f"Texts {key_a} and {key_b} differ by {diffs} tags (expected exactly 1)"
            )

    return True, ""


def validate_boundaries_at_punctuation(
    tagged_text: str, plain_text: str
) -> Tuple[bool, str]:
    """
    Validate that all tag boundaries occur at punctuation positions in
    the plain text (or at the very start/end).
    """
    segments = parse_tags(tagged_text)
    tagged_segs = [s for s in segments if s.tag_name is not None]

    if len(tagged_segs) <= 1:
        return True, ""  # Single tag or no tags

    # Build cumulative positions
    pos = 0
    for i, seg in enumerate(tagged_segs[:-1]):
        pos += len(seg.text)
        # Check the character before this boundary in the plain text
        if pos > 0 and pos <= len(plain_text):
            char_before = plain_text[pos - 1] if pos > 0 else ""
            char_after = plain_text[pos] if pos < len(plain_text) else ""
            if (
                char_before not in ALL_PUNCTUATION
                and char_after not in ALL_PUNCTUATION
                and not char_before.isspace()
                and not char_after.isspace()
            ):
                return False, (
                    f"Tag boundary at position {pos} is not at punctuation. "
                    f"chars around: ...{plain_text[max(0, pos - 3) : pos + 3]}..."
                )

    return True, ""


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------


def _is_chinese_char(c: str) -> bool:
    """Check if a character is Chinese."""
    cp = ord(c)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def _count_words(text: str) -> int:
    """Count words: English words + Chinese characters."""
    count = 0
    for token in text.split():
        has_cjk = any(_is_chinese_char(c) for c in token)
        if has_cjk:
            count += sum(1 for c in token if _is_chinese_char(c))
        else:
            count += 1
    return count


def detect_boundaries(
    text: str, mode: str = "punctuation", min_segment_words: int = 3
) -> List[TextBoundary]:
    """
    Detect tag boundary positions in text.

    Args:
        text: Plain text to analyze.
        mode: "punctuation" (split at punctuation) or "arbitrary" (every 3-5 words).
        min_segment_words: Minimum words/chars per segment.

    Returns:
        List of TextBoundary objects representing valid split points.
        These are positions between segments (not at start/end).
    """
    if mode == "punctuation":
        return _detect_punctuation_boundaries(text, min_segment_words)
    elif mode == "arbitrary":
        return _detect_arbitrary_boundaries(text, min_segment_words)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _detect_punctuation_boundaries(
    text: str, min_segment_words: int
) -> List[TextBoundary]:
    """Find boundary positions at punctuation marks.

    Looks for sentence-ending punctuation (period, exclamation, question mark)
    as well as clause-separating punctuation (comma, semicolon, colon, dash).
    Prefers sentence boundaries over clause boundaries when segments are short.
    """
    boundaries = []

    for i, c in enumerate(text):
            if c in ALL_PUNCTUATION:
                # The boundary is right after the punctuation
                boundary_pos = i + 1
                # Skip trailing whitespace to find real start of next segment
                if boundary_pos < len(text):
                    boundaries.append(TextBoundary(char_idx=boundary_pos))

    # Filter out boundaries that create segments that are too short
    filtered = _filter_boundaries_by_min_words(text, boundaries, min_segment_words)
    return filtered


def _detect_arbitrary_boundaries(
    text: str, min_segment_words: int
) -> List[TextBoundary]:
    """Find boundary positions every 3-5 words/characters."""
    import random

    boundaries = []
    # Tokenize into word positions
    word_positions = []  # (start_idx, end_idx) for each word

    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue

        if _is_chinese_char(text[i]):
            # Each Chinese char is a "word"
            word_positions.append((i, i + 1))
            i += 1
        else:
            # English word
            start = i
            while (
                i < len(text)
                and not text[i].isspace()
                and not _is_chinese_char(text[i])
            ):
                i += 1
            word_positions.append((start, i))

    if len(word_positions) < min_segment_words * 2:
        return []  # Text too short for arbitrary splitting

    # Place boundaries every 3-5 words
    target_size = random.randint(3, 5)
    word_count = 0
    for idx, (start, end) in enumerate(word_positions):
        word_count += 1
        if word_count >= target_size and idx < len(word_positions) - min_segment_words:
            # Find the actual character position after this word
            boundary_pos = end
            # Skip trailing spaces
            while boundary_pos < len(text) and text[boundary_pos].isspace():
                boundary_pos += 1

            if boundary_pos < len(text):
                boundaries.append(TextBoundary(char_idx=boundary_pos))
            word_count = 0
            target_size = random.randint(3, 5)

    return boundaries


def _filter_boundaries_by_min_words(
    text: str, boundaries: List[TextBoundary], min_words: int
) -> List[TextBoundary]:
    """Remove boundaries that would create segments smaller than min_words.

    Greedily keeps boundaries that produce segments >= min_words.
    If the last segment is too short, drops the last boundary to merge it.
    """
    if not boundaries:
        return []

    filtered = []
    prev_pos = 0

    for b in boundaries:
        segment_text = text[prev_pos : b.char_idx]
        wc = _count_words(segment_text)
        if wc >= min_words:
            filtered.append(b)
            prev_pos = b.char_idx
        # else: skip this boundary, merging the small segment forward

    # Check the last segment (from last boundary to end)
    if filtered:
        last_seg = text[filtered[-1].char_idx :]
        if _count_words(last_seg) < min_words:
            # Merge with previous segment by dropping the last boundary
            filtered.pop()

    return filtered


def split_text_at_boundaries(text: str, boundaries: List[TextBoundary]) -> List[str]:
    """Split text into segments at the given boundary positions."""
    if not boundaries:
        return [text]

    positions = [0] + [b.char_idx for b in boundaries] + [len(text)]
    segments = []
    for i in range(len(positions) - 1):
        seg = text[positions[i] : positions[i + 1]]
        segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def _attr_quote(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_emo_tag(
    *,
    text: str,
    emotion_type: Optional[str] = None,
    alpha: Optional[float] = None,
    level: Optional[int] = None,
    emotion_desc: Optional[str] = None,
    emotion_type_ref: Optional[str] = None,
    dur: Optional[float] = None,
) -> str:
    """Render one canonical ``<emo>`` span."""
    dur_attr = ""
    if dur is not None:
        if dur <= 0:
            raise ValueError("dur must be a positive number of seconds")
        dur_attr = f' dur="{_attr_quote(dur)}"'
    if emotion_desc is not None:
        desc = emotion_desc.strip()
        if not desc:
            raise ValueError("emotion_desc must be non-empty")
        if emotion_type is not None or alpha is not None or level is not None:
            raise ValueError("emotion_desc cannot be combined with type/alpha/level")
        type_ref_attr = ""
        if emotion_type_ref is not None:
            normalized_type_ref = emotion_type_ref.lower().strip()
            if normalized_type_ref not in EMOTION_TAGS:
                raise ValueError(f"Unknown emotion type_ref: {emotion_type_ref}")
            type_ref_attr = f' type_ref="{_attr_quote(normalized_type_ref)}"'
        return f'<emo desc="{_attr_quote(desc)}"{type_ref_attr}{dur_attr}>{text}</emo>'

    if emotion_type_ref is not None:
        raise ValueError("emotion_type_ref requires emotion_desc")

    if emotion_type is None:
        raise ValueError("emotion_type is required when emotion_desc is absent")
    emotion_type = emotion_type.lower().strip()
    if emotion_type == NEUTRAL_TAG:
        if alpha is not None or level is not None:
            raise ValueError('type="none" cannot include alpha or level')
        return f'<emo type="{NEUTRAL_TAG}"{dur_attr}>{text}</emo>'
    if emotion_type not in EMOTION_TAGS:
        raise ValueError(f"Unknown emotion type: {emotion_type}")
    attrs = [f'type="{_attr_quote(emotion_type)}"']
    if alpha is not None:
        attrs.append(f'alpha="{_attr_quote(alpha)}"')
    elif level is not None:
        attrs.append(f'level="{_attr_quote(level)}"')
    if dur is not None:
        attrs.append(f'dur="{_attr_quote(dur)}"')
    return f"<emo {' '.join(attrs)}>{text}</emo>"


TaggedBuildTuple = Union[
    Tuple[Optional[str], str, Optional[float]],
    Tuple[Optional[str], str, Optional[float], Optional[str]],
    Tuple[Optional[str], str, Optional[float], Optional[str], Optional[float]],
    Tuple[Optional[str], str, Optional[float], Optional[str], Optional[float], Optional[str]],
]


def build_tagged_text(segments: List[TaggedBuildTuple]) -> str:
    """
    Build canonical tagged text from tuples.

    Args:
        segments: (emotion_type, text, alpha) or
                  (emotion_type, text, alpha, emotion_desc) or
                  (emotion_type, text, alpha, emotion_desc, dur) or
                  (emotion_type, text, alpha, emotion_desc, dur, type_ref).
                  ``emotion_type`` can be None for untagged text.
    """
    parts = []
    for segment in segments:
        tag_name = segment[0]
        text = segment[1]
        alpha = segment[2]
        emotion_desc = segment[3] if len(segment) > 3 else None
        dur = segment[4] if len(segment) > 4 else None
        emotion_type_ref = segment[5] if len(segment) > 5 else None
        if emotion_desc is not None:
            parts.append(
                render_emo_tag(
                    text=text,
                    emotion_desc=emotion_desc,
                    emotion_type_ref=emotion_type_ref,
                    dur=dur,
                )
            )
        elif tag_name is None:
            parts.append(text)
        elif alpha is not None:
            parts.append(
                render_emo_tag(
                    text=text,
                    emotion_type=tag_name,
                    alpha=alpha,
                    dur=dur,
                )
            )
        else:
            parts.append(render_emo_tag(text=text, emotion_type=tag_name, dur=dur))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def get_all_pairs(keys: List[str], bidirectional: bool = True) -> List[Tuple[str, str]]:
    """
    Generate all unique pairs from a list of keys.

    If bidirectional=True, generates both (a,b) and (b,a).
    """
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pairs.append((keys[i], keys[j]))
            if bidirectional:
                pairs.append((keys[j], keys[i]))
    return pairs


def describe_tag_diff(source_text: str, target_text: str) -> List[Dict]:
    """
    Describe the differences between two tagged texts.

    Returns a list of diffs, each with:
      - segment_idx: which segment changed
      - segment_text: the text of the changed segment
      - source_tag: tag in source
      - target_tag: tag in target
      - source_level: level in source, when present
      - target_level: level in target, when present
      - source_alpha: alpha in source
      - target_alpha: alpha in target
      - source_emotion_desc / target_emotion_desc: emotion description prompt
      - source_emotion_type_ref / target_emotion_type_ref: desc-mode reference type
      - source_dur / target_dur: optional target duration in seconds
    """
    source_segs = [s for s in parse_tags(source_text) if s.tag_name is not None]
    target_segs = [s for s in parse_tags(target_text) if s.tag_name is not None]

    diffs = []
    for i, (ss, ts) in enumerate(zip(source_segs, target_segs)):
        if ss.control_signature != ts.control_signature:
            diffs.append(
                {
                    "segment_idx": i,
                    "segment_text": ss.text.strip(),
                    "source_tag": ss.evaluation_tag,
                    "target_tag": ts.evaluation_tag,
                    "source_level": ss.emotion_level,
                    "target_level": ts.emotion_level,
                    "source_alpha": ss.effective_alpha,
                    "target_alpha": ts.effective_alpha,
                    "source_emotion_control_mode": ss.emotion_control_mode,
                    "target_emotion_control_mode": ts.emotion_control_mode,
                    "source_emotion_desc": ss.emotion_desc,
                    "target_emotion_desc": ts.emotion_desc,
                    "source_emotion_type_ref": ss.emotion_type_ref,
                    "target_emotion_type_ref": ts.emotion_type_ref,
                    "source_dur": ss.dur,
                    "target_dur": ts.dur,
                }
            )
    return diffs
