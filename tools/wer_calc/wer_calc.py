"""
WER/CER calculation utilities.

Provides both word-level (WER) and character-level (CER) error rate computation
with proper text normalization. Uses jiwer for underlying computation and
supports Chinese text via CER.

Usage (function interface):
    from wer_calc import compute_wer, compute_cer
    result = compute_wer("hello world", "hello word")
    print(result.wer, result.substitutions, result.deletions, result.insertions)
"""

import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import List, Optional

import jiwer


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class WERResult:
    """Result of a WER/CER computation."""

    error_rate: float  # WER or CER (0.0 – 1.0+)
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_length: int  # Number of reference tokens (words or chars)
    hyp_length: int  # Number of hypothesis tokens
    reference_normalized: str  # Normalized reference text
    hypothesis_normalized: str  # Normalized hypothesis text
    metric: str  # "wer" or "cer"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AggregateWERResult:
    """Sequence-level aggregate over multiple WER/CER result dictionaries."""

    error_rate: Optional[float]
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_length: int
    hyp_length: int
    metric: str
    count: int
    valid_count: int
    skipped_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class ErrorRateAccumulator:
    """Incrementally aggregate WER/CER counts using total reference length."""

    def __init__(self) -> None:
        self.count = 0
        self.valid_count = 0
        self.skipped_count = 0
        self.substitutions = 0
        self.deletions = 0
        self.insertions = 0
        self.hits = 0
        self.ref_length = 0
        self.hyp_length = 0
        self.metrics: set[str] = set()

    def add(self, result: object) -> None:
        self.count += 1
        required = (
            "substitutions",
            "deletions",
            "insertions",
            "hits",
            "ref_length",
            "hyp_length",
        )
        try:
            data = _result_to_dict(result)
            values = {key: int(data[key]) for key in required}
        except (KeyError, TypeError, ValueError):
            self.skipped_count += 1
            return

        self.valid_count += 1
        self.substitutions += values["substitutions"]
        self.deletions += values["deletions"]
        self.insertions += values["insertions"]
        self.hits += values["hits"]
        self.ref_length += values["ref_length"]
        self.hyp_length += values["hyp_length"]
        metric = data.get("metric")
        if isinstance(metric, str) and metric:
            self.metrics.add(metric)

    def to_result(self) -> AggregateWERResult:
        errors = self.substitutions + self.deletions + self.insertions
        if self.valid_count == 0:
            error_rate = None
        elif self.ref_length > 0:
            error_rate: Optional[float] = errors / self.ref_length
        else:
            error_rate = 0.0 if errors == 0 else None
        if not self.metrics:
            metric = "unknown"
        elif len(self.metrics) == 1:
            metric = next(iter(self.metrics))
        else:
            metric = "mixed"
        return AggregateWERResult(
            error_rate=error_rate,
            substitutions=self.substitutions,
            deletions=self.deletions,
            insertions=self.insertions,
            hits=self.hits,
            ref_length=self.ref_length,
            hyp_length=self.hyp_length,
            metric=metric,
            count=self.count,
            valid_count=self.valid_count,
            skipped_count=self.skipped_count,
        )


def _result_to_dict(result: object) -> dict:
    if isinstance(result, WERResult):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    raise TypeError(f"Unsupported WER result type: {type(result).__name__}")


def aggregate_error_rate_results(results: List[object]) -> AggregateWERResult:
    """Aggregate WER/CER results as if references were concatenated."""
    accumulator = ErrorRateAccumulator()
    for result in results:
        accumulator.add(result)
    return accumulator.to_result()


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #

_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0x2A700, 0x2B73F),  # CJK Extension C
    (0x2B740, 0x2B81F),  # CJK Extension D
    (0x2B820, 0x2CEAF),  # CJK Extension E
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Supplement
)


def _is_cjk_char(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _has_cjk(text: str) -> bool:
    return any(_is_cjk_char(ord(c)) for c in text)


def _insert_script_boundary_spaces(text: str) -> str:
    """Insert spaces at CJK↔Latin and CJK↔digit script-switching boundaries.

    Ensures consistent tokenization regardless of input spacing:
        "一个apple"  → "一个 apple"
        "第3个"      → "第 3 个"
        "一个 apple" → "一个 apple"  (already spaced — no change)

    This prevents mismatches like "一个apple" vs "一个 apple" being treated
    as different tokens in WER mode.
    """
    if not text:
        return text
    result = [text[0]]
    for i in range(1, len(text)):
        prev_cp = ord(text[i - 1])
        curr_cp = ord(text[i])
        prev_cjk = _is_cjk_char(prev_cp)
        curr_cjk = _is_cjk_char(curr_cp)
        prev_alnum = text[i - 1].isascii() and text[i - 1].isalnum()
        curr_alnum = text[i].isascii() and text[i].isalnum()
        # Insert space at CJK↔ASCII-alphanumeric boundary
        if (prev_cjk and curr_alnum) or (prev_alnum and curr_cjk):
            result.append(" ")
        result.append(text[i])
    return "".join(result)


def normalize_text(text: str) -> str:
    """Normalize text for error rate computation.

    - Unicode NFKC normalization
    - Lowercase
    - Insert spaces at CJK↔Latin and CJK↔digit script boundaries
    - Remove punctuation (keep alphanumeric, CJK, spaces)
    - Collapse whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Insert spaces at script-switching boundaries (CJK↔Latin, CJK↔digit)
    # before punctuation removal so boundaries are detected accurately.
    text = _insert_script_boundary_spaces(text)
    # Keep: word characters (\w includes CJK in Python 3), spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove digits? No — keep them, they can be meaningful in ASR.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _mixed_language_tokens(text: str) -> List[str]:
    """Tokenize CJK as characters and non-CJK runs by whitespace-delimited words."""
    normalized = normalize_text(text)
    tokens: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current.clear()

    for char in normalized:
        if char.isspace():
            flush()
            continue
        if _is_cjk_char(ord(char)):
            flush()
            tokens.append(char)
            continue
        current.append(char)
    flush()
    return tokens


def _compute_token_error_rate(
    reference_tokens: List[str],
    hypothesis_tokens: List[str],
    *,
    reference_normalized: str,
    hypothesis_normalized: str,
    metric: str,
) -> WERResult:
    if not reference_tokens:
        error_rate = 0.0 if not hypothesis_tokens else float("inf")
        return WERResult(
            error_rate=error_rate,
            substitutions=0,
            deletions=0,
            insertions=len(hypothesis_tokens),
            hits=0,
            ref_length=0,
            hyp_length=len(hypothesis_tokens),
            reference_normalized=reference_normalized,
            hypothesis_normalized=hypothesis_normalized,
            metric=metric,
        )

    output = jiwer.process_words(
        " ".join(reference_tokens),
        " ".join(hypothesis_tokens),
    )
    return WERResult(
        error_rate=output.wer,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
        ref_length=len(reference_tokens),
        hyp_length=len(hypothesis_tokens),
        reference_normalized=reference_normalized,
        hypothesis_normalized=hypothesis_normalized,
        metric=metric,
    )


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #


def compute_wer(
    reference: str,
    hypothesis: str,
) -> WERResult:
    """Compute Word Error Rate.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted / transcribed text.

    Returns:
        WERResult with error rate and detailed statistics.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        error_rate = 0.0 if not hyp_norm else float("inf")
        hyp_words = hyp_norm.split() if hyp_norm else []
        return WERResult(
            error_rate=error_rate,
            substitutions=0,
            deletions=0,
            insertions=len(hyp_words),
            hits=0,
            ref_length=0,
            hyp_length=len(hyp_words),
            reference_normalized=ref_norm,
            hypothesis_normalized=hyp_norm,
            metric="wer",
        )

    output = jiwer.process_words(ref_norm, hyp_norm)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    return WERResult(
        error_rate=output.wer,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
        ref_length=len(ref_words),
        hyp_length=len(hyp_words),
        reference_normalized=ref_norm,
        hypothesis_normalized=hyp_norm,
        metric="wer",
    )


def compute_cer(
    reference: str,
    hypothesis: str,
) -> WERResult:
    """Compute Character Error Rate.

    Suitable for Chinese and other CJK languages where word
    segmentation is ambiguous. Each character becomes a token.

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted / transcribed text.

    Returns:
        WERResult with error rate and detailed statistics (metric="cer").
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    # Remove all spaces so each char is a token
    ref_chars = ref_norm.replace(" ", "")
    hyp_chars = hyp_norm.replace(" ", "")

    if not ref_chars:
        error_rate = 0.0 if not hyp_chars else float("inf")
        return WERResult(
            error_rate=error_rate,
            substitutions=0,
            deletions=0,
            insertions=len(hyp_chars),
            hits=0,
            ref_length=0,
            hyp_length=len(hyp_chars),
            reference_normalized=ref_norm,
            hypothesis_normalized=hyp_norm,
            metric="cer",
        )

    # Space-separate chars for jiwer
    ref_spaced = " ".join(ref_chars)
    hyp_spaced = " ".join(hyp_chars)

    output = jiwer.process_words(ref_spaced, hyp_spaced)

    return WERResult(
        error_rate=output.wer,  # jiwer.process_words gives "WER" which is CER here
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
        ref_length=len(ref_chars),
        hyp_length=len(hyp_chars),
        reference_normalized=ref_norm,
        hypothesis_normalized=hyp_norm,
        metric="cer",
    )


def compute_mixed_error_rate(
    reference: str,
    hypothesis: str,
) -> WERResult:
    """Compute error rate with CJK chars and non-CJK whitespace words as units."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    return _compute_token_error_rate(
        _mixed_language_tokens(reference),
        _mixed_language_tokens(hypothesis),
        reference_normalized=ref_norm,
        hypothesis_normalized=hyp_norm,
        metric="mixed",
    )


def compute_error_rate(
    reference: str,
    hypothesis: str,
    language: str = "auto",
) -> WERResult:
    """Compute error rate with automatic metric selection.

    - auto/mixed → CJK chars + non-CJK whitespace words
    - Chinese language hints → CER
    - Other explicit language hints → WER

    Args:
        reference: Ground-truth text.
        hypothesis: Predicted / transcribed text.
        language: "auto"/"mixed", "zh"/"chinese" (→ CER), or anything else (→ WER).

    Returns:
        WERResult.
    """
    normalized_language = language.lower()
    if normalized_language in ("auto", "mixed"):
        return compute_mixed_error_rate(reference, hypothesis)

    if normalized_language in ("zh", "chinese", "zho", "cmn"):
        return compute_cer(reference, hypothesis)
    return compute_wer(reference, hypothesis)


@dataclass
class LocalWERResult:
    """Result of a local WER/CER computation around edit regions."""

    edit_region_wer: WERResult  # WER in the edit region (±context)
    non_edit_region_wer: WERResult  # WER outside the edit region
    edit_word_indices: List[int]  # Word indices in target text that are edited
    local_window_indices: List[int]  # Word indices in the local window (edit ± context)
    context_words: int  # Context window size K
    local_ref_text: str  # Reference text in the local window
    local_hyp_text: str  # Hypothesis text in the local window (approximate)

    def to_dict(self) -> dict:
        return {
            "edit_region_wer": self.edit_region_wer.to_dict(),
            "non_edit_region_wer": self.non_edit_region_wer.to_dict(),
            "edit_word_indices": self.edit_word_indices,
            "local_window_indices": self.local_window_indices,
            "context_words": self.context_words,
            "local_ref_text": self.local_ref_text,
            "local_hyp_text": self.local_hyp_text,
        }


def _find_edit_regions(
    original_text: str,
    target_text: str,
) -> List[int]:
    """Find word indices in target text that differ from original text.

    Uses difflib.SequenceMatcher to diff normalized word lists.
    Returns sorted list of word indices in the target that are new/changed.
    """
    import difflib

    orig_words = normalize_text(original_text).split()
    targ_words = normalize_text(target_text).split()

    sm = difflib.SequenceMatcher(None, orig_words, targ_words)
    changed_indices = set()

    for op, _i1, _i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            for j in range(j1, j2):
                changed_indices.add(j)

    return sorted(changed_indices)


def _expand_with_context(
    edit_indices: List[int],
    total_words: int,
    context_k: int = 3,
) -> List[int]:
    """Expand edit word indices with ±K context words."""
    window = set()
    for idx in edit_indices:
        for k in range(max(0, idx - context_k), min(total_words, idx + context_k + 1)):
            window.add(k)
    return sorted(window)


def _compute_region_wer_from_alignment(
    output,  # jiwer.WordOutput
    ref_words: List[str],
    hyp_words: List[str],
    region_ref_indices: set,
) -> WERResult:
    """Count WER errors only within a set of reference word indices.

    Uses jiwer alignment chunks to attribute errors to specific ref positions.
    """
    local_s = local_d = local_i = local_h = 0
    local_ref_positions = []
    local_hyp_positions = []

    for chunk in output.alignments[0]:
        chunk_type = chunk.type
        ref_start = chunk.ref_start_idx
        ref_end = chunk.ref_end_idx
        hyp_start = chunk.hyp_start_idx
        hyp_end = chunk.hyp_end_idx

        # Check overlap with region
        ref_range = set(range(ref_start, ref_end))
        overlap = ref_range & region_ref_indices

        if not overlap and chunk_type != "insert":
            continue

        if chunk_type == "equal":
            n = len(overlap)
            local_h += n
            local_ref_positions.extend(sorted(overlap))
            # Map corresponding hyp positions
            for ri in sorted(overlap):
                offset = ri - ref_start
                if hyp_start + offset < hyp_end:
                    local_hyp_positions.append(hyp_start + offset)
        elif chunk_type == "substitute":
            n = len(overlap)
            local_s += n
            local_ref_positions.extend(sorted(overlap))
            for ri in sorted(overlap):
                offset = ri - ref_start
                if hyp_start + offset < hyp_end:
                    local_hyp_positions.append(hyp_start + offset)
        elif chunk_type == "delete":
            n = len(overlap)
            local_d += n
            local_ref_positions.extend(sorted(overlap))
        elif chunk_type == "insert":
            # Insertions are attributed to the ref position they appear at.
            # An insertion at ref_start means extra hyp words appear before ref_start.
            # We attribute it to the local region if ref_start is in the region
            # or if ref_start-1 is in the region (insertion between two local words).
            if ref_start in region_ref_indices or (ref_start - 1) in region_ref_indices:
                local_i += hyp_end - hyp_start
                for hi in range(hyp_start, hyp_end):
                    local_hyp_positions.append(hi)

    local_ref_len = local_h + local_s + local_d
    error_rate = (
        (local_s + local_d + local_i) / local_ref_len if local_ref_len > 0 else 0.0
    )

    local_ref_text = " ".join(
        ref_words[i] for i in sorted(set(local_ref_positions)) if i < len(ref_words)
    )
    local_hyp_text = " ".join(
        hyp_words[i] for i in sorted(set(local_hyp_positions)) if i < len(hyp_words)
    )

    return WERResult(
        error_rate=error_rate,
        substitutions=local_s,
        deletions=local_d,
        insertions=local_i,
        hits=local_h,
        ref_length=local_ref_len,
        hyp_length=len(set(local_hyp_positions)),
        reference_normalized=local_ref_text,
        hypothesis_normalized=local_hyp_text,
        metric="wer",
    )


def compute_local_wer(
    original_text: str,
    target_text: str,
    hypothesis: str,
    context_k: int = 3,
    language: str = "auto",
) -> LocalWERResult:
    """Compute local WER around edited regions and non-edit region WER.

    1. Diff original vs target text to find edited word indices in target.
    2. Expand with ±K context words.
    3. Use jiwer alignment to count errors in local vs non-local regions.

    Args:
        original_text: The original text before editing.
        target_text: The expected target text after editing (reference).
        hypothesis: The ASR transcription of edited audio.
        context_k: Number of context words on each side of edit region.
        language: Language for metric selection ("auto", "zh", etc.).

    Returns:
        LocalWERResult with edit-region and non-edit-region WER.
    """
    if language == "auto":
        import difflib

        orig_tokens = _mixed_language_tokens(original_text)
        ref_tokens = _mixed_language_tokens(target_text)
        hyp_tokens = _mixed_language_tokens(hypothesis)

        sm = difflib.SequenceMatcher(None, orig_tokens, ref_tokens)
        edit_indices = []
        for op, _i1, _i2, j1, j2 in sm.get_opcodes():
            if op != "equal":
                edit_indices.extend(range(j1, j2))

        total_tokens = len(ref_tokens)
        window_indices = _expand_with_context(edit_indices, total_tokens, context_k)
        output = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))

        region_set = set(window_indices)
        non_region_set = set(range(total_tokens)) - region_set

        edit_wer = _compute_region_wer_from_alignment(
            output, ref_tokens, hyp_tokens, region_set
        )
        edit_wer = WERResult(**{**asdict(edit_wer), "metric": "mixed"})
        non_edit_wer = _compute_region_wer_from_alignment(
            output, ref_tokens, hyp_tokens, non_region_set
        )
        non_edit_wer = WERResult(**{**asdict(non_edit_wer), "metric": "mixed"})

        local_ref_text = " ".join(
            ref_tokens[i] for i in window_indices if i < len(ref_tokens)
        )
        local_hyp_text = edit_wer.hypothesis_normalized

        return LocalWERResult(
            edit_region_wer=edit_wer,
            non_edit_region_wer=non_edit_wer,
            edit_word_indices=edit_indices,
            local_window_indices=window_indices,
            context_words=context_k,
            local_ref_text=local_ref_text,
            local_hyp_text=local_hyp_text,
        )

    # For explicit CJK hints, use character-level analysis.
    use_cer = language.lower() in ("zh", "chinese", "zho", "cmn")

    if use_cer:
        # For CER, operate on characters instead of words
        orig_norm = normalize_text(original_text).replace(" ", "")
        targ_norm = normalize_text(target_text).replace(" ", "")
        hyp_norm = normalize_text(hypothesis).replace(" ", "")

        # Find edit regions at character level
        import difflib

        sm = difflib.SequenceMatcher(None, list(orig_norm), list(targ_norm))
        edit_indices = []
        for op, _i1, _i2, j1, j2 in sm.get_opcodes():
            if op != "equal":
                edit_indices.extend(range(j1, j2))

        total_chars = len(targ_norm)
        window_indices = _expand_with_context(edit_indices, total_chars, context_k)

        # Space-separate chars for jiwer
        ref_tokens = list(targ_norm)
        hyp_tokens = list(hyp_norm)
        output = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))

        region_set = set(window_indices)
        non_region_set = set(range(total_chars)) - region_set

        edit_wer = _compute_region_wer_from_alignment(
            output, ref_tokens, hyp_tokens, region_set
        )
        edit_wer = WERResult(**{**asdict(edit_wer), "metric": "cer"})
        non_edit_wer = _compute_region_wer_from_alignment(
            output, ref_tokens, hyp_tokens, non_region_set
        )
        non_edit_wer = WERResult(**{**asdict(non_edit_wer), "metric": "cer"})

        local_ref_text = "".join(
            ref_tokens[i] for i in window_indices if i < len(ref_tokens)
        )
        local_hyp_text = edit_wer.hypothesis_normalized

        return LocalWERResult(
            edit_region_wer=edit_wer,
            non_edit_region_wer=non_edit_wer,
            edit_word_indices=edit_indices,
            local_window_indices=window_indices,
            context_words=context_k,
            local_ref_text=local_ref_text,
            local_hyp_text=local_hyp_text,
        )

    # WER path (word-level)
    ref_norm = normalize_text(target_text)
    hyp_norm = normalize_text(hypothesis)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    edit_indices = _find_edit_regions(original_text, target_text)
    total_words = len(ref_words)
    window_indices = _expand_with_context(edit_indices, total_words, context_k)

    if not ref_words:
        empty = WERResult(0.0, 0, 0, 0, 0, 0, 0, "", "", "wer")
        return LocalWERResult(empty, empty, [], [], context_k, "", "")

    output = jiwer.process_words(ref_norm, hyp_norm)

    region_set = set(window_indices)
    non_region_set = set(range(total_words)) - region_set

    edit_wer = _compute_region_wer_from_alignment(
        output, ref_words, hyp_words, region_set
    )
    non_edit_wer = _compute_region_wer_from_alignment(
        output, ref_words, hyp_words, non_region_set
    )

    local_ref_text = " ".join(
        ref_words[i] for i in window_indices if i < len(ref_words)
    )

    return LocalWERResult(
        edit_region_wer=edit_wer,
        non_edit_region_wer=non_edit_wer,
        edit_word_indices=edit_indices,
        local_window_indices=window_indices,
        context_words=context_k,
        local_ref_text=local_ref_text,
        local_hyp_text=edit_wer.hypothesis_normalized,
    )


def compute_error_rate_batch(
    references: List[str],
    hypotheses: List[str],
    language: str = "auto",
) -> List[WERResult]:
    """Compute error rates for a batch of reference/hypothesis pairs."""
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses"
        )
    return [
        compute_error_rate(ref, hyp, language)
        for ref, hyp in zip(references, hypotheses)
    ]
