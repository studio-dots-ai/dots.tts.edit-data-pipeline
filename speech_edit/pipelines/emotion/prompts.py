"""
LLM prompt templates for emotion/style editing pipeline.

Generate profile-controlled tagged texts and edit instructions.
"""

import json
from typing import Optional

# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------

AVAILABLE_EMOTION_TAGS = (
    "happy, angry, sad, afraid, disgusted, melancholic, surprised, calm"
)

# Import active style tags from tag_utils (default: just whisper)
from .tag_utils import STYLE_TAGS as _ACTIVE_STYLE_TAGS

AVAILABLE_STYLE_TAGS = ", ".join(_ACTIVE_STYLE_TAGS)


def _plan_value(emotion_edit_plan: Optional[dict], key: str, default):
    if emotion_edit_plan is None:
        return default
    return emotion_edit_plan.get(key, default)


def _format_allowed_emotions(emotion_edit_plan: Optional[dict]) -> str:
    allowed = _plan_value(
        emotion_edit_plan,
        "allowed_emotions",
        [tag.strip() for tag in AVAILABLE_EMOTION_TAGS.split(",")],
    )
    return ", ".join(allowed)


def _format_emotion_control_desc(emotion_edit_plan: Optional[dict]) -> str:
    allowed = _format_allowed_emotions(emotion_edit_plan)
    edit_mode = _plan_value(emotion_edit_plan, "edit_mode", "type_change")
    emotion_prompt_mode = _plan_value(emotion_edit_plan, "emotion_prompt_mode", "level")
    if emotion_prompt_mode == "description":
        if edit_mode == "intensity_change":
            mode_rules = """\
- `edit_mode` is `intensity_change`.
- Changed segments must keep the same emotional direction but change the strength, energy, or expressiveness inside `desc`.
- Use at least two noticeably different intensities among the 3 variants so valid intensity edit pairs exist."""
        else:
            mode_rules = """\
- `edit_mode` is `type_change`.
- Changed segments must change the emotional direction described by `desc`.
- The emotion descriptions can be natural, specific, and bilingual, but should focus on vocal delivery rather than content."""
        return f"""\
## External emotion control plan
- `emotion_prompt_mode`: description
{mode_rules}
- All emotion control must use `<emo desc="..." type_ref="TAG_NAME">...</emo>`.
- `type_ref` is the reference emotion label for evaluation and MUST be one of: {allowed}.
- Do NOT use `type` or `level` in description mode.
- Do NOT use style tags."""

    if emotion_prompt_mode != "level":
        raise ValueError(f"Unknown emotion_prompt_mode: {emotion_prompt_mode}")

    if edit_mode == "intensity_change":
        mode_rules = """\
- `edit_mode` is `intensity_change`.
- Changed segments must keep the same emotion tag name and change only `level`.
- Use at least two different levels among the 3 variants so valid intensity edit pairs exist."""
    else:
        mode_rules = """\
- `edit_mode` is `type_change`.
- Changed segments must change the emotion tag name.
- Prefer keeping the same level when changing emotion type, so the main edit is the emotion category."""
    return f"""\
## External emotion control plan
- `allowed_emotions`: {allowed}
{mode_rules}
- `emotion_prompt_mode`: level
- All tagged segments must use emotion types ONLY from `allowed_emotions`.
- Do NOT use any emotion tag outside `allowed_emotions`.
- Every non-neutral emotion tag MUST include an explicit `level`.
- Do NOT use style tags."""


def build_strict_emotion_matrix_prompt(
    plain_text: str,
    sub_type: str,
    emotion_edit_plan: dict,
    *,
    natural_language_instruction_mode: str,
) -> str:
    """Render a compact prompt where code owns all emotion choices."""
    region_count = int(emotion_edit_plan["region_count"])
    matrix = emotion_edit_plan["emotion_matrix"]
    prompt_mode = emotion_edit_plan["emotion_prompt_mode"]
    levels = emotion_edit_plan["region_levels"]
    if sub_type == "single":
        split_rule = "Use one region containing the entire text."
    elif sub_type == "multi_emo_normal":
        split_rule = (
            f"Split into exactly {region_count} complete clauses. Prefer punctuation "
            "or a natural pause as each boundary."
        )
    elif sub_type == "multi_emo_frequent":
        split_rule = (
            f"Split into exactly {region_count} coherent phrases. Put boundaries "
            "inside a sentence where there is no punctuation or natural pause."
        )
    else:
        raise ValueError(f"Strict emotion matrix does not support {sub_type!r}")

    rows = "\n".join(
        f"- Variant {variant}: "
        + ", ".join(
            f"region {index + 1}={emotion}"
            for index, emotion in enumerate(matrix[variant])
        )
        for variant in ("1", "2", "3")
    )
    if prompt_mode == "level":
        control_rule = (
            "Use `<emo type=\"EMOTION\" level=\"N\">text</emo>`. "
            "The fixed levels for regions 1..K are "
            f"{levels}; use the same region level in all variants."
        )
    elif prompt_mode == "description":
        control_rule = (
            "Use `<emo desc=\"concise vocal delivery description\" "
            "type_ref=\"EMOTION\">text</emo>`. Each description must match "
            "type_ref and describe audible delivery such as pitch, energy, "
            "pace, breathiness, trembling, or a smile in the voice. Use a "
            "short natural phrase, roughly 3-8 English words or 4-16 Chinese "
            "characters. English and Chinese descriptions are both allowed. "
            "Do not copy the emotion label as the description, and do not use "
            "bare wrappers such as `happy tone` or `sad voice`. Good examples "
            "include `bright and cheerful delivery`, `tense, trembling breath`, "
            "`低沉而带有哀伤感`, and `轻快明亮的带笑语气`. Do not describe the "
            "text content. If a region keeps the same emotion in two adjacent "
            "variants, copy its description exactly."
        )
    else:
        raise ValueError(f"Unknown emotion_prompt_mode: {prompt_mode!r}")

    include_nl = natural_language_instruction_mode.strip().lower() == "llm"
    nl_rule = ""
    nl_json = ""
    if include_nl:
        nl_rule = (
            "\nAlso return `nl_inst` for all six ordered pairs. Each value is one "
            "short instruction naming only the target emotion change."
        )
        nl_json = (
            ',\n  "nl_inst": {"1->2": "...", "1->3": "...", "2->1": "...", '
            '"2->3": "...", "3->1": "...", "3->2": "..."}'
        )

    return f"""\
Return one JSON object only.

Text: {json.dumps(plain_text, ensure_ascii=False)}

1. Set `canonical_text` to the input text. Do not rewrite, normalize, add, or remove characters.
2. {split_rule}
3. Use the exact same region boundaries in all three variants.
4. Tag every character exactly once. Put boundary whitespace inside the left region, so adjacent tags use `</emo><emo>` with no gap. Do not nest tags.
5. Follow this emotion matrix exactly:
{rows}
6. {control_rule}
Whitespace-only differences are allowed, but never join two English words.{nl_rule}

Output shape:
{{
  "canonical_text": {json.dumps(plain_text, ensure_ascii=False)},
  "tagged_texts": {{
    "1": "fully tagged text",
    "2": "fully tagged text",
    "3": "fully tagged text"
  }}{nl_json}
}}"""

TAG_FORMAT_DESC = """\
Tag format: use canonical `<emo>` tags only.
- Level mode: `<emo type="TAG_NAME" level="N">text</emo>`.
- Description mode: `<emo desc="natural-language emotion description" type_ref="TAG_NAME">text</emo>`.
- Neutral internal spans in level mode: `<emo type="none">text</emo>`.
- Neutral or unchanged spans in description mode must still use `<emo desc="neutral, steady delivery" type_ref="TAG_NAME">text</emo>` with a `type_ref` from the allowed emotion list.
- TAG_NAME: one of the available emotion types listed above.
- level: integer 1-3. 1=mild, 2=moderate, 3=strong. Required for level mode emotion tags.
- desc: a concise natural-language description of vocal emotion and intensity.
- type_ref: the reference emotion category matching desc, used only for evaluation.
- `desc` is the emotion prompt for synthesis. It is not the same thing as the edit instruction.
- The text content inside tags must be EXACTLY the same as the base text being tagged (same words, same punctuation, same order). Do NOT add, remove, or change any characters.
- Tags must cover the ENTIRE text with no gaps. Every character must be inside exactly one tag.
- Tags must NOT overlap or nest."""


TEXT_NORMALIZATION_DESC = """\
- First preprocess the input plain text into a frozen `canonical_text` before generating any tags or instructions.
- `canonical_text` must be a complete spoken-form text:
  - Numbers, years, dates, currencies, percentages, and similar forms must be written out in full spoken form.
    - Example: `2020` -> `twenty twenty`
  - Abbreviations and titles must be expanded to spoken form.
    - Example: `Mr` -> `Mister`, `Dr` -> `Doctor`
- Normalize punctuation strictly:
  - quotation marks must be paired correctly
  - remove orphan or unmatched quotes
  - fix clearly broken punctuation spacing and pairing
- If a word is obviously truncated, corrupted, or incomplete, delete it or replace it with the most likely correct form.
- For English text, use ASCII characters only.
- For Chinese text, use standard Chinese characters.
- `canonical_text` should preserve the original meaning as much as possible while making the text clean, complete, and speakable.
- All tagged texts and text comparisons must be based on `canonical_text`, not the raw input plain text."""


NL_INSTRUCTION_DESC = """\
- Each natural language instruction is a sentence describing how to edit the source speech to get the target.
- Instructions should be concise and  (mix English and Chinese, some specific like "change to happy", some vague like "make it more emotional").
- Do NOT mention the emotion/style and their intensity of the source audio, e.g.,
    - Generate "Change the emotion to happy" instead of "Change the emotion from sad to happy".
    - Generate "Make the emotion more happy" instead of "Increase the happiness level from slight to strong"."""
NL_INSTRUCTION_DESC_MULTI_PLUS = """\
- **NEVER** use "segment", "sentence", "clause", "beginning", or similar terms to refer to the text parts. Instead, reference the exact text content whose tag is changing.
- **CRITICAL: Reference rules for text intervals**:
    - 8+ words: can use "from '...' to '...'" or "从 '...' 到 '...'" (e.g., "Make the part from 'I'm going to the park' to 'now' happy").
    - Under 8 words: MUST use explicit verbatim quoting (e.g., "Make 'the flowers' sad", "Change 'hello' to angry").
- Legal examples:
    - Make "I'm going to the park" happier.
    - Make the part from "To be" to "question" sad.
    - Make 'the flowers' sadder.
- Illegal:
    - Make the first segment happier."""

DESC_INTENSITY_ST_INSTRUCTION_DESC = """\
- Generate `st_inst` for ALL 6 ordered source-target pairs (1→2, 1→3, 2→3, and their reverses).
- `st_inst` is a partial tagged natural-language intensity edit instruction over the exact canonical text.
- Wrap only the changed text spans with `<emo desc="natural-language intensity edit instruction">...</emo>`.
- Do NOT include `type_ref` in `st_inst`; `type_ref` is only for the source/target synthesis tags.
- The text after stripping tags from each `st_inst` value must be EXACTLY `canonical_text`.
- Use concise desc values such as "make the happy delivery stronger", "soften the sad emotion", or "增强快乐的情感"."""


def _needs_desc_intensity_st_inst(emotion_edit_plan: Optional[dict]) -> bool:
    return (
        _plan_value(emotion_edit_plan, "edit_mode", "type_change")
        == "intensity_change"
        and _plan_value(emotion_edit_plan, "emotion_prompt_mode", "level")
        == "description"
    )


def _desc_intensity_st_section(emotion_edit_plan: Optional[dict]) -> str:
    if not _needs_desc_intensity_st_inst(emotion_edit_plan):
        return ""
    return f"""
## Part 4: st_inst
{DESC_INTENSITY_ST_INSTRUCTION_DESC}
"""


def _desc_intensity_st_json(emotion_edit_plan: Optional[dict]) -> str:
    if not _needs_desc_intensity_st_inst(emotion_edit_plan):
        return ""
    return """,
  "st_inst": {
    "1->2": "partial tagged natural-language intensity instruction",
    "1->3": "...",
    "2->1": "...",
    "2->3": "...",
    "3->1": "...",
    "3->2": "..."
  }"""

ST_INSTRUCTION_DESC_MULTI = """\
- Each structured instruction is a special tagged text describing the **difference** between the source and target tagged texts.
- It should only wrap the text segments with tags whose emotion/style is changing, while leaving unchanged segments untagged.
- Example 1 (if the natural language instruction is "Change the emotion of ... **to** happy"):
    - the source is
        <emo type="sad" level="2">I'm so tired,</emo><emo type="surprised" level="1">and I want to go to sleep now.</emo>
    - the target is
        <emo type="happy" level="2">I'm so tired,</emo><emo type="surprised" level="1">and I want to go to sleep now.</emo>
    - then the structured instruction should be
        <emo type="happy" level="2">I'm so tired,</emo>and I want to go to sleep now.
- Example 2 (if the natural language instruction is "Make the emotion of ... **more** sad"):
    - the source is
        <emo type="sad" level="1">I'm so tired,</emo><emo type="surprised" level="1">and I want to go to sleep now.</emo>
    - the target is
        <emo type="sad" level="3">I'm so tired,</emo><emo type="surprised" level="1">and I want to go to sleep now.</emo>
    - then the structured instruction should be
        <emo type="sad" level="3">I'm so tired,</emo>and I want to go to sleep now.
  Structured instructions must write the target state directly; do not use inc/dec placeholders."""

ST_INSTRUCTION_DESC_SINGLE = """\
- Each structured instruction is a special tagged text describing the **difference** between the source and target tagged texts.
- Example 1 (if the natural language instruction is "Change the emotion of ... **to** happy"):
    - the source is
        <emo type="sad" level="2">I'm so tired, and I want to go to sleep now.</emo>
    - the target is
        <emo type="happy" level="2">I'm so tired, and I want to go to sleep now.</emo>
    - then the structured instruction should be
        <emo type="happy" level="2">I'm so tired, and I want to go to sleep now.</emo>
- Example 2 (if the natural language instruction is "Make the emotion of ... **more** sad"):
    - the source is
        <emo type="sad" level="1">I'm so tired, and I want to go to sleep now.</emo>
    - the target is
        <emo type="sad" level="3">I'm so tired, and I want to go to sleep now.</emo>
    - then the structured instruction should be
        <emo type="sad" level="3">I'm so tired, and I want to go to sleep now.</emo>
  Structured instructions must write the target state directly; do not use inc/dec placeholders."""

# ---------------------------------------------------------------------------
# Tagged-text prompts
# ---------------------------------------------------------------------------


def build_single_prompt(
    plain_text: str,
    emotion_edit_plan: Optional[dict] = None,
    natural_language_instruction_mode: str = "llm",
) -> str:
    """Build prompt for single-tag generation.

    Generates 3 tagged text variants, each wrapping the full text in one tag.
    Natural-language instructions are optional. Structured instructions are
    always generated locally from tag diffs in the Plan stage.

    Args:
        plain_text: The plain text to wrap in tags.
    """
    tag_type_constraint = _format_emotion_control_desc(emotion_edit_plan)
    include_nl_inst = natural_language_instruction_mode.strip().lower() == "llm"
    include_st_inst = _needs_desc_intensity_st_inst(emotion_edit_plan)
    nl_section = (
        f"""
## Part 3: nl_inst
Then generate natural language edit instructions for ALL 6 ordered source-target pairs (1→2, 1→3, 2→3, and their reverses).
{NL_INSTRUCTION_DESC}
"""
        if include_nl_inst
        else ""
    )
    st_section = _desc_intensity_st_section(emotion_edit_plan)
    nl_json = (
        """,
  "nl_inst": {
    "1->2": "natural language instruction content",
    "1->3": "...",
    "2->1": "...",
    "2->3": "...",
    "3->1": "...",
    "3->2": "..."
  }"""
        if include_nl_inst
        else ""
    )
    st_json = _desc_intensity_st_json(emotion_edit_plan)
    if include_nl_inst and include_st_inst:
        task_desc = "Generate 3 tagged text variants, natural language edit instructions, and st_inst for all 6 ordered pairs."
    elif include_nl_inst:
        task_desc = "Generate 3 tagged text variants, then generate natural language edit instructions for all 6 ordered pairs."
    elif include_st_inst:
        task_desc = "Generate 3 tagged text variants and st_inst for all 6 ordered pairs. Do not generate natural-language edit instructions."
    else:
        task_desc = "Generate 3 tagged text variants. Do not generate edit instructions."

    return f"""\
You are a data generation assistant for speech editing.
{task_desc}

## Step 0: Text preprocessing
{TEXT_NORMALIZATION_DESC}

Available emotion tags for this sample: {_format_allowed_emotions(emotion_edit_plan)}
{TAG_FORMAT_DESC}

# Output Requirements
## Part 1: canonical_text
- Output `canonical_text` first.

## Part 2: tagged_texts
Given `canonical_text`, create 3 variants. Each variant wraps the ENTIRE text in a single tag. Requirements:
{tag_type_constraint}
- Use variant emotion controls according to the external emotion control plan.
- The text apart from tags must be EXACTLY `canonical_text`. The only difference is the emotion control attributes.

{nl_section}
{st_section}

Output JSON:
{{
  "canonical_text": "normalized spoken-form text",
  "tagged_texts": {{
    "1": "<emo type=\"happy\" level=\"2\">...full text...</emo>",
    "2": "...",
    "3": "..."
  }}{nl_json}{st_json}
}}

Plain text: {plain_text}"""


def build_multi_normal_prompt(
    plain_text: str,
    emotion_edit_plan: Optional[dict] = None,
    natural_language_instruction_mode: str = "llm",
) -> str:
    """Build prompt for multi-tag generation with gradual variation.

    The LLM decides where to place tag boundaries. Boundaries must be at
    semantically meaningful punctuation (commas, periods, etc.), and each
    segment must have relatively complete semantics (not too short).

    Generates 3 tagged text variants with gradual variation. Natural-language
    instructions are optional. Structured instructions are always generated
    locally from tag diffs in the Plan stage.

    Args:
        plain_text: Full plain text.
    """
    tag_constraint = _format_emotion_control_desc(emotion_edit_plan)
    include_nl_inst = natural_language_instruction_mode.strip().lower() == "llm"
    include_st_inst = _needs_desc_intensity_st_inst(emotion_edit_plan)
    nl_section = (
        f"""
## Part 3: nl_inst
Generate natural language edit instructions for ALL 6 ordered source-target pairs (1→2, 1→3, 2→3, and their reverses).
{NL_INSTRUCTION_DESC}
{NL_INSTRUCTION_DESC_MULTI_PLUS}
"""
        if include_nl_inst
        else ""
    )
    st_section = _desc_intensity_st_section(emotion_edit_plan)
    nl_json = (
        """,
  "nl_inst": {
    "1->2": "natural language instruction content",
    "1->3": "...",
    "2->1": "...",
    "2->3": "...",
    "3->1": "...",
    "3->2": "..."
  }"""
        if include_nl_inst
        else ""
    )
    st_json = _desc_intensity_st_json(emotion_edit_plan)
    if include_nl_inst and include_st_inst:
        task_desc = "Generate 3 tagged text variants with gradual variation, natural language edit instructions, and st_inst for all 6 ordered pairs."
    elif include_nl_inst:
        task_desc = "Generate 3 tagged text variants with gradual variation, then generate natural language edit instructions for all 6 ordered pairs."
    elif include_st_inst:
        task_desc = "Generate 3 tagged text variants with gradual variation and st_inst for all 6 ordered pairs. Do not generate natural-language edit instructions."
    else:
        task_desc = "Generate 3 tagged text variants with gradual variation. Do not generate edit instructions."

    return f"""\
You are a data generation assistant for speech editing.
{task_desc}

## Step 0: Text preprocessing
{TEXT_NORMALIZATION_DESC}

Available emotion tags for this sample: {_format_allowed_emotions(emotion_edit_plan)}
{TAG_FORMAT_DESC}

{tag_constraint}

# Output Requirements
## Part 1: canonical_text
- Output `canonical_text` first.

## Part 2: tagged_texts
Given `canonical_text`, split it into segments and create 3 tagged text variants. Requirements:

### Segmentation rules
- Split the text into **1 to 3** contiguous segment(s) that together cover the ENTIRE text.
- Tag boundaries must be placed at semantically meaningful punctuation marks (commas, periods, semicolons, colons, exclamation marks, question marks, or equivalent Chinese punctuation).
- The number of segments depends on the text length and the number of pause punctuation marks in the text. If no suitable punctuation exists for boundary placement, wrap the entire text in one tag (i.e., only 1 segment).
- Each segment must contain relatively complete semantics — do NOT split in the middle of a phrase or clause. Each segment must have at least 4 words. Avoid very short segments (less than 4 words or CJK characters).
- All 3 variants must use the SAME segmentation (same boundary positions).

### Gradual variation rules
- Each variant assigns one canonical `<emo>` control to each segment.
- Variant 1: choose emotion tags from `allowed_emotions` for all segments.
- Variant 2: change EXACTLY ONE segment from variant 1 according to `edit_mode`.
- Variant 3: change EXACTLY ONE segment from variant 2 according to `edit_mode`.
- Use diverse allowed emotion tags for type-change plans, or diverse levels for intensity-change plans.
- The emotion you choose could be nothing to do with the text content.

{nl_section}
{st_section}

Output JSON:
{{
  "canonical_text": "normalized spoken-form text",
  "tagged_texts": {{
    "1": "<emo type=\"happy\" level=\"2\">segment1</emo><emo type=\"sad\" level=\"2\">segment2</emo>...",
    "2": "...",
    "3": "..."
  }}{nl_json}{st_json}
}}

Plain text: {plain_text}"""


def build_multi_frequent_prompt(
    plain_text: str,
    emotion_edit_plan: Optional[dict] = None,
    natural_language_instruction_mode: str = "llm",
) -> str:
    """Build prompt for multi-emotion frequent-switching generation.

    The LLM decides where to place tag boundaries. Boundaries should be at
    semantically meaningful phrase edges where there is NO natural pause
    (no punctuation), simulating rapid emotion switching within a single
    utterance.

    Generates 3 tagged text variants with gradual variation. Natural-language
    instructions are optional. Structured instructions are always generated
    locally from tag diffs in the Plan stage.
    """

    include_nl_inst = natural_language_instruction_mode.strip().lower() == "llm"
    include_st_inst = _needs_desc_intensity_st_inst(emotion_edit_plan)
    nl_section = (
        f"""
## Part 3: nl_inst
Generate natural language edit instructions for ALL 6 ordered pairs (1→2, 1→3, 2→3, and their reverses).
Be concise (reference the text whose tag is being changed) and varied (mix English and Chinese instructions).
{NL_INSTRUCTION_DESC}
{NL_INSTRUCTION_DESC_MULTI_PLUS}
"""
        if include_nl_inst
        else ""
    )
    st_section = _desc_intensity_st_section(emotion_edit_plan)
    nl_json = (
        """,
  "nl_inst": {
    "1->2": "natural language instruction content",
    "1->3": "...",
    "2->1": "...",
    "2->3": "...",
    "3->1": "...",
    "3->2": "..."
  }"""
        if include_nl_inst
        else ""
    )
    st_json = _desc_intensity_st_json(emotion_edit_plan)
    if include_nl_inst and include_st_inst:
        task_desc = "Generate 3 tagged text variants with rapid emotion switching, natural language edit instructions, and st_inst for all 6 ordered pairs."
    elif include_nl_inst:
        task_desc = "Generate 3 tagged text variants with rapid emotion switching, then generate natural language edit instructions for all 6 ordered pairs."
    elif include_st_inst:
        task_desc = "Generate 3 tagged text variants with rapid emotion switching and st_inst for all 6 ordered pairs. Do not generate natural-language edit instructions."
    else:
        task_desc = "Generate 3 tagged text variants with rapid emotion switching. Do not generate edit instructions."
    return f"""\
You are a data generation assistant for speech editing.
{task_desc}

## Step 0: Text preprocessing
{TEXT_NORMALIZATION_DESC}

Available emotion tags for this sample: {_format_allowed_emotions(emotion_edit_plan)}
{TAG_FORMAT_DESC}

{_format_emotion_control_desc(emotion_edit_plan)}

# Output Requirements
## Part 1: canonical_text
- Output `canonical_text` first.

## Part 2: tagged_texts
Given `canonical_text`, split it into segments and create 3 tagged text variants.

### Segmentation rules
- Split the text into **2 or 3** contiguous segments that together cover the ENTIRE text.
- **CRITICAL**: Tag boundaries must be placed where there is NO natural pause and NO punctuation, but where there IS a clear semantic phrase boundary. The goal is to simulate rapid, mid-sentence emotion switching.
- **CRITICAL: Each segment must have at least 4 words**. Do NOT create segments with fewer than 4 words.
- Recommended boundary placement patterns:
  - Split off a trailing clause with at least 4 words.
  - Split off a leading clause with at least 4 words.
  - Split at a mid-sentence conjunction or transition with at least 4 words.
- All 3 variants must use the SAME segmentation (same boundary positions).

### Tagged text rules
- Each variant assigns one emotion tag (with level) to each segment.
- Variant 1: choose emotion tags from `allowed_emotions` for all segments.
- Variant 2: change EXACTLY ONE segment from variant 1 according to `edit_mode`.
- Variant 3: change EXACTLY ONE segment from variant 2 according to `edit_mode`.
- Use diverse allowed emotion tags for type-change plans, or diverse levels for intensity-change plans.
- The emotion you choose could be nothing to do with the text content.

CRITICAL: The text inside tags must be EXACTLY `canonical_text`. Do NOT add, remove, or change any characters.

### Example
Plain text: "I want to go to the park to see the beautiful flowers"

Variant 1: <emo type="happy" level="2">I want to go to the park to </emo><emo type="sad" level="2">see the beautiful flowers.</emo>
Variant 2: <emo type="happy" level="2">I want to go to the park to </emo><emo type="angry" level="2">see the beautiful flowers.</emo>
Variant 3: <emo type="surprised" level="2">I want to go to the park to </emo><emo type="angry" level="2">see the beautiful flowers.</emo>

Note: Each segment has at least 4 words. The boundary is placed before "see the beautiful flowers" — a coherent noun phrase at the end, with no punctuation at the split point.

{nl_section}
{st_section}

Output JSON:
{{
  "canonical_text": "normalized spoken-form text, then split into segments",
  "tagged_texts": {{
    "1": "<emo type=\"happy\" level=\"2\">...segment1...</emo><emo type=\"sad\" level=\"2\">...segment2...</emo>",
    "2": "...",
    "3": "..."
  }}{nl_json}{st_json}
}}

Raw plain text: {plain_text}"""


# ---------------------------------------------------------------------------
# Text Rewrite Prompt
# ---------------------------------------------------------------------------


def build_rewrite_prompt(plain_text: str) -> str:
    """Build prompt to rewrite text while keeping style/topic similar.

    The rewritten text must NOT be longer than the original (character count).
    """
    return f"""\
Rewrite the following text. Keep the same topic and style. The rewritten text must NOT be longer than the original (character count ≤ {len(plain_text)}).

Output ONLY the rewritten text, nothing else.

Original text: {plain_text}"""


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def get_prompt(
    sub_type: str,
    plain_text: str,
    *,
    emotion_edit_plan: Optional[dict] = None,
    natural_language_instruction_mode: str = "llm",
) -> str:
    """Build the prompt for the synthesis mode selected by the sample stage."""
    if emotion_edit_plan and emotion_edit_plan.get("strict"):
        return build_strict_emotion_matrix_prompt(
            plain_text, sub_type, emotion_edit_plan,
            natural_language_instruction_mode=natural_language_instruction_mode,
        )
    if sub_type == "single":
        return build_single_prompt(
            plain_text, emotion_edit_plan=emotion_edit_plan,
            natural_language_instruction_mode=natural_language_instruction_mode,
        )
    if sub_type == "multi_emo_frequent":
        return build_multi_frequent_prompt(
            plain_text, emotion_edit_plan=emotion_edit_plan,
            natural_language_instruction_mode=natural_language_instruction_mode,
        )
    if sub_type == "multi_emo_normal":
        return build_multi_normal_prompt(
            plain_text, emotion_edit_plan=emotion_edit_plan,
            natural_language_instruction_mode=natural_language_instruction_mode,
        )
    raise ValueError(f"Unknown synthesis mode: {sub_type}")
