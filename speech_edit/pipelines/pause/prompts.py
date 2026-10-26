"""
LLM prompt templates for pause editing pipeline.

The LLM receives a plain text plus a fixed 3-level pause mapping. It generates
multiple pause-edit profiles, where each profile may contain one or more pause
insertions at distinct boundary positions.

Public API:
    get_prompt(plain_text, duration_pool, profile_targets) -> str
"""

from typing import Iterable, List, Mapping, Optional

from .tag_utils import ALL_PAUSE_PUNCTUATION


TEXT_NORMALIZATION_DESC = """\
- First preprocess the input plain text into `normed_text` before generating any tags or instructions.
- Default behavior: copy the input plain text into `normed_text` exactly.
- Only change `normed_text` if the raw input is clearly broken, unreadable, or malformed.
- Do not proactively improve grammar, tense, wording, spelling, or spoken-form rendering when the input is already readable.
- If punctuation is obviously broken, keep any repair minimal and preserve the original lexical text.
- For English text, use ASCII characters only.
- For Chinese text, use standard Chinese characters.
- `normed_text` should preserve the original meaning as much as possible while making the text clean, complete, and speakable.
- All operations, structured instructions, and text comparisons must be based on `normed_text`, not the raw input plain text."""


PAUSE_TAG_FORMAT_DESC = """\
## Pause Tag Format

Use exactly one inline self-closing XML tag format:
- `<pause act="ins" level="1"/>`
- `<pause act="ins" level="2"/>`
- `<pause act="ins" level="3"/>`

Rules:
- Only `act="ins"` is allowed.
- Tags are point markers. They do NOT wrap text.
- Tags may be placed at any interior word boundary, including non-punctuation positions.
- Do not place tags inside a word, at the start of the text, or at the end of the text.
- A boundary with punctuation has one position: AFTER the complete punctuation run. Never put a tag before punctuation or between punctuation marks.
- Within one profile, every tag must use a DIFFERENT boundary position.
"""


LEVEL_MAPPING_DESC = """\
## Pause Levels

Use only these 3 discrete pause levels:
- `level="1"` = short pause
- `level="2"` = medium pause
- `level="3"` = long pause

The runtime maps these levels to concrete durations. Do NOT output millisecond values anywhere in the pause tag.
"""


POSITION_SELECTION_GUIDE = """\
## Position Selection

- Within one profile, use exactly the requested number of pause positions.
- Every position inside one profile must be DIFFERENT.
- Both punctuation and non-punctuation boundaries are allowed.
- Prefer linguistically plausible boundaries such as phrase edges, clause edges, light breath positions, or meaningful semantic breaks.
- Do not place a pause inside a tightly bound phrase if a more natural boundary is available.
- Keep the lexical text unchanged. You are only deciding where to add extra silence.
- Across different profiles, avoid returning identical pause-position sets.
"""


CHINESE_SURFACE_FORM_GUARD = """\
## Chinese and Mixed-Script Surface Form Guard

- For Chinese text, after removing pause tags, the remaining text must match `normed_text` exactly character-for-character.
- Never insert extra spaces between adjacent Chinese characters.
- Never insert extra spaces before or after Chinese punctuation such as `，。！？；：、“”‘’（）《》`.
- Preserve the exact existing spacing pattern in `normed_text` for mixed Chinese-English text.
  - Example: if `normed_text` contains `钉钉 app`, keep exactly that one space.
  - Do NOT rewrite it as `钉钉app`, `钉钉  app`, or `钉钉 app `.
- Do not add any extra spaces around commas, periods, or other punctuation unless that exact space already exists in `normed_text`.
- Apply this exact-copy rule to all of these strings:
  - `operation`
"""


ENGLISH_SURFACE_FORM_GUARD = """\
## English Surface Form Guard

- For English text, assume the input plain text is already the canonical text unless it is truly unreadable.
- Set `normed_text` equal to the readable input plain text exactly.
- After removing pause tags, the lexical text in all structured strings must remain an exact copy of `normed_text`.
- Do NOT correct grammar, tense, agreement, articles, prepositions, or word choice.
- Do NOT rewrite abbreviations, contractions, digits, times, years, currencies, or percentages into spoken form when they already appear in the input.
- Do NOT normalize ASR-style spellings or disfluencies.
  - Keep `u` as `u`, not `you`.
  - Keep `9pm` as `9pm`, not `nine p m`.
  - Keep `yeaah` as `yeaah`, not `yeah`.
- The only intended surface change for English is inserting the pause tags themselves.
"""


ST_INSTRUCTION_DESC = """\
## Structured Instructions

The structured operation is the `normed_text` with pause insertion tags inserted at the edit positions.

Rules:
- `operation` must be the canonical forward tagged text using `act="ins"`.
- After stripping the pause tag, the text must be identical to `normed_text`.
- Do not insert any additional spaces into the lexical text while adding pause tags.
- Do not generate reverse-direction structured or natural language instructions. The pipeline constructs them deterministically.
"""


def _retry_feedback_block(validation_errors: Optional[Iterable[str]]) -> str:
    if not validation_errors:
        return ""
    unique_errors: List[str] = []
    seen: set[str] = set()
    for item in validation_errors:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique_errors.append(text)
    if not unique_errors:
        return ""
    bullets = "\n".join(f"- {item}" for item in unique_errors[:12])
    return f"""\
## Previous Attempt Feedback

The previous output failed validation. Regenerate the entire JSON object from scratch.
Fix every issue below exactly and do not repeat any of them:
{bullets}

Most importantly:
- Do not rewrite English words, grammar, tense, digits, abbreviations, or disfluency spellings.
- Do not add extra spaces into Chinese or mixed Chinese-English text.
- Do not place pause tags at invalid boundaries.
- Ensure every stripped structured instruction exactly matches `normed_text`.
"""


OUTPUT_FORMAT_DESC = r"""
## Output Format

Return a JSON object:
```json
{
  "normed_text": "normalized spoken-form text with original punctuation preserved",
  "profiles": [
    {
      "profile_id": 1,
      "operation": "normed_text with one or more <pause act=\"ins\" level=\"2\"/> tags inserted"
    }
  ]
}
```

Important:
- Use only `level="1"`, `level="2"`, or `level="3"`.
- Each profile must target a DIFFERENT interior boundary position.
- `profile_id` starts from 1.
"""


EXAMPLES = r"""
## Example

Input text: `She walked slowly through the beautiful garden.`

Valid single-pause profile:
```json
{
  "profile_id": 1,
  "operation": "She walked slowly<pause act=\"ins\" level=\"2\"/> through the beautiful garden."
}
```

Chinese input: `下雨了,我们决定明天再出发。`

Valid three-pause profile for boundary_types=['punct', 'nonpunct', 'nonpunct'] and levels=[3, 2, 1], in left-to-right order:
```json
{
  "profile_id": 1,
  "operation": "下雨了,<pause act=\"ins\" level=\"3\"/>我们决定<pause act=\"ins\" level=\"2\"/>明天<pause act=\"ins\" level=\"1\"/>再出发。"
}
```

Invalid examples:
- inserting inside a word
- inserting immediately BEFORE a comma to satisfy a `nonpunct` slot
- inserting after the final word or final punctuation mark
- using `rem`, `red`, or any millisecond duration in the generated operation tag
- repeating the same boundary position twice in one profile
"""


def _profile_targets_desc(profile_targets: List[Mapping[str, object]]) -> str:
    pause_punctuation = " ".join(sorted(ALL_PAUSE_PUNCTUATION))
    lines = [
        "## Profile Targets",
        "",
        "Generate exactly these profiles. Each pause tag must follow its slot in order.",
        f"- `boundary_types`: `punct` means immediately after one of these pause punctuation characters: {pause_punctuation}. `nonpunct` means any other valid word or phrase boundary. Neither type permits a position immediately BEFORE punctuation or inside a punctuation run.",
        "- Match slots in left-to-right text order. For ['punct', 'nonpunct', 'nonpunct'], put the two nonpunct tags later in the text than the punct tag.",
        "- `levels`: use the exact pause level for each slot.",
        "",
    ]
    for target in profile_targets:
        lines.append(
            "- profile_id={profile_id}: num_operations={num_operations}, "
            "boundary_types={boundary_types}, levels={levels}".format(
                profile_id=target["profile_id"],
                num_operations=target["num_operations"],
                boundary_types=target["boundary_types"],
                levels=target["levels"],
            )
        )
        for index, (kind, level) in enumerate(zip(target["boundary_types"], target["levels"]), 1):
            position = (
                "immediately AFTER pause punctuation"
                if kind == "punct"
                else "at a word/phrase boundary that is NOT after pause punctuation and NOT before punctuation"
            )
            lines.append(
                f"Profile {target['profile_id']}, tag {index} from left to right: level={level}, {position}."
            )
    return "\n".join(lines)


def build_pause_edit_prompt(
    plain_text: str,
    duration_pool: List[int],
    profile_targets: List[Mapping[str, object]],
    *,
    validation_errors: Optional[Iterable[str]] = None,
) -> str:
    """Build prompt for generating pause edit profiles."""
    duration_str = ", ".join(f"{idx + 1}->{dur}ms" for idx, dur in enumerate(duration_pool))
    retry_feedback = _retry_feedback_block(validation_errors)
    profile_targets_desc = _profile_targets_desc(profile_targets)
    num_profiles = len(profile_targets)

    return f"""\
You are a data generation assistant for speech pause editing.
Generate exactly {num_profiles} pause edit profile(s) for the given text.

## Step 0: Text preprocessing
{TEXT_NORMALIZATION_DESC}

{PAUSE_TAG_FORMAT_DESC}

{LEVEL_MAPPING_DESC}

Runtime level mapping for this call: [{duration_str}]

{POSITION_SELECTION_GUIDE}

{ENGLISH_SURFACE_FORM_GUARD}

{CHINESE_SURFACE_FORM_GUARD}

{ST_INSTRUCTION_DESC}

{EXAMPLES}

{OUTPUT_FORMAT_DESC}

{retry_feedback}

Generate the requested profiles now. Use exactly the specified number of DIFFERENT interior boundary positions for each profile and assign exactly the specified level to each chosen position.

{profile_targets_desc}

Plain text: {plain_text}"""


def get_prompt(
    plain_text: str,
    duration_pool: List[int],
    profile_targets: List[Mapping[str, object]],
    *,
    validation_errors: Optional[Iterable[str]] = None,
) -> str:
    """Get the pause edit prompt."""
    return build_pause_edit_prompt(
        plain_text=plain_text,
        duration_pool=duration_pool,
        profile_targets=profile_targets,
        validation_errors=validation_errors,
    )
