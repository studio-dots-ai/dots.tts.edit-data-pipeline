"""Prompt builders for the text edit pipeline."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from .generation_plan import GenerationSpec


NL_SCHEMA = {
    "simple_en": ["...", "..."],
    "simple_zh": ["...", "..."],
    "hard_en": ["...", "..."],
    "hard_zh": ["...", "..."],
}


INSTRUCTION_GUIDE = """
## Natural Language Instruction Requirements

For each generation, generate:
- 4 simple instructions: 2 English + 2 Chinese. They should be direct and template-like.
- 4 hard instructions: 2 English + 2 Chinese. They should be natural, varied, and conversational.

Simple examples:
- EN substitution: "Replace 'old' with 'new'." / "Change 'old' to 'new'."
- ZH substitution: "把'old'改成'new'。" / "将'old'替换为'new'。"
- EN insertion: "Insert 'new' after 'anchor'." / "Add 'new' at the beginning."
- ZH insertion: "在'anchor'后插入'new'。" / "在句首插入'new'。"
- EN deletion: "Delete 'old'." / "Remove 'old'."
- ZH deletion: "删除'old'。" / "把'old'去掉。"

Hard instructions may explain intent, but every instruction must still unambiguously describe exactly the same edit.
Use quotes around source and target text. Do not include XML edit tags in natural language instructions.

Language-slot rules:
- `simple_en` and `hard_en`: the instruction wording should be English.
- `simple_zh` and `hard_zh`: the instruction wording should be Chinese.
- The language rule applies to the instruction wording, not to quoted source/target text.
- When referring to source or target text, quote it exactly in its original language.
""".strip()


TAG_GUIDE = """
## Tagged Text Requirements

Use only these structural edit tags:
- <del>old text</del>
- <ins>new text</ins>
- <sub targ="new text">old text</sub>

Hard contract:
- Start from the canonical source text copied character-for-character, then wrap or insert tags.
- Removing all edit tags while keeping the old text inside <del> and <sub> must exactly recover the canonical source text.
- Do not add, drop, rewrite, reorder, normalize, or paraphrase any unedited source character.
- The source text has canonical whitespace: do not restore leading/trailing spaces or spaces around punctuation.
- Close tags exactly as </del>, </ins>, and </sub>. Never use HTML tags such as </a> or </b>.
- Do not escape structural edit tags as <\/del>, <\/ins>, or <\/sub>.
- Do not use nested, overlapping, or crossing tags.
- Escape &, <, >, and " inside tag text and <sub targ="..."> attributes.
  Apostrophes may stay as plain apostrophes unless JSON escaping requires otherwise.
""".strip()


OP_COUNT_GUIDE = """
## Edit Counting Rules

Every <del>...</del>, <ins>...</ins>, and <sub targ="...">...</sub> counts as exactly one edit.
- A replacement MUST be one <sub> tag. Do not express one replacement as <del>old</del><ins>new</ins>.
- A deletion MUST be one <del> tag.
- An insertion MUST be one <ins> tag.
- For op_count=1, output exactly one edit tag total.
- For op_count=2, output exactly two edit tags total.
- For op_count=3, output exactly three edit tags total.
Before returning, count the edit tags in each generation and verify it equals that generation's op_count.
""".strip()


SPAN_LEVEL_GUIDE = """
## Span Granularity Rules

- word: edit complete words or complete Chinese words/short terms; do not edit partial English words.
- phrase: edit short contiguous phrases, usually 2-4 words or one short Chinese phrase.
- clause: edit one or two natural clause-sized spans; keep boundaries at phrase or punctuation edges.
- arbitrary: choose natural word, phrase, or clause spans.
- sentence: use exactly one <sub> tag whose old text is the full canonical source text.
""".strip()


FORWARD_FEW_SHOT = """
## Valid Forward Examples

Example A, one English word substitution:
Canonical Source Text: "hello brave world"
Plan: {"000":{"span_level":"word","op_count":1}}
Valid generation:
{
  "generations": {
    "000": {
      "st": "hello <sub targ=\\\"kind\\\">brave</sub> world",
      "nl": {
        "simple_en": ["Replace 'brave' with 'kind'.", "Change 'brave' to 'kind'."],
        "simple_zh": ["把'brave'改成'kind'。", "将'brave'替换为'kind'。"],
        "hard_en": ["Please make the word 'brave' say 'kind' instead.", "Swap 'brave' for 'kind' while keeping the rest unchanged."],
        "hard_zh": ["请把'brave'换成'kind'，其他内容不变。", "把原来的'brave'调整为'kind'。"]
      }
    }
  }
}

Example B, two Chinese substitutions:
Canonical Source Text: "他今天来到学校,然后开始上课。"
Plan: {"000":{"span_level":"phrase","op_count":2}}
Valid generation:
{
  "generations": {
    "000": {
      "st": "他<sub targ=\\\"明天\\\">今天</sub>来到<sub targ=\\\"图书馆\\\">学校</sub>,然后开始上课。",
      "nl": {
        "simple_en": ["Replace '今天' with '明天' and '学校' with '图书馆'.", "Change '今天' to '明天', and change '学校' to '图书馆'."],
        "simple_zh": ["把'今天'改成'明天'，把'学校'改成'图书馆'。", "将'今天'替换为'明天'，并将'学校'替换为'图书馆'。"],
        "hard_en": ["Please update the time from '今天' to '明天' and the place from '学校' to '图书馆'.", "Make two edits: change '今天' into '明天', then change '学校' into '图书馆'."],
        "hard_zh": ["请把时间从'今天'改为'明天'，地点从'学校'改为'图书馆'。", "这里需要两处修改：把'今天'换成'明天'，再把'学校'换成'图书馆'。"]
      }
    }
  }
}
""".strip()


REVERSE_FEW_SHOT = """
## Valid Reverse Instruction Example

Input reverse item:
{
  "generation_id": "000",
  "source_text": "hello kind world",
  "target_text": "hello brave world",
  "st": "hello <sub targ=\\\"brave\\\">kind</sub> world"
}
Valid output:
{
  "instructions": {
    "000": {
      "nl": {
        "simple_en": ["Replace 'kind' with 'brave'.", "Change 'kind' to 'brave'."],
        "simple_zh": ["把'kind'改成'brave'。", "将'kind'替换为'brave'。"],
        "hard_en": ["Restore the edited word by changing 'kind' back to 'brave'.", "Please swap 'kind' for 'brave' and leave the rest unchanged."],
        "hard_zh": ["请把'kind'恢复成'brave'。", "将修改后的'kind'换回原来的'brave'。"]
      }
    }
  }
}
""".strip()


def build_forward_generation_prompt(
    canonical_text: str,
    *,
    specs: Sequence[GenerationSpec],
    min_edit_distance_pct: float | None = None,
) -> str:
    generation_plan = {
        spec.generation_id: spec.to_prompt_dict()
        for spec in specs
    }
    skeleton = {
        "generations": {
            spec.generation_id: {
                "st": "<tagged canonical source text>",
                "nl": NL_SCHEMA,
            }
            for spec in specs
        }
    }
    large_edit_block = ""
    if min_edit_distance_pct and min_edit_distance_pct > 0:
        threshold_pct = min_edit_distance_pct * 100.0
        large_edit_block = f"""

## Large-Edit Requirement

- Make each target text substantially different from the source using broad insertions, deletions, or substitutions.
- Keep part of the source outside edit tags; do not rewrite the entire sentence unless the plan uses sentence span_level.
- Substitutions may be long-to-short, short-to-long, or long-to-long, and the edit distance must reach at least {threshold_pct:.1f}% of the source or target text.
- For multi-edit plans, spread edits across spans and leave visible unchanged text between or around edits.
""".rstrip()
    return f"""
You are an expert speech text edit data generator.
Return JSON only. Do not explain.

## Task
Generate {len(specs)} tagged text edit generations for the canonical source text.
For every generation, return both:
1. `st`: the canonical tagged text edit.
2. `nl`: natural language instructions for that same edit.

{TAG_GUIDE}

{OP_COUNT_GUIDE}

{SPAN_LEVEL_GUIDE}

{large_edit_block}

## Generation Plan
{json.dumps(generation_plan, ensure_ascii=False, indent=2)}

Rules:
- Each generation must strictly follow its own span_level and op_count.
- The `generations` object must contain exactly these generation ids: {json.dumps(list(generation_plan), ensure_ascii=False)}.
- Make edits meaningful and realistic for speech editing data.

{INSTRUCTION_GUIDE}

{FORWARD_FEW_SHOT}

Output contract:
- JSON parses and has no markdown fences.
- Every requested generation id is present exactly once.
- The tagged `st` preserves the full canonical source text except for valid edit tags.
- The edit-tag count equals op_count for every generation.
- Instruction field names select the instruction wording language; quoted source/target text stays in its original language.

## Required JSON Shape
{json.dumps(skeleton, ensure_ascii=False, indent=2)}

## Canonical Source Text
{json.dumps(canonical_text, ensure_ascii=False)}
""".strip()


def build_reverse_instruction_prompt(items: Sequence[Dict[str, Any]]) -> str:
    skeleton = {
        "instructions": {
            str(item["generation_id"]): {"nl": NL_SCHEMA}
            for item in items
        }
    }
    return f"""
You are an expert speech text edit instruction writer.
Return JSON only. Do not explain.

## Task
Generate reverse-direction natural language instructions for each tagged edit below.
The reverse tagged text `st` is fixed. Do not rewrite it. Only describe it in natural language.

{INSTRUCTION_GUIDE}

{REVERSE_FEW_SHOT}

Output contract:
- JSON parses and has no markdown fences.
- The `instructions` object contains exactly these generation ids: {json.dumps([str(item["generation_id"]) for item in items], ensure_ascii=False)}.
- Do not output or rewrite `st`.
- Instruction field names select the instruction wording language; quoted source/target text stays in its original language.

## Inputs
{json.dumps(list(items), ensure_ascii=False, indent=2)}

## Required JSON Shape
{json.dumps(skeleton, ensure_ascii=False, indent=2)}
""".strip()
