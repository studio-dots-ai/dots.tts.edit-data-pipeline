"""Tests for pause_edit_pipeline.prompts."""

import json
import re

from speech_edit.pipelines.pause.prompts import (
    EXAMPLES,
    build_pause_edit_prompt,
    get_prompt,
)
from speech_edit.pipelines.pause.tag_utils import parse_pause_tags, strip_pause_tags, validate_pause_tags

PROFILE_TARGETS = [
    {
        "profile_id": 1,
        "num_operations": 2,
        "boundary_types": ["punct", "nonpunct"],
        "levels": [3, 2],
    }
]


class TestBuildPauseEditPrompt:
    def test_json_examples_are_valid_and_preserve_source_text(self):
        prompt = get_prompt("Hello world.", [200, 500, 800], PROFILE_TARGETS)
        blocks = re.findall(r"```json\n(.*?)```", prompt, re.DOTALL)
        assert len(blocks) == 3
        for block in blocks:
            json.loads(block)
        example = json.loads(re.search(r"```json\n(.*?)```", EXAMPLES, re.DOTALL).group(1))
        plain = "She walked slowly through the beautiful garden."
        assert strip_pause_tags(example["operation"]) == plain
        assert validate_pause_tags(example["operation"], plain) == (True, "")

    def test_chinese_example_matches_ordered_boundary_targets(self):
        examples = re.findall(r"```json\n(.*?)```", EXAMPLES, re.DOTALL)
        example = json.loads(examples[1])
        plain = "下雨了,我们决定明天再出发。"
        assert strip_pause_tags(example["operation"]) == plain
        assert validate_pause_tags(example["operation"], plain) == (True, "")
        tags = parse_pause_tags(example["operation"], plain)
        assert ["punct" if tag.punctuation else "nonpunct" for tag in tags] == [
            "punct", "nonpunct", "nonpunct"
        ]
        assert [tag.level for tag in tags] == [3, 2, 1]
        invalid = example["operation"].replace(
            ',<pause act="ins" level="3"/>', '<pause act="ins" level="3"/>,', 1
        )
        assert validate_pause_tags(invalid, plain)[0] is False

    def test_basic_prompt(self):
        prompt = build_pause_edit_prompt(
            plain_text="He ran quickly, but she kept walking.",
            duration_pool=[200, 500, 800],
            profile_targets=PROFILE_TARGETS,
        )
        assert "He ran quickly, but she kept walking." in prompt
        assert "1->200ms" in prompt
        assert "2->500ms" in prompt
        assert "3->800ms" in prompt
        assert "exactly 1 pause edit profile" in prompt
        assert "profile_id=1: num_operations=2" in prompt

    def test_contains_level_based_tag_format(self):
        prompt = build_pause_edit_prompt(
            plain_text="Hello world.",
            duration_pool=[200, 500, 800],
            profile_targets=PROFILE_TARGETS,
        )
        assert '<pause act="ins" level="1"/>' in prompt
        assert '<pause act="ins" level="2"/>' in prompt
        assert '<pause act="ins" level="3"/>' in prompt
        assert '<pause act="red" level="2"/>' not in prompt
        assert '<pause act="rem"' not in prompt
        assert '"operation"' in prompt
        assert '"decrease_pause"' not in prompt

    def test_contains_examples(self):
        prompt = build_pause_edit_prompt(
            plain_text="Hello world.",
            duration_pool=[200, 500, 800],
            profile_targets=PROFILE_TARGETS,
        )
        assert "Invalid examples" in prompt
        assert "non-punctuation positions" in prompt or "non-punctuation" in prompt
        assert "interior boundary position" in prompt
        assert "Generate exactly these profiles" in prompt

    def test_profile_targets_control_operation_slots(self):
        prompt = build_pause_edit_prompt(
            plain_text="Hello world.",
            duration_pool=[200, 500, 800],
            profile_targets=PROFILE_TARGETS,
        )
        assert "boundary_types=['punct', 'nonpunct']" in prompt
        assert "levels=[3, 2]" in prompt
        assert "DIFFERENT interior boundary positions" in prompt

    def test_get_prompt_alias(self):
        p1 = get_prompt("Hello.", [200, 500, 800], PROFILE_TARGETS)
        p2 = build_pause_edit_prompt("Hello.", [200, 500, 800], PROFILE_TARGETS)
        assert p1 == p2

    def test_prompt_contains_chinese_surface_form_guard(self):
        prompt = build_pause_edit_prompt(
            plain_text="大家都说越努力,越幸运这句话不适用于钉钉 app.",
            duration_pool=[200, 500, 800],
            profile_targets=PROFILE_TARGETS,
        )
        assert "Never insert extra spaces between adjacent Chinese characters" in prompt
        assert "钉钉 app" in prompt
        assert "must match `normed_text` exactly character-for-character" in prompt

    def test_prompt_contains_retry_feedback(self):
        prompt = build_pause_edit_prompt(
            plain_text="你好世界。",
            duration_pool=[200, 500, 800],
            profile_targets=PROFILE_TARGETS,
            validation_errors=[
                "Profile 1: st_inst[increase_pause] text mismatch",
                "Profile 2: tag validation failed: Text mismatch after stripping tags.",
            ],
        )
        assert "Previous Attempt Feedback" in prompt
        assert "st_inst[increase_pause] text mismatch" in prompt
        assert "Do not rewrite English words, grammar, tense, digits, abbreviations, or disfluency spellings" in prompt
        assert "Do not add extra spaces into Chinese or mixed Chinese-English text" in prompt

    def test_prompt_contains_english_surface_form_guard(self):
        prompt = build_pause_edit_prompt(
            plain_text="Marcel said he would learned of Gregory's abduction at around 9pm.",
            duration_pool=[200, 500, 800],
            profile_targets=PROFILE_TARGETS,
        )
        assert "English Surface Form Guard" in prompt
        assert "Set `normed_text` equal to the readable input plain text exactly" in prompt
        assert "Keep `9pm` as `9pm`, not `nine p m`" in prompt
        assert "Keep `u` as `u`, not `you`" in prompt
