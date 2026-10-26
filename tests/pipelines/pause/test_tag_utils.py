"""Tests for pause_edit_pipeline.tag_utils."""

import pytest

from speech_edit.pipelines.pause.tag_utils import pause_boundaries



from speech_edit.pipelines.pause.tag_utils import (
    INSTRUCTION_ACTIONS,
    PAUSE_LEVEL_TO_DURATION_MS,
    PAUSE_TAG_PATTERN,
    compute_generated_text,
    inject_duration,
    make_text_versions,
    parse_instruction_pause_tags,
    parse_pause_tags,
    reverse_operation,
    strip_pause_tags,
    tags_to_aligner_queries,
    tags_to_boundary_queries,
    validate_pause_tags,
)


class TestPauseTagPattern:
    def test_ins_with_level(self):
        m = PAUSE_TAG_PATTERN.search('<pause act="ins" level="2"/>')
        assert m is not None
        assert m.group(1) == "ins"
        assert m.group(2) == "2"

    def test_rejects_rem(self):
        assert PAUSE_TAG_PATTERN.search('<pause act="rem"/>') is None


class TestParsePauseTags:
    def test_single_insert_at_non_punctuation_boundary(self):
        tagged = 'She walked slowly <pause act="ins" level="2"/> through the garden.'
        tags = parse_pause_tags(tagged, strip_pause_tags(tagged))
        assert len(tags) == 1
        assert tags[0].act == "ins"
        assert tags[0].level == 2
        assert tags[0].delta_dur_ms == 500
        assert tags[0].punctuation is None

    def test_no_tags(self):
        assert parse_pause_tags("Just plain text.", "Just plain text.") == []

    def test_multiple_insert_positions(self):
        tagged = (
            'She walked<pause act="ins" level="1"/> slowly through the '
            'beautiful<pause act="ins" level="2"/> garden.'
        )
        tags = parse_pause_tags(tagged, strip_pause_tags(tagged))
        assert len(tags) == 2
        assert [tag.level for tag in tags] == [1, 2]


class TestParseInstructionPauseTags:
    def test_reduction_keeps_positive_duration_magnitude(self):
        tagged = 'She walked slowly <pause act="red" level="2"/> through the garden.'
        tags = parse_instruction_pause_tags(tagged, strip_pause_tags(tagged))

        assert len(tags) == 1
        assert tags[0].act == "red"
        assert tags[0].level == 2
        assert tags[0].delta_dur_ms == 500
        assert tags[0].char_idx == len("She walked slowly")


class TestStripPauseTags:
    def test_strip_insert_tag(self):
        tagged = 'She walked slowly <pause act="ins" level="2"/> through the garden.'
        assert strip_pause_tags(tagged) == "She walked slowly  through the garden."


class TestValidatePauseTags:
    def test_valid_insert_at_non_punctuation_boundary(self):
        normed = "She walked slowly through the garden."
        tagged = 'She walked slowly <pause act="ins" level="2"/> through the garden.'
        valid, err = validate_pause_tags(tagged, normed)
        assert valid, err

    def test_invalid_insert_at_start(self):
        normed = "She walked slowly through the garden."
        tagged = '<pause act="ins" level="2"/> She walked slowly through the garden.'
        valid, err = validate_pause_tags(tagged, normed)
        assert not valid
        assert "start" in err.lower()

    def test_invalid_insert_at_end(self):
        normed = "She walked slowly through the garden."
        tagged = 'She walked slowly through the garden. <pause act="ins" level="2"/>'
        valid, err = validate_pause_tags(tagged, normed)
        assert not valid
        assert "end" in err.lower()

    def test_valid_multiple_insert_positions(self):
        normed = "She walked slowly through the beautiful garden."
        tagged = (
            'She walked<pause act="ins" level="1"/> slowly through the '
            'beautiful<pause act="ins" level="2"/> garden.'
        )
        valid, err = validate_pause_tags(tagged, normed)
        assert valid, err

    def test_reduction_requires_explicit_instruction_actions(self):
        normed = "She walked slowly through the garden."
        tagged = 'She walked slowly <pause act="red" level="2"/> through the garden.'
        valid, _ = validate_pause_tags(tagged, normed)
        assert not valid
        valid, err = validate_pause_tags(
            tagged,
            normed,
            allowed_actions=INSTRUCTION_ACTIONS,
        )
        assert valid, err
        assert parse_instruction_pause_tags(tagged, strip_pause_tags(tagged))[0].act == "red"

    def test_mixed_instruction_actions_are_valid(self):
        normed = "One two three four."
        tagged = (
            'One<pause act="ins" level="1"/> two three'
            '<pause act="red" level="2"/> four.'
        )
        valid, err = validate_pause_tags(
            tagged,
            normed,
            allowed_actions=INSTRUCTION_ACTIONS,
        )
        assert valid, err

    @pytest.mark.parametrize(
        ("allowed_actions", "rejected_action"),
        [(None, "red"), ({"ins"}, "red"), ({"red"}, "ins")],
    )
    def test_mixed_actions_cannot_hide_an_unauthorized_tag(self, allowed_actions, rejected_action):
        tagged = (
            'One<pause act="ins" level="1"/> two three'
            '<pause act="red" level="2"/> four.'
        )
        valid, error = validate_pause_tags(
            tagged, "One two three four.", allowed_actions=allowed_actions,
        )
        assert not valid
        assert error == f"Invalid action: {rejected_action}"

    def test_duplicate_insert_positions_rejected(self):
        normed = "Hello world."
        tagged = 'Hello<pause act="ins" level="1"/><pause act="ins" level="2"/> world.'
        valid, err = validate_pause_tags(tagged, normed)
        assert not valid
        assert "duplicate" in err.lower()

    def test_chinese_spacing_around_punctuation_is_tolerated(self):
        normed = "大家都说越努力,越幸运这句话不适用于钉钉app."
        tagged = '大家都说越努力,<pause act="ins" level="1"/> 越幸运这句话不适用于钉钉 app.'
        valid, err = validate_pause_tags(tagged, normed)
        assert valid, err

    def test_english_double_dash_spacing_is_tolerated(self):
        normed = "I will make it up to you--i will teach you how to be a cook."
        tagged = 'I will make it up to you-- <pause act="ins" level="2"/> i will teach you how to be a cook.'
        valid, err = validate_pause_tags(tagged, normed)
        assert valid, err


class TestComputeTargetText:
    def test_insert_keeps_transcript_unchanged(self):
        normed = "She walked slowly through the garden."
        tags = parse_pause_tags(
            'She walked slowly <pause act="ins" level="1"/> through the garden.', normed
        )
        result = compute_generated_text(normed, tags)
        assert result == normed

    def test_no_ops(self):
        normed = "Just plain text."
        assert compute_generated_text(normed, []) == normed


class TestMakeTextVersions:
    def test_single_text_version(self):
        normed = "She walked slowly through the garden."
        tags = parse_pause_tags(
            'She walked slowly <pause act="ins" level="1"/> through the garden.', normed
        )
        versions = make_text_versions(normed, tags)
        assert versions == {"text": normed}


class TestReverseOperation:
    def test_reverse_operation_flips_ins_to_red(self):
        tagged = 'She walked slowly <pause act="ins" level="2"/> through the garden.'
        assert reverse_operation(tagged) == 'She walked slowly <pause act="red" level="2"/> through the garden.'

    def test_reverse_operation_flips_red_to_ins(self):
        tagged = 'She walked slowly <pause act="red" level="2"/> through the garden.'
        assert reverse_operation(tagged) == 'She walked slowly <pause act="ins" level="2"/> through the garden.'


class TestInjectDuration:
    def test_injects_level_into_bare_insert(self):
        tagged = 'She walked slowly <pause act="ins"/> through the garden.'
        result = inject_duration(tagged, 500)
        assert '<pause act="ins" level="2"/>' in result

    def test_converts_old_delta_duration_to_level(self):
        tagged = 'She walked slowly <pause act="ins" delta_dur="800ms"/> through the garden.'
        result = inject_duration(tagged, 800)
        assert '<pause act="ins" level="3"/>' in result


class TestTagsToAlignerQueries:
    def test_basic_query(self):
        normed = "She walked slowly through the garden."
        tags = parse_pause_tags(
            'She walked slowly <pause act="ins" level="2"/> through the garden.', normed
        )
        queries = tags_to_aligner_queries(normed, tags)
        assert len(queries) == 1
        assert queries[0]["left_query"] == "slowly"
        assert queries[0]["right_query"] == "through"
        assert queries[0]["delta_dur_ms"] == PAUSE_LEVEL_TO_DURATION_MS[2]


class TestReferenceCoordinates:
    def test_three_tags_with_surrounding_spaces(self):
        normed = "One two three four."
        tag = '<pause act="ins" level="1"/>'
        operation = f"One {tag} two {tag} three {tag} four."
        assert validate_pause_tags(operation, normed) == (True, "")
        tags = parse_pause_tags(operation, normed)
        assert [item.char_idx for item in tags] == [3, 7, 13]
        assert [item["boundary_query_full"] for item in tags_to_boundary_queries(normed, tags)] == [
            "One<q></q> two three four.",
            "One two<q></q> three four.",
            "One two three<q></q> four.",
        ]

    def test_reference_whitespace_is_not_collapsed_in_coordinates(self):
        normed = "  One   two\tthree four."
        operation = 'One two <pause act="ins" level="1"/> three four.'
        assert validate_pause_tags(operation, normed) == (True, "")
        tags = parse_pause_tags(operation, normed)
        assert tags[0].char_idx == len("  One   two")
        assert tags_to_boundary_queries(normed, tags)[0]["boundary_query_full"] == (
            "  One   two<q></q>\tthree four."
        )

    def test_cjk_and_punctuation_spacing_maps_to_reference(self):
        normed = "我们今天一起出发，明天回来。"
        operation = (
            '我们 今天 <pause act="ins" level="1"/>一起出发 ， '
            '<pause act="ins" level="2"/> 明天回来。'
        )
        assert validate_pause_tags(operation, normed) == (True, "")
        tags = parse_pause_tags(operation, normed)
        assert [item.char_idx for item in tags] == [4, 9]
        assert [item.punctuation for item in tags] == [None, "，"]
        assert tags_to_boundary_queries(normed, tags)[0]["boundary_query_full"] == (
            "我们今天<q></q>一起出发，明天回来。"
        )

    @pytest.mark.parametrize(
        ("normed", "operation"),
        [
            ("I am a big cat.", 'I\'m a<pause act="ins" level="1"/> big cat.'),
            ("One two three.", 'one<pause act="ins" level="1"/> two three.'),
            ("We are a part of it.", 'We are apart<pause act="ins" level="1"/> of it.'),
            ("我们用 app 工作。", '我们用 a pp<pause act="ins" level="1"/> 工作。'),
        ],
    )
    def test_text_changes_fail_in_both_validation_and_parsing(self, normed, operation):
        valid, error = validate_pause_tags(operation, normed)
        assert not valid
        assert "Text mismatch" in error
        with pytest.raises(ValueError, match="Text mismatch"):
            parse_pause_tags(operation, normed)

    @pytest.mark.parametrize(
        ("normed", "operation", "reason"),
        [
            ("  One two.", ' <pause act="ins" level="1"/> One two.', "start"),
            ("One two.  ", 'One two. <pause act="ins" level="1"/> ', "end"),
            (
                "One   two.",
                'One <pause act="ins" level="1"/>  <pause act="ins" level="2"/> two.',
                "Duplicate",
            ),
            (
                "我们今天出发。",
                '我们 <pause act="ins" level="1"/> <pause act="ins" level="2"/> 今天出发。',
                "Duplicate",
            ),
            ("One two.", 'O<pause act="ins" level="1"/>ne two.', "word boundary"),
            ("We go apart.", 'We go a <pause act="ins" level="1"/> part.', "Text mismatch"),
        ],
    )
    def test_reference_mapping_preserves_boundary_rejections(self, normed, operation, reason):
        valid, error = validate_pause_tags(operation, normed)
        assert not valid
        assert reason in error


class TestPauseBoundaries:
    @pytest.mark.parametrize("text", [
        "Going whitewater rafting is too scary.",
        "Rowing a boat on Sun Moon Lake is relaxing.",
    ])
    def test_final_period_is_not_an_interior_boundary(self, text):
        boundaries = pause_boundaries(text)
        assert len(boundaries) == len(text.split()) - 1
        assert {kind for _, kind in boundaries} == {"nonpunct"}

    def test_contractions_and_punctuation_have_one_boundary_each(self):
        text = "One, two, it's one, two, you."
        boundaries = pause_boundaries(text)
        assert boundaries == [(4, "punct"), (9, "punct"), (14, "nonpunct"),
                              (19, "punct"), (24, "punct")]

    def test_punctuation_run_and_whitespace_are_not_extra_positions(self):
        assert pause_boundaries("  One,...  two.  ") == [(9, "punct")]
        assert pause_boundaries("我们，今天。") == [(1, "nonpunct"), (3, "punct"), (4, "nonpunct")]

    @pytest.mark.parametrize("text", [
        "One, two, it's one, two, you.", "我们，今天。", "One,...  two.",
    ])
    def test_enumerated_boundaries_pass_the_shared_validator(self, text):
        for index, kind in pause_boundaries(text):
            operation = text[:index] + '<pause act="ins" level="1"/>' + text[index:]
            valid, reason = validate_pause_tags(operation, text)
            assert valid, reason
            tag = parse_pause_tags(operation, text)[0]
            assert tag.char_idx == index
            assert ("punct" if tag.punctuation else "nonpunct") == kind

    @pytest.mark.parametrize("operation", [
        'It<pause act="ins" level="1"/>\'s here.',
        'It\'<pause act="ins" level="1"/>s here.',
        'One<pause act="ins" level="1"/>, two.',
        'One,<pause act="ins" level="1"/>.. two.',
    ])
    def test_tags_cannot_split_a_word_or_duplicate_punctuation_boundary(self, operation):
        assert not validate_pause_tags(operation, strip_pause_tags(operation))[0]
