
import pytest

from speech_edit.data.text_markup import (
    DEL_TAG,
    INS_TAG,
    SUB_TAG,
    edit_segments,
    op_counts,
    render_source,
    render_target,
    render_target_with_segment_spans,
    reverse_tagged_instruction,
    validate_tagged_instruction,
)


CASES = [
    ("hello <sub targ=\"kind\">brave</sub> world", "hello brave world", "hello kind world"),
    ("<del>hello </del>world", "hello world", "world"),
    ("hello <ins>wide </ins>world", "hello world", "hello wide world"),
    ("我喜欢<sub targ=\"香蕉\">苹果</sub>", "我喜欢苹果", "我喜欢香蕉"),
    ("请<del>不要</del>讲话", "请不要讲话", "请讲话"),
    ("请<ins>认真</ins>讲话", "请讲话", "请认真讲话"),
    ("mix <sub targ=\"中文\">English</sub> here", "mix English here", "mix 中文 here"),
    ("mix <del>中文</del> text", "mix 中文 text", "mix text"),
    ("mix <ins>中文 </ins>text", "mix text", "mix 中文 text"),
    ("a <sub targ=\"x\">one</sub> b <del>two</del> c", "a one b two c", "a x b c"),
    ("a <sub targ=\"x\">one</sub> b <ins>new </ins>c", "a one b c", "a x b new c"),
    ("a <del>one</del> b <ins>two</ins>", "a one b ", "a b two"),
    ("<ins>Start </ins>middle", "middle", "Start middle"),
    ("middle<ins> end</ins>", "middle", "middle end"),
    ("say <sub targ=\"&quot;yes&quot;\">&quot;no&quot;</sub>", 'say "no"', 'say "yes"'),
    ("a <sub targ=\"&lt;b&gt;\">&amp;</sub>", "a &", "a <b>"),
]


@pytest.mark.parametrize("st,source,target", CASES)
def test_tag_contract_and_reverse(st, source, target):
    validate_tagged_instruction(st, source, target)
    assert render_source(st) == source
    assert render_target(st) == target
    reverse_st = reverse_tagged_instruction(st)
    assert render_source(reverse_st) == target
    forward_edits = edit_segments(st)
    reverse_edits = edit_segments(reverse_st)
    assert len(reverse_edits) == len(forward_edits)
    for forward, reverse in zip(forward_edits, reverse_edits):
        if forward.kind == DEL_TAG:
            assert reverse.kind == INS_TAG
            assert reverse.source == ""
            assert reverse.target == forward.source
        elif forward.kind == INS_TAG:
            assert reverse.kind == DEL_TAG
            assert forward.target.strip() in reverse.source
            assert reverse.target == ""
        else:
            assert forward.kind == SUB_TAG
            assert reverse.kind == SUB_TAG
            assert forward.target.strip() in reverse.source
            assert reverse.target == forward.source


def test_render_target_normalizes_edit_boundaries():
    assert render_target("No, <ins> actually</ins>it's not fine.") == (
        "No, actually it's not fine."
    )
    assert render_target("hello<ins>wide</ins>world") == "hello wide world"
    assert render_target("请<ins>认真</ins>讲话") == "请认真讲话"
    assert render_target("hello <ins>wide</ins> , world") == "hello wide, world"
    assert render_target("I can<ins>'t</ins> go") == "I can't go"
    assert render_target("state<ins>-of-the-art</ins> model") == (
        "state-of-the-art model"
    )
    assert render_target("don<ins>'</ins>t stop") == "don't stop"
    assert render_target("pre<ins>-</ins>trained model") == "pre-trained model"
    assert render_target("students<ins>'</ins> books") == "students' books"
    assert render_target("score 3 - 2 <ins>today</ins>") == "score 3 - 2 today"
    assert render_target("say <ins>'</ins> hello") == "say ' hello"


def test_render_target_maps_segments_after_boundary_normalization():
    rendered, spans = render_target_with_segment_spans(
        "the first<ins>few</ins> words"
    )
    assert rendered == "the first few words"
    assert spans[1] == (10, 13)

    rendered, spans = render_target_with_segment_spans("don<ins>'</ins>t stop")
    assert rendered == "don't stop"
    assert spans[1] == (3, 4)


def test_segments_have_character_spans_without_text_search():
    st = "go go <sub targ=\"went\">go</sub> go"
    segments = edit_segments(st)

    assert len(segments) == 1
    assert segments[0].source_start == len("go go ")
    assert segments[0].source_end == len("go go go")
    assert segments[0].target_start == len("go go ")
    assert segments[0].target_end == len("go go went")


def test_op_counts_and_nl_instructions():
    st = "<ins>please </ins>a <sub targ=\"b\">c</sub> <del>d</del>"

    assert op_counts(st) == {"del": 1, "ins": 1, "sub": 1}
