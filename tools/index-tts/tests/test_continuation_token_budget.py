from types import SimpleNamespace

import pytest

from adapter import IndexTTS2


def _fake_tts(pos_rows: int = 1818, gpt_max_mel_tokens: int = 1815) -> IndexTTS2:
    tts = IndexTTS2.__new__(IndexTTS2)
    tts.gpt = SimpleNamespace(
        mel_pos_embedding=SimpleNamespace(
            emb=SimpleNamespace(num_embeddings=pos_rows)
        ),
        max_mel_tokens=gpt_max_mel_tokens,
    )
    tts.cfg = SimpleNamespace(gpt=SimpleNamespace(max_mel_tokens=gpt_max_mel_tokens))
    return tts


def test_continuation_budget_accounts_for_forward_padding():
    tts = _fake_tts(pos_rows=1818)

    assert tts._max_vocode_code_tokens() == 1816
    with pytest.warns(RuntimeWarning, match="Reducing max_mel_tokens"):
        capped = tts._cap_generation_tokens_for_vocode(
            requested_max_mel_tokens=1500,
            num_prefix_tokens=1403,
        )

    assert capped == 413


def test_continuation_budget_rejects_prefix_with_no_room():
    tts = _fake_tts(pos_rows=1818)

    with pytest.raises(ValueError, match="prefix_tokens=1816"):
        tts._cap_generation_tokens_for_vocode(
            requested_max_mel_tokens=1500,
            num_prefix_tokens=1816,
        )


def test_continuation_text_token_guard_rejects_overlong_full_text():
    tts = _fake_tts()

    with pytest.raises(ValueError, match="601 BPE tokens > max_text_tokens=600"):
        tts._validate_continuation_text_tokens(601)
