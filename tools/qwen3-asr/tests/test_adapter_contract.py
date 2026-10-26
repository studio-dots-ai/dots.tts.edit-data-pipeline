"""Verify that the aligner adapter preserves official multilingual tokenization."""


def test_processor_preserves_official_multilingual_tokenization():
    from qwen3_asr_service.aligner import Qwen3ForceAlignProcessor, OfficialProcessor
    adapter, official = Qwen3ForceAlignProcessor(), OfficialProcessor()
    for text, language in [('Hello, world!', 'English'), ('你好世界', 'Chinese'), ('こんにちは世界', 'Japanese'), ('안녕하세요 세계', 'Korean')]:
        assert adapter.encode_timestamp(text, language) == official.encode_timestamp(text, language)
