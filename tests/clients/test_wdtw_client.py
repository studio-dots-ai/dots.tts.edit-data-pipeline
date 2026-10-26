from speech_edit.clients.services import WDTWClient


class _Response:
    status_code = 200
    text = "{}"

    def json(self):
        return {
            "wdtw": 0.0,
            "distance": 0.0,
            "normalizer": 1.0,
            "mode": "preserved",
            "status": "ok",
        }


def test_wdtw_client_sends_current_source_target_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("speech_edit.clients.services.requests.post", fake_post)

    result = WDTWClient("http://wdtw").compute(
        source_audio=b"RIFFsource",
        target_audio=b"RIFFtarget",
        source_transcript="hello",
        target_transcript="hello there",
        preservation_st_inst='hello <ins>there</ins>',
        source_alignments=[{"word": "hello", "start": 0.0, "end": 0.4}],
        target_alignments=[{"word": "hello", "start": 0.0, "end": 0.4}],
        language="eng",
        timeout=123,
    )

    assert result["status"] == "ok"
    assert captured["url"] == "http://wdtw/compute_wdtw"
    assert captured["timeout"] == 123
    payload = captured["json"]
    assert payload["source_transcript"] == "hello"
    assert payload["target_transcript"] == "hello there"
    assert payload["preservation_st_inst"] == 'hello <ins>there</ins>'
    assert payload["source_audio"]["format"] == "wav"
    assert payload["target_audio"]["format"] == "wav"
    assert payload["source_alignments"] == [{"word": "hello", "start": 0.0, "end": 0.4}]
    assert payload["target_alignments"] == [{"word": "hello", "start": 0.0, "end": 0.4}]
    assert "aligner_url" not in payload
    assert "aligner_backend" not in payload
    for legacy_key in (
        "original_audio",
        "edited_audio",
        "original_transcript",
        "original_alignments",
        "edited_alignments",
    ):
        assert legacy_key not in payload
