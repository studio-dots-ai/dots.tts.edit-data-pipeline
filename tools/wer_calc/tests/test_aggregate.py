from pathlib import Path
import sys

from fastapi.testclient import TestClient


WER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WER_ROOT))

from api_service import app  # noqa: E402
from wer_calc import aggregate_error_rate_results, compute_error_rate  # noqa: E402


def test_aggregate_error_rate_uses_total_reference_length_not_utterance_mean():
    short_bad = compute_error_rate("one", "two", "en")
    long_good = compute_error_rate("a b c d e f g h i", "a b c d e f g h i", "en")

    aggregate = aggregate_error_rate_results([short_bad, long_good]).to_dict()

    assert aggregate["error_rate"] == 1 / 10
    assert aggregate["ref_length"] == 10
    assert aggregate["valid_count"] == 2
    assert aggregate["metric"] == "wer"


def test_aggregate_error_rate_reports_mixed_metric():
    english = compute_error_rate("hello world", "hello", "en")
    chinese = compute_error_rate("你好世界", "你好", "zh")

    aggregate = aggregate_error_rate_results([english, chinese]).to_dict()

    assert aggregate["metric"] == "mixed"
    assert aggregate["error_rate"] == (1 + 2) / (2 + 4)


def test_auto_uses_mixed_language_units():
    result = compute_error_rate("一个apple好吃", "一个orange好吃", "auto")

    assert result.metric == "mixed"
    assert result.ref_length == 5
    assert result.hyp_length == 5
    assert result.substitutions == 1


def test_service_auto_uses_mixed_language_units():
    client = TestClient(app)
    response = client.post(
        "/compute_wer",
        json={
            "reference": "一个apple好吃",
            "hypothesis": "一个orange好吃",
            "language": "auto",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metric"] == "mixed"
    assert data["ref_length"] == 5


def test_aggregate_error_rate_empty_reference_with_insertion_has_null_rate():
    result = compute_error_rate("", "extra", "en")

    aggregate = aggregate_error_rate_results([result]).to_dict()

    assert aggregate["error_rate"] is None
    assert aggregate["insertions"] == 1
    assert aggregate["ref_length"] == 0


def test_aggregate_error_rate_all_missing_has_null_rate():
    aggregate = aggregate_error_rate_results([None, {"error_rate": 0.5}]).to_dict()

    assert aggregate["error_rate"] is None
    assert aggregate["valid_count"] == 0
    assert aggregate["skipped_count"] == 2


def test_service_aggregate_endpoint_and_batch_include_aggregate():
    client = TestClient(app)
    payload = {
        "pairs": [
            {"reference": "one", "hypothesis": "two", "language": "en"},
            {
                "reference": "a b c d e f g h i",
                "hypothesis": "a b c d e f g h i",
                "language": "en",
            },
        ],
        "include_aggregate": True,
    }

    batch_response = client.post("/compute_wer_batch", json=payload)
    assert batch_response.status_code == 200
    batch_data = batch_response.json()
    assert batch_data["aggregate"]["error_rate"] == 1 / 10

    aggregate_response = client.post(
        "/aggregate_wer_results",
        json={"results": batch_data["results"]},
    )
    assert aggregate_response.status_code == 200
    assert aggregate_response.json()["error_rate"] == 1 / 10
