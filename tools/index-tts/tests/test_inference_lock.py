import threading

import api_service as synthesis_service


def test_all_model_endpoints_use_the_global_inference_lock():
    assert synthesis_service.synthesize.uses_global_inference_lock is True
    assert synthesis_service.extract_tokens.uses_global_inference_lock is True
    assert synthesis_service.synthesize_from_tokens.uses_global_inference_lock is True


def test_global_inference_lock_serializes_requests_and_releases_after_failure():
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    failures = []

    @synthesis_service.serialized_model_request
    def failing_request():
        first_entered.set()
        assert release_first.wait(timeout=2)
        raise RuntimeError("expected failure")

    @synthesis_service.serialized_model_request
    def following_request():
        second_entered.set()

    def run_failing_request():
        try:
            failing_request()
        except Exception as exc:  # noqa: BLE001 - asserted below
            failures.append(exc)

    first_thread = threading.Thread(target=run_failing_request)
    second_thread = threading.Thread(target=following_request)
    first_thread.start()
    assert first_entered.wait(timeout=2)
    second_thread.start()

    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_entered.is_set()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)


def test_global_inference_lock_is_reentrant_for_existing_model_helpers():
    entered = False

    @synthesis_service.serialized_model_request
    def request_with_nested_model_lock():
        nonlocal entered
        with synthesis_service.INFERENCE_LOCK:
            entered = True

    request_with_nested_model_lock()
    assert entered is True
