#!/usr/bin/env python3

from podscripter import _normalize_srt_cues


def assert_float_close(a, b, tol=1e-6):
    if abs(a - b) > tol:
        raise AssertionError(f"Expected {b} but got {a}")


def test_trims_to_next_start_minus_gap():
    segs = [
        {"start": 0.0, "end": 10.0, "text": "Hello"},
        {"start": 3.0, "end": 4.0, "text": "world"},
    ]
    out = _normalize_srt_cues(segs, max_duration=6.0, min_gap=0.2, min_duration=1.0)
    # First end should be trimmed to 2.8 (3.0 - 0.2)
    assert_float_close(out[0]["end"], 2.8)
    # Second remains within original bounds (max_duration not hit)
    assert_float_close(out[1]["start"], 3.0)
    assert_float_close(out[1]["end"], 4.0)


def test_max_duration_clamp():
    segs = [{"start": 0.0, "end": 20.0, "text": "Long"}]
    out = _normalize_srt_cues(segs, max_duration=6.0, min_gap=0.2, min_duration=1.0)
    assert_float_close(out[0]["end"], 6.0)


def test_min_duration_enforced():
    segs = [{"start": 1.0, "end": 1.0, "text": "Zero length"}]
    out = _normalize_srt_cues(segs, max_duration=6.0, min_gap=0.2, min_duration=1.0)
    assert_float_close(out[0]["end"], 2.0)  # start + min_duration


