#!/usr/bin/env python3

import pytest

from podscripter import _dedupe_segments, _accumulate_segments

pytestmark = pytest.mark.core


class Seg:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


def assert_eq(a, b):
    if a != b:
        raise AssertionError(f"Expected {b!r} but got {a!r}")


def test_dedupe_segments_overlap():
    global_segments = [
        (0.0, 1.0, "a"),
        (0.9, 1.2, "b"),
        (1.3, 2.0, "c"),
    ]
    out, new_last = _dedupe_segments(
        [(s, e, t) for s, e, t in global_segments], last_end=0.95, epsilon=0.05
    )
    texts = [x["text"] for x in out]
    assert "c" in texts, f"Expected 'c' in deduped output, got {texts}"
    assert round(new_last, 3) == 2.0


def test_accumulate_segments_with_offset():
    local = [Seg(0.0, 1.0, "hello"), Seg(1.0, 2.0, "world")]
    chunk_start = 10.0
    deduped, text, new_last = _accumulate_segments(local, chunk_start, last_end=9.5, epsilon=0.05)
    assert_eq([round(d["start"], 2) for d in deduped], [10.0, 11.0])
    assert_eq([round(d["end"], 2) for d in deduped], [11.0, 12.0])
    assert_eq(text, "hello\nworld")
    assert_eq(round(new_last, 2), 12.0)


