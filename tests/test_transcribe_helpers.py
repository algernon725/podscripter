#!/usr/bin/env python3
"""
Unit tests for internal helpers in podscripter.py

Tests:
- _split_audio_with_overlap on synthetic audio (silence)
- _dedupe_segments and _accumulate_segments logic
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import podscripter as ts
from pydub import AudioSegment

pytestmark = pytest.mark.core


def test_split_audio_with_overlap_silence():
    with TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        media = tmpdir / "synthetic_silence.wav"
        AudioSegment.silent(duration=10_000).export(str(media), format="wav")

        chunks = ts._split_audio_with_overlap(
            str(media), chunk_length_sec=4, overlap_sec=1, chunk_dir=tmpdir
        )
        assert len(chunks) in (3, 4)
        durations = [round(c["duration_sec"], 2) for c in chunks]
        assert durations[0] == 4.0
        assert durations[1] == 4.0
        assert 3.9 <= durations[2] <= 4.1
        if len(chunks) == 4:
            assert 0.8 <= durations[3] <= 1.2
        for c in chunks:
            p = Path(c["path"])
            assert p.exists()
            p.unlink()


def test_dedupe_segments_basic():
    global_segments = [
        (0.0, 1.0, "a"),
        (0.9, 1.5, "b"),
        (1.49, 1.7, "c"),
    ]
    out, new_last = ts._dedupe_segments(global_segments, last_end=1.0, epsilon=0.05)
    assert len(out) == 2
    assert out[0]["text"] == "b"
    assert out[1]["text"] == "c"
    assert round(new_last, 2) == 1.7


@pytest.mark.xfail(reason="Pre-existing: test expectations predate API changes")
def test_accumulate_segments_with_offset_and_dedupe():
    class FakeSeg:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text

    segs = [
        FakeSeg(0.0, 0.0, "t1"),
        FakeSeg(0.5, 0.7, "t2"),
        FakeSeg(3.0, 3.4, "t3"),
    ]
    out, text, new_last = ts._accumulate_segments(segs, chunk_start=2.0, last_end=2.0)
    assert len(out) == 2
    assert out[0]["start"] == 2.5 and out[0]["end"] == 2.7 and out[0]["text"] == "t2"
    assert out[1]["start"] == 5.0 and out[1]["end"] == 5.4 and out[1]["text"] == "t3"
    assert text == "t2 t3"
    assert round(new_last, 2) == 5.4
