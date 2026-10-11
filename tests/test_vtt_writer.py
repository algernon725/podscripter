#!/usr/bin/env python3
"""
WebVTT output (`--output_format vtt`) and the cue timestamp format it shares with SRT.

VTT and SRT carry the same normalized cues; only the file syntax differs. These tests
pin the syntax a browser `<track>` element needs: the `WEBVTT` header, `.` before the
milliseconds, and escaped markup characters in cue text.
"""

import re

import pytest

from podscripter import (
    InvalidInputError,
    _format_cue_timestamp,
    _transcribe_with_sentences,
    _write_srt,
    _write_vtt,
)

pytestmark = pytest.mark.core

VTT_TIMING = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$")


def _write_and_read(writer, segments, tmp_path, suffix):
    out = tmp_path / f"out.{suffix}"
    writer(segments, str(out))
    return out.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "seconds, sep, expected",
    [
        (0.0, ".", "00:00:00.000"),
        (2.3, ".", "00:00:02.300"),  # truncation would give .299
        (2.3, ",", "00:00:02,300"),
        (3725.5, ".", "01:02:05.500"),
        (59.9996, ".", "00:01:00.000"),  # rounding carries into the next minute
    ],
)
def test_format_cue_timestamp(seconds, sep, expected):
    assert _format_cue_timestamp(seconds, sep) == expected, f"{seconds}s with {sep!r}"


def test_vtt_header_and_unnumbered_cues(tmp_path):
    segs = [
        {"start": 0.0, "end": 1.5, "text": " Hello there. "},
        {"start": 2.0, "end": 3.25, "text": "How are you?"},
    ]
    content = _write_and_read(_write_vtt, segs, tmp_path, "vtt")
    assert content.startswith("WEBVTT\n\n"), "VTT must start with the WEBVTT header and a blank line"
    blocks = content.removeprefix("WEBVTT\n\n").strip().split("\n\n")
    assert blocks == [
        "00:00:00.000 --> 00:00:01.500\nHello there.",
        "00:00:02.000 --> 00:00:03.250\nHow are you?",
    ], f"Unexpected cue blocks: {blocks}"
    for block in blocks:
        assert VTT_TIMING.match(block.splitlines()[0]), f"Bad timing line in {block!r}"


@pytest.mark.parametrize(
    "writer, suffix, first_block",
    [
        (_write_srt, "srt", "1\n00:00:01,000 --> 00:00:02,300\nHi."),
        (_write_vtt, "vtt", "00:00:01.000 --> 00:00:02.300\nHi."),
    ],
)
def test_srt_and_vtt_syntax_differ_only_where_expected(tmp_path, writer, suffix, first_block):
    segs = [{"start": 1.0, "end": 2.3, "text": "Hi."}]
    content = _write_and_read(writer, segs, tmp_path, suffix)
    assert first_block in content, f"{suffix}: expected {first_block!r} in {content!r}"


def test_vtt_escapes_markup_characters(tmp_path):
    segs = [{"start": 0.0, "end": 2.0, "text": "AT&T and R&D <grin> a --> b"}]
    content = _write_and_read(_write_vtt, segs, tmp_path, "vtt")
    cue_text = content.strip().split("\n")[-1]
    assert cue_text == "AT&amp;T and R&amp;D &lt;grin&gt; a --&gt; b", f"Got {cue_text!r}"
    assert "-->" not in cue_text, "'-->' in cue text would be parsed as a timing line"


def test_vtt_collapses_blank_lines_inside_cue_text(tmp_path):
    segs = [{"start": 0.0, "end": 2.0, "text": "first line\n\nsecond line"}]
    content = _write_and_read(_write_vtt, segs, tmp_path, "vtt")
    assert content.endswith("00:00:00.000 --> 00:00:02.000\nfirst line\nsecond line\n\n"), (
        f"A blank line would end the cue early: {content!r}"
    )


def test_vtt_skips_empty_cues(tmp_path):
    segs = [
        {"start": 0.0, "end": 1.0, "text": "   "},
        {"start": 1.0, "end": 2.0, "text": "Kept."},
    ]
    content = _write_and_read(_write_vtt, segs, tmp_path, "vtt")
    assert content.count("-->") == 1, f"Empty cue should be skipped: {content!r}"


@pytest.mark.parametrize("text", ["¿Qué pasó? ¡Ay, señor!", "Ça va, très bien."])
@pytest.mark.parametrize("writer, suffix", [(_write_vtt, "vtt"), (_write_srt, "srt")])
def test_subtitle_writers_are_utf8(tmp_path, writer, suffix, text):
    segs = [{"start": 0.0, "end": 2.0, "text": text}]
    content = _write_and_read(writer, segs, tmp_path, suffix)
    assert text in content, f"{suffix}: non-ASCII text did not round-trip as UTF-8"


def test_unknown_output_format_is_rejected(tmp_path):
    media = tmp_path / "audio.mp3"
    media.write_bytes(b"")  # exists, so the format check is what fails
    with pytest.raises(InvalidInputError, match="json"):
        _transcribe_with_sentences(str(media), None, "en", "json", write_output=False)
