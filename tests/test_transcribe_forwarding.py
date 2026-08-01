#!/usr/bin/env python3
"""
Guards the thin-wrapper contract of `transcribe()`.

`transcribe()` is the public library API; it does no work of its own and forwards
everything to `_transcribe_with_sentences()`. A parameter declared on the wrapper
but omitted from that forwarding call fails *silently* — Python accepts the keyword,
the inner default wins, and the caller gets a normal result with the requested
behavior missing. `dump_merge_metadata` was dropped this way until v0.11.1.
"""

import inspect

import pytest

import podscripter
from podscripter import _transcribe_with_sentences, transcribe

pytestmark = pytest.mark.core

# Forwarded positionally in the call, so they never appear in the captured kwargs.
PASSED_POSITIONALLY = {"media_file", "output_dir", "language", "output_format", "single_call"}


@pytest.fixture
def forwarded_kwargs(monkeypatch):
    """Capture the kwargs `transcribe()` forwards, without running a transcription."""
    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        raise _Stop()

    monkeypatch.setattr(podscripter, "_transcribe_with_sentences", spy)
    with pytest.raises(_Stop):
        transcribe("audio.mp3", output_dir="out", dump_merge_metadata=True)
    return captured


class _Stop(Exception):
    """Aborts the wrapper once the forwarding call is reached."""


def test_dump_merge_metadata_is_forwarded(forwarded_kwargs):
    assert forwarded_kwargs.get("dump_merge_metadata") is True, (
        "transcribe(dump_merge_metadata=True) must reach _transcribe_with_sentences; "
        f"got {forwarded_kwargs.get('dump_merge_metadata', '<missing>')!r}"
    )


def test_every_wrapper_parameter_is_forwarded(forwarded_kwargs):
    expected = set(inspect.signature(transcribe).parameters) - PASSED_POSITIONALLY
    missing = expected - set(forwarded_kwargs)
    assert not missing, (
        f"transcribe() declares {sorted(missing)} but never forwards them, so callers "
        "are silently ignored. Add them to the _transcribe_with_sentences(...) call."
    )


def test_forwarded_parameters_all_exist_on_the_inner_function():
    inner = set(inspect.signature(_transcribe_with_sentences).parameters)
    wrapper = set(inspect.signature(transcribe).parameters)
    unknown = wrapper - inner
    assert not unknown, f"transcribe() declares parameters the inner function cannot accept: {sorted(unknown)}"


def test_every_wrapper_parameter_is_documented():
    doc = transcribe.__doc__ or ""
    undocumented = [
        name for name in inspect.signature(transcribe).parameters
        if name != "media_file" and f"{name}:" not in doc
    ]
    assert not undocumented, f"transcribe() parameters missing from the Args docstring: {undocumented}"
