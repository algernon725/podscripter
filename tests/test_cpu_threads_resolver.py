#!/usr/bin/env python3

import argparse
import logging
import os

import pytest

from podscripter import _detect_cpu_count, _positive_int, _resolve_cpu_threads

pytestmark = pytest.mark.core


def pin_affinity(monkeypatch, cpus):
    """Force _detect_cpu_count() to see a specific set of usable CPUs."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(cpus), raising=False)


# --- precedence: CLI > OMP_NUM_THREADS > detection ---------------------------

def test_cli_value_beats_env_and_detection(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    pin_affinity(monkeypatch, range(8))
    assert _resolve_cpu_threads(3) == 3, "explicit --cpu-threads must win over env and detection"


def test_env_used_when_no_cli_value(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    pin_affinity(monkeypatch, range(8))
    assert _resolve_cpu_threads(None) == 2, "OMP_NUM_THREADS must be honored, not overwritten"


def test_detection_used_when_env_absent(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    pin_affinity(monkeypatch, {0, 1, 2, 3})
    assert _resolve_cpu_threads(None) == 4, "must fall back to the detected core count"


def test_env_whitespace_tolerated(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "  6  ")
    pin_affinity(monkeypatch, range(8))
    assert _resolve_cpu_threads(None) == 6, "surrounding whitespace must be stripped"


def test_empty_env_falls_through_without_warning(monkeypatch, caplog):
    monkeypatch.setenv("OMP_NUM_THREADS", "   ")
    pin_affinity(monkeypatch, {0, 1, 2})
    with caplog.at_level(logging.WARNING, logger="podscripter"):
        assert _resolve_cpu_threads(None) == 3
    assert "OMP_NUM_THREADS" not in caplog.text, "an empty value is not a stated intent; no warning expected"


# --- invalid OMP_NUM_THREADS: warn and ignore -------------------------------

@pytest.mark.parametrize("env_value", ["abc", "8.5", "0", "-4"])
def test_invalid_env_falls_back_to_detection(monkeypatch, env_value):
    monkeypatch.setenv("OMP_NUM_THREADS", env_value)
    pin_affinity(monkeypatch, {0, 1, 2})
    assert _resolve_cpu_threads(None) == 3, f"invalid OMP_NUM_THREADS={env_value!r} must be ignored, not fatal"


def test_invalid_env_warns(monkeypatch, caplog):
    monkeypatch.setenv("OMP_NUM_THREADS", "abc")
    pin_affinity(monkeypatch, {0, 1, 2})
    with caplog.at_level(logging.WARNING, logger="podscripter"):
        _resolve_cpu_threads(None)
    assert "Ignoring invalid OMP_NUM_THREADS" in caplog.text, f"expected a warning, got: {caplog.text!r}"


# --- detection ---------------------------------------------------------------

def test_detect_uses_affinity(monkeypatch):
    pin_affinity(monkeypatch, {0, 1, 2})
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert _detect_cpu_count() == 3, "affinity must take precedence over os.cpu_count()"


def test_detect_falls_back_when_affinity_unavailable(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert _detect_cpu_count() == 6, "must fall back to os.cpu_count() on platforms without sched_getaffinity"


def test_detect_falls_back_when_affinity_raises(monkeypatch):
    def boom(_pid):
        raise OSError("no affinity for you")

    monkeypatch.setattr(os, "sched_getaffinity", boom, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert _detect_cpu_count() == 6, "an OSError from sched_getaffinity must fall back, not propagate"


def test_detect_floors_at_one_when_cpu_count_unknown(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert _detect_cpu_count() == 1, "os.cpu_count() returning None must floor at 1"


def test_detect_floors_at_one_on_empty_affinity(monkeypatch):
    pin_affinity(monkeypatch, set())
    assert _detect_cpu_count() == 1, "must never return 0 threads"


# --- library callers bypass argparse and get clamped ------------------------

@pytest.mark.parametrize("cli_value", [0, -2])
def test_library_values_clamped_to_one(monkeypatch, cli_value):
    # Env set to prove the CLI branch short-circuits before the env is read.
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    assert _resolve_cpu_threads(cli_value) == 1, "non-positive library values must clamp to 1, not fall through to env"


# --- argparse type function --------------------------------------------------

def test_positive_int_accepts_positive():
    assert _positive_int("4") == 4


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "3.5", ""])
def test_positive_int_rejects(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int(bad)
