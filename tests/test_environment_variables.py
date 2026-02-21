"""
Test to verify environment variables are set correctly.
"""

import os

import pytest

pytestmark = pytest.mark.core


def test_environment_variables():
    """Test that deprecated environment variables are not set."""

    deprecated_vars = ['TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE']
    for var in deprecated_vars:
        assert var not in os.environ, f"Deprecated variable {var} is still set"
