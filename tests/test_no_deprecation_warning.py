"""
Test to verify no deprecation warnings occur.
"""

import warnings

import pytest

pytestmark = pytest.mark.core


def test_no_deprecation_warning():
    """Test that no deprecation warnings occur when importing transformers."""

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            import transformers  # noqa: F401
        except ImportError:
            pytest.skip("transformers not installed")

        deprecation_warnings = [
            warning for warning in w
            if 'deprecated' in str(warning.message).lower()
        ]
        assert len(deprecation_warnings) == 0, (
            f"Found deprecation warnings: {[str(x.message) for x in deprecation_warnings]}"
        )

        transformers_cache_warnings = [
            warning for warning in w
            if 'TRANSFORMERS_CACHE' in str(warning.message)
        ]
        assert len(transformers_cache_warnings) == 0, (
            f"Found TRANSFORMERS_CACHE warnings: {[str(x.message) for x in transformers_cache_warnings]}"
        )
