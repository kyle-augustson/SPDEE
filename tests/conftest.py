"""Project-wide pytest configuration.

Defines markers so users can opt out of slow research-grade tests
during routine iteration:

    # Default (excludes slow tests).
    python -m pytest tests/

    # All tests including slow ones.
    python -m pytest tests/ -m "slow or not slow"

    # Only slow tests.
    python -m pytest tests/ -m slow

Markers:

* ``slow``: takes more than ~1 min in the sandbox. Used for
  resolution-scaling sweeps and other research-grade validation
  rather than routine regressions. Exists so the default
  suite stays fast (~4 min) for everyday iteration.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that take >1 min "
        "(deselect with '-m \"not slow\"')",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked ``slow`` unless the user opts in.

    Behavior matches the file-level exclusion already used for the
    Zhang 2001 study (`test_evp_zhang_2001.py`): excluded from the
    default run, available via explicit opt-in.

    Opt-in is via ``-m``. If the user passes ``-m slow`` or
    ``-m "slow or not slow"`` the deselection here is skipped.
    """
    marker_expr = config.getoption("-m")
    if marker_expr:
        # User explicitly requested marker filtering — don't second-
        # guess them.
        return
    skip_slow = pytest.mark.skip(
        reason="slow test; run with -m \"slow or not slow\" to include"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
