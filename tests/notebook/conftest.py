"""Isolate the process-wide style applied by Notebook entry points."""

from collections.abc import Iterator

import matplotlib
import pytest


@pytest.fixture(autouse=True)
def restore_rc_params() -> Iterator[None]:
    """Restore caller rcParams after tests construct a Notebook host."""
    with matplotlib.rc_context():
        yield
