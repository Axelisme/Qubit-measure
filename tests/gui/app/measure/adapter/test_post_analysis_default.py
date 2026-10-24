"""Unrouted post-analysis fails without a declared second stage."""

from unittest.mock import MagicMock

import pytest
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._native_save_support import SavingAdapter


def test_base_post_analyze_raises() -> None:
    with pytest.raises(NotImplementedError, match="post-analysis"):
        SavingAdapter.post_analyze(MagicMock(), plots=Plots(NonPresentingHost()))
