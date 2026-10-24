"""Post-analysis defaults on a non-opt-in adapter.

An adapter that does NOT declare ``post_analysis`` reports the capability as
False and its inherited base ``post_analyze`` / ``post_analyze_params_cls`` /
``get_post_analyze_params`` fast-fail (Fast-Fail guard against routing
post-analysis to an adapter that never implemented it).
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.base import Experiment
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.plotting.plots import NonPresentingHost, Plots


class _FakeAdapter(BaseAdapter[Any, Any, Any, Any]):
    """Minimal no-analysis adapter that does not opt into post-analysis."""

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE
    )
    exp_cls = Experiment


def test_default_capability_has_no_post_analysis() -> None:
    assert AdapterCapabilities().post_analysis is False
    assert _FakeAdapter.capabilities.post_analysis is False


def test_base_post_analyze_raises() -> None:
    with pytest.raises(NotImplementedError, match="post-analysis"):
        _FakeAdapter().post_analyze(MagicMock(), plots=Plots(NonPresentingHost()))
