"""Post-analysis defaults on a non-opt-in adapter.

An adapter that does NOT declare ``post_analysis`` reports the capability as
False and its inherited base ``post_analyze`` / ``post_analyze_params_cls`` /
``get_post_analyze_params`` fast-fail (Fast-Fail guard against routing
post-analysis to an adapter that never implemented it).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    CfgSectionSpec,
    CfgSectionValue,
)


class _FakeAdapter(BaseAdapter[Any, Any, Any, Any]):
    """Minimal FIT-only adapter that does not opt into post-analysis."""

    exp_cls = MagicMock

    @classmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        return CfgSectionSpec()

    def make_default_value(self, ctx: Any) -> CfgSectionValue:
        del ctx
        return CfgSectionValue()

    def make_filename_stem(self, ctx: Any) -> str:
        del ctx
        return "fake"


def test_default_capability_has_no_post_analysis() -> None:
    assert AdapterCapabilities().post_analysis is False
    assert _FakeAdapter.capabilities.post_analysis is False


def test_base_post_analyze_raises() -> None:
    with pytest.raises(NotImplementedError, match="post-analysis"):
        _FakeAdapter().post_analyze(MagicMock())


def test_base_get_post_analyze_params_raises() -> None:
    with pytest.raises(NotImplementedError, match="post-analysis"):
        _FakeAdapter().get_post_analyze_params(MagicMock(), MagicMock())


def test_base_post_analyze_params_cls_reflects_base_return() -> None:
    # The base ``get_post_analyze_params`` is annotated ``-> Any``; reflection
    # surfaces that. A non-opt-in adapter is never routed to post-analysis, so
    # this reflected type is only a Fast-Fail-adjacent default, not load-bearing.
    assert _FakeAdapter.post_analyze_params_cls() is Any
