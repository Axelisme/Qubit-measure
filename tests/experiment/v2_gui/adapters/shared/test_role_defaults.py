"""Tests for the notebook-aligned role default wrappers."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from zcu_tools.experiment.v2_gui.adapters.shared import (
    default_pi,
    default_pi2,
    default_qub_probe,
    default_res_probe,
    default_reset,
)
from zcu_tools.gui.adapter import ModuleRefValue


def _empty_ctx() -> MagicMock:
    """A ctx with empty md/ml — exercises the blank fallback path."""
    ctx = MagicMock()
    ctx.md.get.side_effect = lambda k, d=None: d
    ctx.md.__contains__ = lambda self, k: False
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ctx.ml = ml
    return ctx


def test_default_qub_probe_returns_blank_pulse_custom_ref():
    v = default_qub_probe(_empty_ctx())
    assert isinstance(v, ModuleRefValue)
    # blank probe is a Custom (inline) pulse, not a library reference
    assert v.chosen_key.startswith("<Custom:")


def test_default_pi_falls_back_to_blank_when_lib_empty():
    v = default_pi(_empty_ctx())
    assert isinstance(v, ModuleRefValue)
    assert v.chosen_key.startswith("<Custom:")


def test_default_pi2_falls_back_to_blank_when_lib_empty():
    v = default_pi2(_empty_ctx())
    assert isinstance(v, ModuleRefValue)
    assert v.chosen_key.startswith("<Custom:")


def test_default_res_probe_returns_module_ref():
    v = default_res_probe(_empty_ctx())
    assert isinstance(v, ModuleRefValue)


def test_default_reset_optional_returns_none_when_no_lib_entry():
    v = default_reset(_empty_ctx(), optional=True)
    assert v is None


def test_default_reset_non_optional_returns_blank():
    v = default_reset(_empty_ctx())
    assert isinstance(v, ModuleRefValue)


def test_role_wrappers_compose_with_value_with_field():
    # role wrapper + value OO override chains cleanly
    v = cast(Any, default_qub_probe(_empty_ctx())).with_field("gain", 0.3)
    assert isinstance(v, ModuleRefValue)
    assert cast(Any, v.value.fields["gain"]).value == 0.3
