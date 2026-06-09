"""Controller setup tests — SetupRequest routing, predictor / flux fallbacks.

The mock path is exercised end-to-end; the real path's helpers are unit-tested
in isolation (predictor degradation, flux-address validation) without touching
the network — ``make_soc_proxy`` is never called here. Setup writes the soc /
predictor into the active ``exp_context`` (the session SSOT) and the run reads
it back.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.state import SetupRequest
from zcu_tools.gui.app.autofluxdep.tools import SimplePredictor

# --- setup routing: mock request + legacy use_mock shorthand ---


def test_setup_mock_request_connects_mock_soc():
    ctrl = build_core()
    ctrl.setup(SetupRequest(use_mock=True))
    ctx = ctrl.state.exp_context
    assert ctx.soc is not None
    assert "MockQickSoc" in type(ctx.soc).__name__
    # the mock path leaves the raw predictor unset — the run falls back to a
    # SimplePredictor stand-in in _build_tools.
    assert ctx.predictor is None
    assert ctrl.state.has_setup


def test_setup_legacy_use_mock_still_works():
    # the existing call sites use setup(use_mock=True) — keep it working
    ctrl = build_core()
    ctrl.setup(use_mock=True)
    assert "MockQickSoc" in type(ctrl.state.exp_context.soc).__name__


def test_setup_emits_setup_done():
    from zcu_tools.gui.app.autofluxdep.event_bus import SetupDonePayload

    ctrl = build_core()
    seen = []
    ctrl.bus.subscribe(SetupDonePayload, lambda p: seen.append(True))
    ctrl.setup(use_mock=True)
    assert seen == [True]


# --- predictor loading: a missing / bad file leaves the raw predictor unset;
#     the SimplePredictor fallback then happens in _build_tools ---


def test_predictor_blank_path_returns_none():
    assert Controller._load_predictor("") is None


def test_predictor_bad_path_returns_none():
    assert Controller._load_predictor("/nonexistent/params.json") is None


def test_build_tools_falls_back_to_simple_predictor():
    # with no raw FluxoniumPredictor in the context, the sweep's adaptive
    # predictor is the SimplePredictor stand-in, so a mock / unconfigured run
    # still drives the same calibrate loop.
    ctrl = build_core()
    ctrl.setup(use_mock=True)
    tools = ctrl._build_tools()
    assert isinstance(tools.predictor, SimplePredictor)


# --- flux device: a real setup needs a non-blank address (Fast Fail) ---


def test_flux_blank_address_raises():
    with pytest.raises(ValueError, match="address is required"):
        Controller._connect_flux_device("")


def test_flux_whitespace_address_raises():
    with pytest.raises(ValueError, match="address is required"):
        Controller._connect_flux_device("   ")
