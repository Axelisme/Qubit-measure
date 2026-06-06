"""Controller setup tests — SetupRequest routing, predictor / flux fallbacks.

The mock path is exercised end-to-end; the real path's helpers are unit-tested
in isolation (predictor degradation, flux-address validation) without touching
the network — ``make_soc_proxy`` is never called here.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.state import SetupRequest

# --- setup routing: mock request + legacy use_mock shorthand ---


def test_setup_mock_request_builds_resources():
    ctrl = build_core()
    ctrl.setup(SetupRequest(use_mock=True))
    res = ctrl.state.resources
    assert res is not None
    assert "MockQickSoc" in type(res.soc).__name__
    assert type(res.predictor).__name__ == "SimplePredictor"
    assert ctrl.state.has_setup


def test_setup_legacy_use_mock_still_works():
    # the 11 existing call sites use setup(use_mock=True) — keep it working
    ctrl = build_core()
    ctrl.setup(use_mock=True)
    res = ctrl.state.resources
    assert res is not None
    assert "MockQickSoc" in type(res.soc).__name__


def test_setup_emits_setup_done():
    from zcu_tools.gui.app.autofluxdep.event_bus import SetupDonePayload

    ctrl = build_core()
    seen = []
    ctrl.bus.subscribe(SetupDonePayload, lambda p: seen.append(True))
    ctrl.setup(use_mock=True)
    assert seen == [True]


# --- predictor loading: degrades to SimplePredictor on missing / bad file ---


def test_predictor_blank_path_falls_back():
    p = Controller._load_predictor("")
    assert type(p).__name__ == "SimplePredictor"


def test_predictor_bad_path_falls_back():
    p = Controller._load_predictor("/nonexistent/params.json")
    assert type(p).__name__ == "SimplePredictor"


# --- flux device: a real setup needs a non-blank address (Fast Fail) ---


def test_flux_blank_address_raises():
    with pytest.raises(ValueError, match="address is required"):
        Controller._connect_flux_device("")


def test_flux_whitespace_address_raises():
    with pytest.raises(ValueError, match="address is required"):
        Controller._connect_flux_device("   ")
