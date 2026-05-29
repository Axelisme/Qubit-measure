"""Unit tests for zcu_tools.gui.state."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.gui.adapter import CfgSchema, CfgSectionSpec, CfgSectionValue, SavePaths
from zcu_tools.gui.state import (
    DeviceState,
    DeviceStatus,
    State,
    TabInteractionState,
    TabState,
)


def _make_ctx():
    return MagicMock()


def _make_adapter():
    return MagicMock()


def _add_tab(state: State, tab_id: str, adapter: MagicMock) -> object:
    cfg_schema = CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())
    state.add_tab(
        tab_id,
        TabState(adapter_name="fake", adapter=adapter, cfg_schema=cfg_schema),
    )
    return cfg_schema


@dataclass
class _AnalyzeParams:
    threshold: float


def test_tab_interaction_state_creation():
    state = TabInteractionState(
        global_run_active=True,
        is_running=False,
        is_analyzing=True,
        is_saving_data=False,
        has_context=False,
        has_active_context=False,
        has_soc=True,
        has_run_result=True,
        has_analyze_result=False,
        has_figure=False,
    )
    assert state.global_run_active is True
    assert state.is_running is False
    assert state.is_analyzing is True
    assert state.is_saving_data is False


def test_add_tab_then_get_tab_returns_correct_tabstate():
    state = State(_make_ctx())
    adapter = _make_adapter()
    cfg_schema = _add_tab(state, "t1", adapter)
    tab = state.get_tab("t1")
    assert isinstance(tab, TabState)
    assert tab.adapter_name == "fake"
    assert tab.adapter is adapter
    assert tab.cfg_schema is cfg_schema


def test_add_tab_duplicate_raises():
    state = State(_make_ctx())
    adapter = _make_adapter()
    _add_tab(state, "t1", adapter)
    with pytest.raises(ValueError, match="already exists"):
        dup_adapter = _make_adapter()
        _add_tab(state, "t1", dup_adapter)


def test_remove_tab_clears_active_tab_id():
    state = State(_make_ctx())
    adapter = _make_adapter()
    _add_tab(state, "t1", adapter)
    state.set_active_tab("t1")
    state.remove_tab("t1")
    assert state.active_tab_id is None


def test_remove_busy_tab_raises():
    state = State(_make_ctx())
    adapter = _make_adapter()
    _add_tab(state, "t1", adapter)
    state.set_tab_analyzing("t1", True)
    with pytest.raises(RuntimeError, match="busy tab"):
        state.remove_tab("t1")


def test_set_active_tab_unknown_raises():
    state = State(_make_ctx())
    with pytest.raises(KeyError):
        state.set_active_tab("nonexistent")


def test_set_tab_running_updates_running_tab_id():
    state = State(_make_ctx())
    adapter = _make_adapter()
    _add_tab(state, "t1", adapter)
    assert state.is_run_active() is False
    state.set_tab_running("t1", True)
    assert state.is_run_active() is True
    assert state.running_tab_id == "t1"
    assert state.is_tab_running("t1") is True
    state.set_tab_running("t1", False)
    assert state.is_run_active() is False
    assert state.running_tab_id is None


def test_is_tab_busy_checks_per_tab_flags():
    state = State(_make_ctx())
    adapter = _make_adapter()
    _add_tab(state, "t1", adapter)
    _add_tab(state, "t2", adapter)
    assert state.is_tab_busy("t1") is False
    state.set_tab_saving_data("t1", True)
    assert state.is_tab_busy("t1") is True
    assert state.is_tab_busy("t2") is False


def test_update_tab_result_stores_result_and_clears_stale_analyze_data():
    state = State(_make_ctx())
    adapter = _make_adapter()
    _add_tab(state, "t1", adapter)
    state.update_tab_analyze_params("t1", _AnalyzeParams(threshold=0.5))
    fig = Figure()
    state.update_tab_analyze("t1", object(), fig)
    state.update_tab_result("t1", object())
    tab = state.get_tab("t1")
    assert tab.analyze_result is None
    assert tab.figure is None  # figure is cleared with stale analyze data
    assert tab.analyze_param_instance is None


def test_update_tab_analyze_stores_analyze_result_and_figure():
    state = State(_make_ctx())
    adapter = _make_adapter()
    _add_tab(state, "t1", adapter)
    analyze_result = object()
    fig = Figure()
    state.update_tab_analyze("t1", analyze_result, fig)
    tab = state.get_tab("t1")
    assert tab.analyze_result is analyze_result
    assert tab.figure is fig


def test_update_tab_analyze_params_stores_instance():
    state = State(_make_ctx())
    adapter = _make_adapter()
    _add_tab(state, "t1", adapter)
    params = _AnalyzeParams(threshold=0.2)
    state.update_tab_analyze_params("t1", params)
    assert state.get_tab("t1").analyze_param_instance is params


def test_update_tab_save_path_overrides_sets_both_paths():
    state = State(_make_ctx())
    adapter = _make_adapter()
    _add_tab(state, "t1", adapter)
    state.update_tab_save_path_overrides(
        "t1", SavePaths("/tmp/custom-data", "/tmp/custom-image")
    )
    assert state.get_effective_save_paths("t1") == SavePaths(
        "/tmp/custom-data", "/tmp/custom-image"
    )


def test_set_context_replaces_exp_context():
    ctx1 = _make_ctx()
    ctx2 = _make_ctx()
    state = State(ctx1)
    assert state.exp_context is ctx1
    state.set_context(ctx2)
    assert state.exp_context is ctx2


# ----------------------------------------------------------------------
# Device state mutators / version bumps
# ----------------------------------------------------------------------


def _dev(
    name: str = "dev", *, status: DeviceStatus, remember: bool = True
) -> DeviceState:
    return DeviceState(
        name=name,
        type_name="FakeDevice",
        address="addr",
        status=status,
        remember=remember,
    )


def test_put_device_inserts_and_bumps():
    state = State(_make_ctx())
    state.put_device(_dev(status=DeviceStatus.CONNECTED))
    assert state.get_device("dev") is not None
    assert state.has_device("dev") is True
    assert state.version.get("device:dev") == 1


def test_put_device_replaces_existing():
    state = State(_make_ctx())
    state.put_device(_dev(status=DeviceStatus.CONNECTING))
    state.put_device(_dev(status=DeviceStatus.CONNECTED))
    dev = state.get_device("dev")
    assert dev is not None and dev.status is DeviceStatus.CONNECTED
    assert state.version.get("device:dev") == 2


def test_set_device_status_replaces_and_bumps():
    state = State(_make_ctx())
    state.put_device(_dev(status=DeviceStatus.CONNECTING))
    state.set_device_status("dev", DeviceStatus.CONNECTED)
    dev = state.get_device("dev")
    assert dev is not None and dev.status is DeviceStatus.CONNECTED
    assert dev.error is None
    assert state.version.get("device:dev") == 2


def test_set_device_info_bumps():
    state = State(_make_ctx())
    state.put_device(_dev(status=DeviceStatus.CONNECTED))
    info = BaseDeviceInfo(address="addr", type="FakeDevice")
    state.set_device_info("dev", info)
    dev = state.get_device("dev")
    assert dev is not None and dev.info is info
    assert state.version.get("device:dev") == 2


def test_set_device_remember_bumps():
    state = State(_make_ctx())
    state.put_device(_dev(status=DeviceStatus.CONNECTED, remember=True))
    state.set_device_remember("dev", False)
    dev = state.get_device("dev")
    assert dev is not None and dev.remember is False
    assert state.version.get("device:dev") == 2


def test_refresh_device_info_cache_does_not_bump():
    state = State(_make_ctx())
    state.put_device(_dev(status=DeviceStatus.CONNECTED))
    assert state.version.get("device:dev") == 1
    info = BaseDeviceInfo(address="addr", type="FakeDevice")
    state.refresh_device_info_cache("dev", info)
    dev = state.get_device("dev")
    assert dev is not None and dev.info is info
    # Cache refresh on read is NOT a semantic write — version must not advance.
    assert state.version.get("device:dev") == 1


def test_remove_device_drops_version_prefix():
    state = State(_make_ctx())
    state.put_device(_dev(status=DeviceStatus.CONNECTED))
    assert state.version.get("device:dev") == 1
    state.remove_device("dev")
    assert state.has_device("dev") is False
    # Dropped key reads as version 0 (gone) — guard treats a dep on it as stale.
    assert state.version.get("device:dev") == 0


def test_list_devices_sorted_by_name():
    state = State(_make_ctx())
    state.put_device(_dev("zeta", status=DeviceStatus.CONNECTED))
    state.put_device(_dev("alpha", status=DeviceStatus.MEMORY_ONLY))
    names = [d.name for d in state.list_devices()]
    assert names == ["alpha", "zeta"]


def test_get_device_unknown_returns_none():
    state = State(_make_ctx())
    assert state.get_device("nope") is None
    assert state.has_device("nope") is False
