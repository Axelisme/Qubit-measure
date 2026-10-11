from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.app.measure.adapter import ContextReadiness
from zcu_tools.gui.app.measure.state import SessionEnv, State
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.value_lookup import ValueKey, ValueRegistry
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


def _state() -> State:
    return State(SessionEnv(parameters=None, soc=None, soccfg=None))


def test_context_service_injects_value_lookup_without_context_bump() -> None:
    state = _state()
    registry = ValueRegistry()
    before = state.version.get("context")
    ContextService(
        state, MagicMock(), values=registry, bind_assets=ArbWaveformDatabase.info_at
    )
    assert state.session_env.values is registry
    assert state.version.get("context") == before


def test_project_context_preserves_injected_value_lookup(tmp_path) -> None:
    state = _state()
    registry = ValueRegistry()
    svc = ContextService(
        state, MagicMock(), values=registry, bind_assets=ArbWaveformDatabase.info_at
    )
    svc.install_entry(
        ResultEntry.create(
            "C", result_root=tmp_path / "res", database_root=tmp_path / "db"
        )
    )
    assert state.session_env.values is registry
    assert state.session_env.readiness is ContextReadiness.EMPTY


def test_use_context_preserves_lookup_when_io_returns_fresh_context(tmp_path) -> None:
    state = _state()
    registry = ValueRegistry()
    svc = ContextService(
        state, MagicMock(), values=registry, bind_assets=ArbWaveformDatabase.info_at
    )
    entry = ResultEntry.create(
        "C", result_root=tmp_path / "res", database_root=tmp_path / "db"
    )
    entry.new_point("flux_0.0_A", description="")
    svc.install_entry(entry)
    svc.use_point(PointUseRequest("flux_0.0_A"))
    assert state.session_env.values is registry
    assert state.session_env.active_label == "flux_0.0_A"
    assert state.session_env.readiness is ContextReadiness.ACTIVE


def test_new_context_preserves_lookup_when_io_returns_fresh_context(tmp_path) -> None:
    state = _state()
    registry = ValueRegistry()
    svc = ContextService(
        state, MagicMock(), values=registry, bind_assets=ArbWaveformDatabase.info_at
    )
    entry = ResultEntry.create(
        "C", result_root=tmp_path / "res", database_root=tmp_path / "db"
    )
    entry.new_point("base", description="")
    svc.install_entry(entry)
    value, unit = 1.0, "V"
    label = f"flux_{value}_{unit}"
    svc.new_point(
        PointCreateRequest(label, "", CommandOrigin("gui"), clone_from="base")
    )
    assert state.session_env.values is registry
    assert state.session_env.active_label == "flux_1.0_V"
    assert state.session_env.readiness is ContextReadiness.ACTIVE


def test_context_service_lists_and_reads_value_sources() -> None:
    state = _state()
    registry = ValueRegistry()
    registry.register(
        ValueKey("device.flux.value", float),
        lambda: 0.125,
        owner="device:flux",
        description="Named device cached value.",
    )
    svc = ContextService(
        state, MagicMock(), values=registry, bind_assets=ArbWaveformDatabase.info_at
    )
    assert [info.key for info in svc.list_value_sources()] == ["device.flux.value"]
    info, value = svc.read_value_source("device.flux.value", "float")
    assert info.type_name == "float"
    assert info.owner == "device:flux"
    assert value == 0.125
