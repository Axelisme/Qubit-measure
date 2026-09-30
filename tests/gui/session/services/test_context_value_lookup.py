from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.app.measure.adapter import ContextReadiness
from zcu_tools.gui.app.measure.state import SessionEnv, State
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.value_lookup import (
    EmptyValueLookup,
    ValueKey,
    ValueRegistry,
)


def _state() -> State:
    return State(SessionEnv(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None))


def test_context_service_injects_value_lookup_without_context_bump() -> None:
    state = _state()
    registry = ValueRegistry()
    before = state.version.get("context")

    ContextService(state, MagicMock(), MagicMock(), values=registry)

    assert state.session_env.values is registry
    assert state.version.get("context") == before


def test_project_context_preserves_injected_value_lookup() -> None:
    state = _state()
    registry = ValueRegistry()
    svc = ContextService(state, MagicMock(), MagicMock(), values=registry)

    svc.set_project_context(MagicMock(), MagicMock(), "C", "Q", "R", "/res", "/db")

    assert state.session_env.values is registry
    assert state.session_env.readiness is ContextReadiness.DRAFT


def test_use_context_preserves_lookup_when_io_returns_fresh_context() -> None:
    state = _state()
    registry = ValueRegistry()
    io = MagicMock()
    io.use_context.return_value = SessionEnv(
        md=MagicMock(),
        ml=MagicMock(),
        soc=None,
        soccfg=None,
        values=EmptyValueLookup(),
    )
    svc = ContextService(state, io, MagicMock(), values=registry)

    old_ctx = state.session_env
    svc.use_context("flux_0.0_A")

    io.use_context.assert_called_once_with("flux_0.0_A", old_ctx)
    assert state.session_env.values is registry
    assert state.session_env.active_label == "flux_0.0_A"
    assert state.session_env.readiness is ContextReadiness.ACTIVE


def test_new_context_preserves_lookup_when_io_returns_fresh_context() -> None:
    state = _state()
    registry = ValueRegistry()
    io = MagicMock()
    io.new_context.return_value = SessionEnv(
        md=MagicMock(),
        ml=MagicMock(),
        soc=None,
        soccfg=None,
        values=EmptyValueLookup(),
    )
    io.get_active_label.return_value = "flux_1.0_V"
    io.list_contexts.return_value = ["base"]
    svc = ContextService(state, io, MagicMock(), values=registry)

    old_ctx = state.session_env
    svc.new_context(value=1.0, unit="V", clone_from="base")

    io.new_context.assert_called_once_with(
        old_ctx, value=1.0, unit="V", clone_from="base", label=None
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
    svc = ContextService(state, MagicMock(), MagicMock(), values=registry)

    assert [info.key for info in svc.list_value_sources()] == ["device.flux.value"]
    info, value = svc.read_value_source("device.flux.value", "float")

    assert info.type_name == "float"
    assert info.owner == "device:flux"
    assert value == 0.125
