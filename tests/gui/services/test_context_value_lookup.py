from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.app.main.adapter import ContextReadiness
from zcu_tools.gui.app.main.state import ExpContext, State
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.value_lookup import EmptyValueLookup, ValueRegistry


def _state() -> State:
    return State(ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None))


def test_context_service_injects_value_lookup_without_context_bump() -> None:
    state = _state()
    registry = ValueRegistry()
    before = state.version.get("context")

    ContextService(state, MagicMock(), MagicMock(), values=registry)

    assert state.exp_context.values is registry
    assert state.version.get("context") == before


def test_startup_context_preserves_injected_value_lookup() -> None:
    state = _state()
    registry = ValueRegistry()
    svc = ContextService(state, MagicMock(), MagicMock(), values=registry)

    svc.set_startup_context(MagicMock(), MagicMock(), "C", "Q", "R", "/res", "/db")

    assert state.exp_context.values is registry
    assert state.exp_context.readiness is ContextReadiness.DRAFT


def test_use_context_preserves_lookup_when_io_returns_fresh_context() -> None:
    state = _state()
    registry = ValueRegistry()
    io = MagicMock()
    io.use_context.return_value = ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=None,
        soccfg=None,
        values=EmptyValueLookup(),
    )
    svc = ContextService(state, io, MagicMock(), values=registry)

    old_ctx = state.exp_context
    svc.use_context("flux_0.0_A")

    io.use_context.assert_called_once_with("flux_0.0_A", old_ctx)
    assert state.exp_context.values is registry
    assert state.exp_context.active_label == "flux_0.0_A"
    assert state.exp_context.readiness is ContextReadiness.ACTIVE


def test_new_context_preserves_lookup_when_io_returns_fresh_context() -> None:
    state = _state()
    registry = ValueRegistry()
    io = MagicMock()
    io.new_context.return_value = ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=None,
        soccfg=None,
        values=EmptyValueLookup(),
    )
    io.get_active_label.return_value = "flux_1.0_V"
    svc = ContextService(state, io, MagicMock(), values=registry)

    old_ctx = state.exp_context
    svc.new_context(value=1.0, unit="V", clone_from="base")

    io.new_context.assert_called_once_with(
        old_ctx, value=1.0, unit="V", clone_from="base"
    )
    assert state.exp_context.values is registry
    assert state.exp_context.active_label == "flux_1.0_V"
    assert state.exp_context.readiness is ContextReadiness.ACTIVE
