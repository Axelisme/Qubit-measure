"""Tests for ContextService."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

from zcu_tools.gui.adapter import ContextReadiness
from zcu_tools.gui.event_bus import GuiEvent
from zcu_tools.gui.services.context import ContextService
from zcu_tools.gui.state import ExpContext, State


def test_context_service_has_project():
    io_mock = MagicMock()
    io_mock.has_project = True

    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    svc = ContextService(state, io_mock, MagicMock())
    assert svc.has_project()


def test_context_service_has_context():
    io_mock = MagicMock()
    io_mock.has_context = False

    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    state.has_startup_context = True

    svc = ContextService(state, io_mock, MagicMock())
    assert svc.has_context()


def test_context_service_get_flux_dir():
    io_mock = MagicMock()
    io_mock.get_active_label.return_value = "flux_1.23_A"

    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    ctx = ExpContext(
        md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="/base/dir"
    )
    state.set_context(ctx)

    svc = ContextService(state, io_mock, MagicMock())
    import os

    expected = os.path.join("/base/dir", "exps", "flux_1.23_A")
    assert svc.get_flux_dir() == expected


def test_context_service_set_startup_context():
    state = State(
        ExpContext(
            md=MagicMock(),
            ml=MagicMock(),
            soc=None,
            soccfg=None,
            result_dir="",
            active_label="old_flux",
            readiness=ContextReadiness.ACTIVE,
        )
    )
    bus = MagicMock()
    io_mock = MagicMock()

    svc = ContextService(state, io_mock, bus)

    md = MagicMock()
    ml = MagicMock()

    svc.set_startup_context(
        md,
        ml,
        chip_name="C1",
        qub_name="Q1",
        res_name="R1",
        result_dir="/res",
        database_path="/db",
    )

    assert state.has_startup_context
    assert state.exp_context.chip_name == "C1"
    assert state.exp_context.result_dir == "/res"
    assert state.exp_context.active_label == ""
    assert state.exp_context.readiness is ContextReadiness.DRAFT
    assert not svc.is_active_context()
    bus.emit.assert_called_once()
    assert bus.emit.call_args[0][0] == GuiEvent.CONTEXT_SWITCHED


def test_context_service_use_context():
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    bus = MagicMock()
    io_mock = MagicMock()

    mock_md = MagicMock()
    mock_ml = MagicMock()
    mock_ctx = ExpContext(
        md=mock_md, ml=mock_ml, soc=None, soccfg=None, result_dir="/base"
    )
    io_mock.use_context.return_value = mock_ctx

    svc = ContextService(state, io_mock, bus)

    # We must have a baseline result_dir in the startup context for use_context to inherit
    state.exp_context = dataclasses.replace(state.exp_context, result_dir="/base")
    old_ctx = state.exp_context

    svc.use_context("flux_1.0_A")

    io_mock.use_context.assert_called_with("flux_1.0_A", old_ctx)
    assert state.exp_context.md == mock_md
    assert state.exp_context.ml == mock_ml
    assert state.exp_context.result_dir == "/base"
    assert state.exp_context.active_label == "flux_1.0_A"
    assert state.exp_context.readiness is ContextReadiness.ACTIVE
    assert svc.is_active_context()
    bus.emit.assert_called_once()


def test_context_service_new_context():
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    bus = MagicMock()
    io_mock = MagicMock()

    # baseline MD/ML
    base_md = MagicMock()
    base_ml = MagicMock()
    base_ctx = dataclasses.replace(state.exp_context, md=base_md, ml=base_ml)
    state.exp_context = base_ctx

    mock_ctx = ExpContext(
        md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="/base"
    )
    io_mock.new_context.return_value = mock_ctx
    io_mock.get_active_label.return_value = "flux_1.5_V"

    svc = ContextService(state, io_mock, bus)

    svc.new_context(value=1.5, unit="V", clone_from_current=True)

    io_mock.new_context.assert_called_with(
        base_ctx, value=1.5, unit="V", clone_from_current=True
    )
    bus.emit.assert_called_once()
    assert bus.emit.call_args[0][0] == GuiEvent.CONTEXT_SWITCHED
    assert state.exp_context.active_label == "flux_1.5_V"
    assert state.exp_context.readiness is ContextReadiness.ACTIVE
