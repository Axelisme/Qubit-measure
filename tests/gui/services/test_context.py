"""Tests for ContextService."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import ContextReadiness
from zcu_tools.gui.app.main.services.context import ContextService
from zcu_tools.gui.app.main.state import ExpContext, State
from zcu_tools.gui.session.events import SessionEvent
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


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
        ExpContext(
            md=MagicMock(),
            ml=MagicMock(),
            soc=None,
            soccfg=None,
            result_dir="",
            readiness=ContextReadiness.DRAFT,
        )
    )

    svc = ContextService(state, io_mock, MagicMock())
    assert svc.has_context()
    assert svc.has_startup_context()
    assert not svc.is_active_context()


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

    assert svc.has_startup_context()
    assert state.exp_context.chip_name == "C1"
    assert state.exp_context.result_dir == "/res"
    assert state.exp_context.active_label == ""
    assert state.exp_context.readiness is ContextReadiness.DRAFT
    assert not svc.is_active_context()
    bus.emit.assert_called_once()
    assert bus.emit.call_args[0][0].EVENT == SessionEvent.CONTEXT_SWITCHED


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
    ctx_version_before = state.version.get("context")

    svc.use_context("flux_1.0_A")

    io_mock.use_context.assert_called_with("flux_1.0_A", old_ctx)
    assert state.exp_context.md == mock_md
    assert state.exp_context.ml == mock_ml
    assert state.exp_context.result_dir == "/base"
    assert state.exp_context.active_label == "flux_1.0_A"
    assert state.exp_context.readiness is ContextReadiness.ACTIVE
    assert svc.is_active_context()
    bus.emit.assert_called_once()
    # Switching context fully swaps md/ml → context version must advance.
    assert state.version.get("context") == ctx_version_before + 1


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

    svc.new_context(value=1.5, unit="V", clone_from="src_label")

    io_mock.new_context.assert_called_with(
        base_ctx, value=1.5, unit="V", clone_from="src_label"
    )
    bus.emit.assert_called_once()
    assert bus.emit.call_args[0][0].EVENT == SessionEvent.CONTEXT_SWITCHED
    assert state.exp_context.active_label == "flux_1.5_V"
    assert state.exp_context.readiness is ContextReadiness.ACTIVE


def test_context_service_readiness_transitions_drive_has_context_queries():
    """has_context / has_startup_context / is_active_context all derive from readiness."""
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    io_mock = MagicMock()
    io_mock.has_context = False
    svc = ContextService(state, io_mock, MagicMock())

    # EMPTY
    assert not svc.has_context()
    assert not svc.has_startup_context()
    assert not svc.is_active_context()

    # DRAFT
    svc.set_startup_context(MagicMock(), MagicMock(), "C", "Q", "R", "/res", "/db")
    assert svc.has_context()
    assert svc.has_startup_context()
    assert not svc.is_active_context()

    # ACTIVE
    io_mock.use_context.return_value = ExpContext(
        md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="/res"
    )
    svc.use_context("flux_1.0_A")
    assert svc.has_context()
    assert not svc.has_startup_context()
    assert svc.is_active_context()


# ---------------------------------------------------------------------------
# Helper for md/ml mutation tests
# ---------------------------------------------------------------------------


def _make_active_state() -> tuple[State, ContextService]:
    md = MetaDict()
    ml = ModuleLibrary()
    state = State(
        ExpContext(
            md=md,
            ml=ml,
            soc=None,
            soccfg=None,
            result_dir="",
            readiness=ContextReadiness.ACTIVE,
        )
    )
    bus = MagicMock()
    svc = ContextService(state, MagicMock(), bus)
    return state, svc


# ---------------------------------------------------------------------------
# del_md_attr
# ---------------------------------------------------------------------------


def test_del_md_attr_removes_attribute_and_bumps_context():
    state, svc = _make_active_state()
    state.exp_context.md.r_f = 6000.0
    before = state.version.get("context")

    svc.del_md_attr("r_f")

    assert not hasattr(state.exp_context.md, "r_f")
    assert state.version.get("context") == before + 1


def test_del_md_attr_emits_md_changed():
    state, svc = _make_active_state()
    state.exp_context.md.r_f = 6000.0
    bus: MagicMock = svc._bus  # type: ignore[assignment]

    svc.del_md_attr("r_f")

    bus.emit.assert_called()
    event_calls = [c[0][0].EVENT for c in bus.emit.call_args_list]
    assert SessionEvent.MD_CHANGED in event_calls


# ---------------------------------------------------------------------------
# del_ml_module
# ---------------------------------------------------------------------------


def test_del_ml_module_removes_module_and_bumps_context():
    state, svc = _make_active_state()
    ml = state.exp_context.ml
    fake_module = MagicMock()
    ml.register_module(qub=fake_module)
    before = state.version.get("context")

    svc.del_ml_module("qub")

    assert "qub" not in ml.modules
    assert state.version.get("context") == before + 1


def test_del_ml_module_emits_ml_changed():
    state, svc = _make_active_state()
    ml = state.exp_context.ml
    fake_module = MagicMock()
    ml.register_module(qub=fake_module)
    bus: MagicMock = svc._bus  # type: ignore[assignment]
    bus.reset_mock()

    svc.del_ml_module("qub")

    event_calls = [c[0][0].EVENT for c in bus.emit.call_args_list]
    assert SessionEvent.ML_CHANGED in event_calls


# ---------------------------------------------------------------------------
# rename_ml_module / rename_ml_waveform
# ---------------------------------------------------------------------------


def test_rename_ml_module_moves_key_and_emits_once():
    state, svc = _make_active_state()
    ml = state.exp_context.ml
    fake_module = MagicMock()
    ml.register_module(qub=fake_module)
    bus: MagicMock = svc._bus  # type: ignore[assignment]
    bus.reset_mock()
    before = state.version.get("context")

    svc.rename_ml_module("qub", "qub2")

    assert "qub" not in ml.modules
    assert "qub2" in ml.modules
    assert state.version.get("context") == before + 1
    event_calls = [c[0][0].EVENT for c in bus.emit.call_args_list]
    assert event_calls.count(SessionEvent.ML_CHANGED) == 1


def test_rename_ml_module_clash_fails():
    state, svc = _make_active_state()
    ml = state.exp_context.ml
    ml.register_module(a=MagicMock(), b=MagicMock())
    with pytest.raises(RuntimeError, match="already exists"):
        svc.rename_ml_module("a", "b")
    assert "a" in ml.modules  # untouched


def test_rename_ml_module_missing_fails():
    state, svc = _make_active_state()
    with pytest.raises(RuntimeError, match="No module named"):
        svc.rename_ml_module("nope", "x")


def test_rename_ml_module_empty_name_fails():
    state, svc = _make_active_state()
    state.exp_context.ml.register_module(qub=MagicMock())
    with pytest.raises(RuntimeError, match="must not be empty"):
        svc.rename_ml_module("qub", "")


def test_rename_ml_waveform_moves_key():
    state, svc = _make_active_state()
    ml = state.exp_context.ml
    ml.register_waveform(gauss=MagicMock())
    svc.rename_ml_waveform("gauss", "gauss2")
    assert "gauss" not in ml.waveforms
    assert "gauss2" in ml.waveforms


# ---------------------------------------------------------------------------
# del_ml_waveform
# ---------------------------------------------------------------------------


def test_del_ml_waveform_removes_waveform_and_bumps_context():
    state, svc = _make_active_state()
    ml = state.exp_context.ml
    fake_wf = MagicMock()
    ml.register_waveform(gauss=fake_wf)
    before = state.version.get("context")

    svc.del_ml_waveform("gauss")

    assert "gauss" not in ml.waveforms
    assert state.version.get("context") == before + 1
