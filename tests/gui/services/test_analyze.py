"""Unit tests for AnalyzeService.

Covers start_analyze, _on_analyze_finished, and _on_analyze_failed
using a real State + real EventBus, with the runner mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import ContextReadiness
from zcu_tools.gui.app.main.event_bus import EventBus, GuiEvent
from zcu_tools.gui.app.main.services.analyze import AnalyzeService
from zcu_tools.gui.app.main.services.guard import AnalyzePermit
from zcu_tools.gui.app.main.state import ExpContext, Session, State
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_state(tab_id: str = "tab1") -> State:
    ctx = ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=MagicMock(),
        soccfg=MagicMock(),
        result_dir="/tmp",
        readiness=ContextReadiness.ACTIVE,
    )
    state = State(ctx)
    state.add_tab(
        tab_id,
        Session(adapter_name="fake", adapter=MagicMock(), cfg_schema=MagicMock()),
    )
    # Provide a fake run_result so the tab is not empty
    state.update_tab_result(tab_id, object())
    return state


def _make_service(
    state: State,
    bus: EventBus,
) -> tuple[AnalyzeService, MagicMock]:
    from zcu_tools.gui.app.main.services.operation_gate import OperationGate

    runner = MagicMock()
    writeback = MagicMock()
    writeback.compute_items_for_tab.return_value = []
    # Real (Qt-free) gate so analyze takes a genuine async-handle lease.
    svc = AnalyzeService(state, runner, bus, writeback, OperationGate())
    return svc, runner


# ---------------------------------------------------------------------------
# start_analyze — normal path
# ---------------------------------------------------------------------------


def test_start_analyze_calls_runner(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    svc, runner = _make_service(state, bus)

    svc.start_analyze(AnalyzePermit(tab_id="tab1"), analyze_params_instance=object())

    runner.start_analyze.assert_called_once()
    assert state.get_tab("tab1").is_analyzing is True


def test_start_analyze_emits_interaction_event(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    received: list[str] = []
    bus.subscribe(GuiEvent.TAB_INTERACTION_CHANGED, lambda p: received.append(p.tab_id))

    svc, _ = _make_service(state, bus)
    svc.start_analyze(AnalyzePermit(tab_id="tab1"), analyze_params_instance=object())

    assert "tab1" in received


def test_start_analyze_passes_figure_container(qapp):  # noqa: ARG001
    state = _make_state()
    svc, runner = _make_service(state, EventBus())
    container = MagicMock()

    svc.start_analyze(
        AnalyzePermit(tab_id="tab1"),
        analyze_params_instance=object(),
        figure_container=container,
    )

    _, kwargs = runner.start_analyze.call_args
    assert (
        kwargs.get("figure_container") is container
        or container in runner.start_analyze.call_args[0]
    )


# ---------------------------------------------------------------------------
# start_analyze — busy tab rejection
# ---------------------------------------------------------------------------


def test_start_analyze_rejects_busy_tab(qapp):  # noqa: ARG001
    state = _make_state()
    state.set_tab_running("tab1", True)
    svc, _ = _make_service(state, EventBus())

    with pytest.raises(RuntimeError, match="busy"):
        svc.start_analyze(
            AnalyzePermit(tab_id="tab1"), analyze_params_instance=object()
        )


# ---------------------------------------------------------------------------
# _on_analyze_finished
# ---------------------------------------------------------------------------


def test_on_analyze_finished_updates_state(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    svc, _ = _make_service(state, bus)

    # Put tab into analyzing state first
    state.set_tab_analyzing("tab1", True)

    fake_result = MagicMock()
    fake_result.figure = MagicMock()

    finished_signals: list = []
    svc.analyze_finished.connect(lambda tid, res: finished_signals.append((tid, res)))

    # Invoke the callback directly (runner is a MagicMock — its signals are not real Qt signals)
    svc._on_analyze_finished("tab1", fake_result)

    assert state.get_tab("tab1").analyze_result is fake_result
    assert state.get_tab("tab1").is_analyzing is False
    assert len(finished_signals) == 1
    assert finished_signals[0] == ("tab1", fake_result)


def test_on_analyze_finished_emits_interaction_event(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    received: list[str] = []
    bus.subscribe(GuiEvent.TAB_INTERACTION_CHANGED, lambda p: received.append(p.tab_id))
    svc, _ = _make_service(state, bus)

    state.set_tab_analyzing("tab1", True)

    fake_result = MagicMock()
    fake_result.figure = None
    svc._on_analyze_finished("tab1", fake_result)

    assert "tab1" in received


# ---------------------------------------------------------------------------
# _on_analyze_failed
# ---------------------------------------------------------------------------


def test_on_analyze_failed_resets_state(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    svc, _ = _make_service(state, bus)

    state.set_tab_analyzing("tab1", True)
    assert state.get_tab("tab1").is_analyzing is True

    failed_signals: list = []
    svc.analyze_failed.connect(lambda tid, err: failed_signals.append((tid, err)))

    error = RuntimeError("analysis failed")
    svc._on_analyze_failed("tab1", error)

    assert state.get_tab("tab1").is_analyzing is False
    assert len(failed_signals) == 1
    assert failed_signals[0] == ("tab1", error)


def test_on_analyze_failed_emits_interaction_event(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    received: list[str] = []
    bus.subscribe(GuiEvent.TAB_INTERACTION_CHANGED, lambda p: received.append(p.tab_id))
    svc, _ = _make_service(state, bus)

    state.set_tab_analyzing("tab1", True)
    svc._on_analyze_failed("tab1", RuntimeError("oops"))

    assert "tab1" in received
