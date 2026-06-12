"""Unit tests for PostAnalyzeService.

Mirrors tests/gui/services/test_analyze.py: a real State + real EventBus, with
BackgroundService mocked. Covers the gate (no primary analyze result), the
submit-to-bg path, and the finished/failed terminal slots.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.gui.app.main.adapter import ContextReadiness
from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
from zcu_tools.gui.app.main.services.post_analyze import PostAnalyzeService
from zcu_tools.gui.app.main.state import ExpContext, Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_state(tab_id: str = "tab1", *, with_analyze: bool = True) -> State:
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
    state.update_tab_result(tab_id, object())
    if with_analyze:
        fake_analyze = MagicMock()
        fake_analyze.figure = None
        state.update_tab_analyze(tab_id, fake_analyze, None)
    return state


def _make_service(state: State, bus: EventBus) -> tuple[PostAnalyzeService, MagicMock]:
    bg = MagicMock()
    svc = PostAnalyzeService(state, bg, bus, OperationHandles())
    return svc, bg


def test_start_post_analyze_submits_to_bg(qapp):  # noqa: ARG001
    state = _make_state()
    svc, bg = _make_service(state, EventBus())

    svc.start_post_analyze("tab1", post_analyze_params_instance=object())

    bg.submit.assert_called_once()
    assert bg.submit.call_args.kwargs["run_in_pool"] is False
    assert state.get_tab("tab1").is_analyzing is True


def test_start_post_analyze_emits_interaction_event(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    received: list[str] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.tab_id))

    svc, _ = _make_service(state, bus)
    svc.start_post_analyze("tab1", post_analyze_params_instance=object())

    assert "tab1" in received


def test_start_post_analyze_gates_on_missing_primary_result(qapp):  # noqa: ARG001
    state = _make_state(with_analyze=False)
    svc, bg = _make_service(state, EventBus())

    with pytest.raises(RuntimeError, match="no primary analyze result"):
        svc.start_post_analyze("tab1", post_analyze_params_instance=object())
    bg.submit.assert_not_called()


def test_start_post_analyze_rejects_busy_tab(qapp):  # noqa: ARG001
    state = _make_state()
    state.set_tab_running("tab1", True)
    svc, _ = _make_service(state, EventBus())

    with pytest.raises(RuntimeError, match="busy"):
        svc.start_post_analyze("tab1", post_analyze_params_instance=object())


def test_start_post_analyze_passes_figure_container(qapp):  # noqa: ARG001
    state = _make_state()
    svc, bg = _make_service(state, EventBus())
    container = MagicMock()

    svc.start_post_analyze(
        "tab1", post_analyze_params_instance=object(), figure_container=container
    )

    scopes = bg.submit.call_args.args[1]
    assert scopes.figure_container is container


def test_on_post_analyze_finished_updates_state(qapp):  # noqa: ARG001
    state = _make_state()
    svc, _ = _make_service(state, EventBus())
    state.set_tab_analyzing("tab1", True)

    post_result = MagicMock()
    post_result.figure = Figure()

    finished: list = []
    svc.post_analyze_finished.connect(lambda tid, r: finished.append((tid, r)))

    svc._on_post_analyze_finished("tab1", post_result)

    tab = state.get_tab("tab1")
    assert tab.post_analyze_result is post_result
    assert tab.post_figure is post_result.figure
    assert tab.is_analyzing is False
    assert finished == [("tab1", post_result)]


def test_on_post_analyze_failed_resets_state(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    svc, _ = _make_service(state, bus)
    state.set_tab_analyzing("tab1", True)

    failed: list = []
    svc.post_analyze_failed.connect(lambda tid, err: failed.append((tid, err)))

    error = RuntimeError("post analysis failed")
    svc._on_post_analyze_failed("tab1", error)

    assert state.get_tab("tab1").is_analyzing is False
    assert failed == [("tab1", error)]
