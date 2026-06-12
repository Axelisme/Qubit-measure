"""Unit tests for AnalyzeService.

Covers start_analyze, _on_analyze_finished, and _on_analyze_failed
using a real State + real EventBus, with BackgroundService mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import ContextReadiness
from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
from zcu_tools.gui.app.main.services.analyze import AnalyzeService
from zcu_tools.gui.app.main.services.guard import AnalyzePermit
from zcu_tools.gui.app.main.state import ExpContext, Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
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
    from zcu_tools.gui.session.operation_handles import OperationHandles

    bg = MagicMock()  # BackgroundService stand-in; submit() is inspected per-test
    writeback = MagicMock()
    writeback.compute_items_for_tab.return_value = []
    # Real (Qt-free) Handles so analyze takes a genuine async handle (no
    # exclusion — ADR-0019).
    svc = AnalyzeService(state, bg, bus, writeback, OperationHandles())
    return svc, bg


def _make_two_tab_state() -> State:
    state = _make_state("tab1")
    state.add_tab(
        "tab2",
        Session(adapter_name="fake", adapter=MagicMock(), cfg_schema=MagicMock()),
    )
    state.update_tab_result("tab2", object())
    return state


# ---------------------------------------------------------------------------
# start_analyze — normal path
# ---------------------------------------------------------------------------


def test_start_analyze_submits_to_bg(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    svc, bg = _make_service(state, bus)

    svc.start_analyze(AnalyzePermit(tab_id="tab1"), analyze_params_instance=object())

    bg.submit.assert_called_once()
    # FIT analyze is the OffMain-thread strategy (not pooled).
    assert bg.submit.call_args.kwargs["run_in_pool"] is False
    assert state.get_tab("tab1").is_analyzing is True


def test_start_analyze_emits_interaction_event(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    received: list[str] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.tab_id))

    svc, _ = _make_service(state, bus)
    svc.start_analyze(AnalyzePermit(tab_id="tab1"), analyze_params_instance=object())

    assert "tab1" in received


def test_start_analyze_passes_figure_container_in_scopes(qapp):  # noqa: ARG001
    state = _make_state()
    svc, bg = _make_service(state, EventBus())
    container = MagicMock()

    svc.start_analyze(
        AnalyzePermit(tab_id="tab1"),
        analyze_params_instance=object(),
        figure_container=container,
    )

    # The container is carried by the OffMainScopes (2nd positional arg to submit).
    scopes = bg.submit.call_args.args[1]
    assert scopes.figure_container is container


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
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.tab_id))
    svc, _ = _make_service(state, bus)

    state.set_tab_analyzing("tab1", True)

    fake_result = MagicMock()
    fake_result.figure = None
    svc._on_analyze_finished("tab1", fake_result)

    assert "tab1" in received


# ---------------------------------------------------------------------------
# Interactive analysis (no worker; result produced on the user's Done)
# ---------------------------------------------------------------------------


def test_start_interactive_marks_analyzing_without_bg(qapp):  # noqa: ARG001
    state = _make_state()
    svc, bg = _make_service(state, EventBus())

    token = svc.start_interactive(AnalyzePermit(tab_id="tab1"))

    assert isinstance(token, int)
    assert state.get_tab("tab1").is_analyzing is True
    bg.submit.assert_not_called()  # INTERACTIVE never starts a worker (main-thread)


def test_start_interactive_rejects_busy_tab(qapp):  # noqa: ARG001
    state = _make_state()
    state.set_tab_running("tab1", True)
    svc, _ = _make_service(state, EventBus())

    with pytest.raises(RuntimeError, match="busy"):
        svc.start_interactive(AnalyzePermit(tab_id="tab1"))


def test_finish_interactive_runs_the_fit_terminal_path(qapp):  # noqa: ARG001
    state = _make_state()
    svc, _ = _make_service(state, EventBus())
    svc.start_interactive(AnalyzePermit(tab_id="tab1"))

    fake_result = MagicMock()
    fake_result.figure = MagicMock()
    session = MagicMock()
    session.finish.return_value = fake_result

    finished: list = []
    svc.analyze_finished.connect(lambda tid, res: finished.append((tid, res)))

    svc.finish_interactive("tab1", session)

    session.finish.assert_called_once_with()
    # Same terminal effects as a FIT result: State updated, analyzing cleared,
    # lease released, analyze_finished emitted (so the agent's result-poll wakes).
    assert state.get_tab("tab1").analyze_result is fake_result
    assert state.get_tab("tab1").is_analyzing is False
    assert finished == [("tab1", fake_result)]


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
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.tab_id))
    svc, _ = _make_service(state, bus)

    state.set_tab_analyzing("tab1", True)
    svc._on_analyze_failed("tab1", RuntimeError("oops"))

    assert "tab1" in received


# ---------------------------------------------------------------------------
# Concurrent tabs — no exclusion gate (ADR-0019): each settles its own token
# ---------------------------------------------------------------------------


def test_two_tabs_settle_their_own_tokens(qapp):  # noqa: ARG001
    state = _make_two_tab_state()
    svc, _ = _make_service(state, EventBus())
    handles = svc._handles

    token1 = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object()
    )
    token2 = svc.start_analyze(
        AnalyzePermit(tab_id="tab2"), analyze_params_instance=object()
    )

    # Two distinct live handles — the second start must not clobber the first.
    assert token1 != token2
    assert handles.poll(token1) is None  # still pending
    assert handles.poll(token2) is None
    assert handles.live_count() == 2

    # Finish tab1 only: its token settles, tab2's stays live.
    r1 = MagicMock()
    r1.figure = None
    svc._on_analyze_finished("tab1", r1)

    outcome1 = handles.poll(token1)
    assert outcome1 is not None and outcome1.status == "finished"
    assert handles.poll(token2) is None  # tab2 untouched
    assert state.get_tab("tab1").is_analyzing is False
    assert state.get_tab("tab2").is_analyzing is True
    assert handles.live_count() == 1

    # Finish tab2: its own (later) token settles.
    r2 = MagicMock()
    r2.figure = None
    svc._on_analyze_finished("tab2", r2)

    outcome2 = handles.poll(token2)
    assert outcome2 is not None and outcome2.status == "finished"
    assert handles.live_count() == 0
    assert "tab2" not in svc._active_tokens


# ---------------------------------------------------------------------------
# Terminal slot post-processing raises — tab cleared, handle settled failed
# ---------------------------------------------------------------------------


def test_on_analyze_finished_post_processing_raise_settles_failed(qapp):  # noqa: ARG001
    state = _make_state()
    svc, _ = _make_service(state, EventBus())
    handles = svc._handles

    token = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object()
    )

    # Make the service-side post-processing raise (writeback compute blows up).
    # _writeback is the MagicMock injected by _make_service.
    boom = RuntimeError("writeback boom")
    writeback = svc._writeback
    assert isinstance(writeback, MagicMock)
    writeback.compute_items_for_tab.side_effect = boom

    failed: list = []
    svc.analyze_failed.connect(lambda tid, err: failed.append((tid, err)))

    result = MagicMock()
    result.figure = None
    # Must not raise out of the slot (would crash Qt).
    svc._on_analyze_finished("tab1", result)

    # Tab cleared, handle settled failed, failure signal emitted — mirroring the
    # worker-side _failed path.
    assert state.get_tab("tab1").is_analyzing is False
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == "failed"
    assert outcome.error == str(boom)
    assert failed == [("tab1", boom)]
    assert handles.live_count() == 0
