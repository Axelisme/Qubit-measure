"""Unit tests for AnalyzeService.

Covers start_analyze, the FIT terminal paths, interactive analyze, and
cancel_interactive — using a real State + real EventBus, with bg mocked.

Stage 2c: AnalyzeService is now an OperationRunner client. _StagedAnalyzeService
no longer has a _submit helper; FIT/post paths use _submit_with_runner internally.
Tests drive terminal paths by capturing bg.last_on_done / on_error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import ContextReadiness
from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.main.services.analyze import AnalyzeService
from zcu_tools.gui.app.main.services.guard import AnalyzePermit
from zcu_tools.gui.app.main.state import ExpContext, Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.event_bus import EventMeta, EventOrigin
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from ._progress_fakes import DirectProgressTransport


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


class _FakeBg:
    """Synchronous background executor stub: captures callbacks for per-test driving."""

    def __init__(self, *, fail_submit: bool = False) -> None:
        self._fail_submit = fail_submit
        self.last_on_done: Callable[[Any], None] | None = None
        self.last_on_error: Callable[[Exception], None] | None = None
        self.submit_count = 0

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if self._fail_submit:
            raise RuntimeError("submit boom")
        self.submit_count += 1
        self.last_on_done = on_done
        self.last_on_error = on_error


def _make_service(
    state: State,
    bus: EventBus,
    *,
    fail_submit: bool = False,
) -> tuple[AnalyzeService, _FakeBg]:
    bg = _FakeBg(fail_submit=fail_submit)
    handles = OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    writeback = MagicMock()
    writeback.compute_items_for_tab.return_value = []
    runner = OperationRunner(MagicMock(), handles, progress, bg, bus)  # type: ignore[arg-type]
    svc = AnalyzeService(state, runner, bus, writeback, handles)
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

    assert bg.submit_count == 1
    assert state.get_tab("tab1").is_analyzing is True


def test_start_analyze_emits_interaction_event(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))

    svc, _ = _make_service(state, bus)
    svc.start_analyze(AnalyzePermit(tab_id="tab1"), analyze_params_instance=object())

    assert received == [TabInteractionFact.PRIMARY_ANALYZE_STARTED]


def test_start_analyze_submit_rejection_emits_restore_fact(qapp):  # noqa: ARG001
    state = _make_state()
    old_figure = MagicMock()
    old_post_figure = MagicMock()
    state.get_tab("tab1").figure = old_figure
    state.get_tab("tab1").post_figure = old_post_figure
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, _ = _make_service(state, bus, fail_submit=True)

    with pytest.raises(RuntimeError, match="submit boom"):
        svc.start_analyze(
            AnalyzePermit(tab_id="tab1"), analyze_params_instance=object()
        )

    assert received == [TabInteractionFact.PRIMARY_ANALYZE_START_REJECTED]
    assert state.get_tab("tab1").figure is old_figure
    assert state.get_tab("tab1").post_figure is old_post_figure
    assert state.get_tab("tab1").is_analyzing is False


def test_start_analyze_work_thunk_captures_figure_container(qapp):  # noqa: ARG001
    # The figure_container is captured in the work thunk's closure via
    # ``figure_ambient`` (ADR-0026 §2). Verify submit receives a single thunk
    # (no OffMainScopes arg).
    state = _make_state()
    svc, bg = _make_service(state, EventBus())
    container = MagicMock()

    svc.start_analyze(
        AnalyzePermit(tab_id="tab1"),
        analyze_params_instance=object(),
        figure_container=container,
    )

    assert bg.submit_count == 1  # submitted, work is a closure with figure_container


# ---------------------------------------------------------------------------
# start_analyze — busy tab rejection
# ---------------------------------------------------------------------------


def test_start_analyze_rejects_busy_tab(qapp):  # noqa: ARG001
    state = _make_state()
    state.set_tab_running("tab1", True)
    svc, _ = _make_service(state, EventBus())

    with pytest.raises(FailedPreconditionError, match="busy") as exc_info:
        svc.start_analyze(
            AnalyzePermit(tab_id="tab1"), analyze_params_instance=object()
        )

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""


# ---------------------------------------------------------------------------
# on_terminal (FIT) — finish / fail paths via bg callbacks
# ---------------------------------------------------------------------------


def test_on_analyze_finished_updates_state(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    svc, bg = _make_service(state, bus)

    token = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object()
    )

    fake_result = MagicMock()
    fake_result.figure = MagicMock()

    finished_signals: list = []
    svc.analyze_finished.connect(lambda tid, res: finished_signals.append((tid, res)))

    # Trigger the on_done path (runner calls on_terminal with ok=True)
    assert bg.last_on_done is not None
    bg.last_on_done(fake_result)

    assert state.get_tab("tab1").analyze_result is fake_result
    assert state.get_tab("tab1").is_analyzing is False
    assert len(finished_signals) == 1
    assert finished_signals[0] == ("tab1", fake_result)
    # Handle settles on terminal
    outcome = svc._handles.poll(token)
    assert outcome is not None and outcome.status == "finished"


def test_on_analyze_finished_emits_interaction_event(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, bg = _make_service(state, bus)

    svc.start_analyze(AnalyzePermit(tab_id="tab1"), analyze_params_instance=object())

    fake_result = MagicMock()
    fake_result.figure = None
    assert bg.last_on_done is not None
    bg.last_on_done(fake_result)

    assert received == [
        TabInteractionFact.PRIMARY_ANALYZE_STARTED,
        TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED,
    ]


# ---------------------------------------------------------------------------
# Interactive analysis (no worker; result produced on the user's Done)
# ---------------------------------------------------------------------------


def test_start_interactive_marks_analyzing_without_bg(qapp):  # noqa: ARG001
    state = _make_state()
    svc, bg = _make_service(state, EventBus())

    token = svc.start_interactive(AnalyzePermit(tab_id="tab1"))

    assert isinstance(token, int)
    assert state.get_tab("tab1").is_analyzing is True
    assert bg.submit_count == 0  # INTERACTIVE never starts a worker (main-thread)


def test_start_interactive_rejects_busy_tab(qapp):  # noqa: ARG001
    state = _make_state()
    state.set_tab_running("tab1", True)
    svc, _ = _make_service(state, EventBus())

    with pytest.raises(FailedPreconditionError, match="busy") as exc_info:
        svc.start_interactive(AnalyzePermit(tab_id="tab1"))

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""


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


def test_interactive_start_and_finish_keep_captured_operation_origin(
    qapp,
) -> None:  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    svc, _ = _make_service(state, bus)
    observed: list[tuple[TabInteractionFact, EventMeta]] = []
    bus.subscribe_with_meta(
        TabInteractionChangedPayload,
        lambda payload, meta: observed.append((payload.fact, meta)),
    )
    session = MagicMock()
    result = MagicMock()
    result.figure = None
    session.finish.return_value = result

    with bus.origin(EventOrigin(kind="agent", client_id="client-a")):
        token = svc.start_interactive(AnalyzePermit(tab_id="tab1"))
    svc.finish_interactive("tab1", session)

    assert [fact for fact, _meta in observed] == [
        TabInteractionFact.PRIMARY_ANALYZE_STARTED,
        TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED,
    ]
    assert [meta.origin for _fact, meta in observed] == [
        EventOrigin(kind="agent", client_id="client-a", operation_id=str(token)),
        EventOrigin(kind="agent", client_id="client-a", operation_id=str(token)),
    ]


# ---------------------------------------------------------------------------
# cancel_interactive — agent-side settle of an in-flight interactive picker
# ---------------------------------------------------------------------------


def test_cancel_interactive_clears_analyzing_and_settles_cancelled(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    svc, _ = _make_service(state, bus)
    handles = svc._handles

    token = svc.start_interactive(AnalyzePermit(tab_id="tab1"))
    assert state.get_tab("tab1").is_analyzing is True

    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    # A cancel is not a failure: analyze_failed must NOT fire (no error dialog).
    failed: list = []
    svc.analyze_failed.connect(lambda tid, err: failed.append((tid, err)))

    cancelled = svc.cancel_interactive("tab1")

    assert cancelled is True
    # is_analyzing cleared -> the tab is no longer busy, so it can be closed.
    assert state.get_tab("tab1").is_analyzing is False
    assert state.is_tab_busy("tab1") is False
    # The handle settles cancelled (the agent's analyze-poll resolves), exactly once.
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "cancelled"
    assert handles.live_count() == 0
    assert "tab1" not in svc._active_tokens
    # Interaction event fired; no failure signal.
    assert received == [TabInteractionFact.PRIMARY_ANALYZE_CANCELLED]
    assert failed == []


def test_cancel_interactive_no_inflight_is_graceful_noop(qapp):  # noqa: ARG001
    state = _make_state()
    svc, _ = _make_service(state, EventBus())

    # No analyze started at all -> graceful no-op, no raise, no state change.
    assert svc.cancel_interactive("tab1") is False
    assert state.get_tab("tab1").is_analyzing is False
    assert svc._handles.live_count() == 0


def test_cancel_interactive_does_not_touch_fit_analyze(qapp):  # noqa: ARG001
    state = _make_state()
    svc, bg = _make_service(state, EventBus())
    handles = svc._handles

    # A worker-backed FIT analyze is in flight (its callback will settle it later).
    token = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object()
    )
    assert bg.submit_count == 1

    # cancel_interactive must not reach into a FIT analyze (settling its handle
    # while the worker callback is still pending would double-settle on finish).
    assert svc.cancel_interactive("tab1") is False
    assert state.get_tab("tab1").is_analyzing is True  # FIT analyze untouched
    assert handles.poll(token) is None  # still live, settles via its own callback


def test_cancel_interactive_is_idempotent(qapp):  # noqa: ARG001
    state = _make_state()
    svc, _ = _make_service(state, EventBus())
    svc.start_interactive(AnalyzePermit(tab_id="tab1"))

    assert svc.cancel_interactive("tab1") is True
    # A second cancel finds nothing in flight -> graceful no-op.
    assert svc.cancel_interactive("tab1") is False


def test_finish_after_cancel_interactive_is_inert(qapp):  # noqa: ARG001
    # Defensive: if a late Done arrives after a cancel, the already-settled token
    # is gone, so finish_interactive must not resurrect the tab into analyzing.
    state = _make_state()
    svc, _ = _make_service(state, EventBus())
    svc.start_interactive(AnalyzePermit(tab_id="tab1"))
    svc.cancel_interactive("tab1")

    session = MagicMock()
    result = MagicMock()
    result.figure = None
    session.finish.return_value = result
    svc.finish_interactive("tab1", session)

    # State was updated by the (inert) finish path but the tab is not analyzing and
    # no second handle leaked.
    assert state.get_tab("tab1").is_analyzing is False
    assert svc._handles.live_count() == 0


# ---------------------------------------------------------------------------
# Background analyze failure
# ---------------------------------------------------------------------------


def test_background_analyze_failure_resets_state(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    svc, bg = _make_service(state, bus)

    token = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object()
    )
    assert state.get_tab("tab1").is_analyzing is True

    failed_signals: list = []
    svc.analyze_failed.connect(lambda tid, err: failed_signals.append((tid, err)))

    error = RuntimeError("analysis failed")
    assert bg.last_on_error is not None
    bg.last_on_error(error)

    assert state.get_tab("tab1").is_analyzing is False
    assert len(failed_signals) == 1
    outcome = svc._handles.poll(token)
    assert outcome is not None and outcome.status == "failed"


def test_background_analyze_failure_emits_interaction_event(qapp):  # noqa: ARG001
    state = _make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, bg = _make_service(state, bus)

    svc.start_analyze(AnalyzePermit(tab_id="tab1"), analyze_params_instance=object())
    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("oops"))

    assert received == [
        TabInteractionFact.PRIMARY_ANALYZE_STARTED,
        TabInteractionFact.PRIMARY_ANALYZE_FAILED,
    ]


# ---------------------------------------------------------------------------
# Concurrent tabs — no exclusion gate (ADR-0019): each settles its own token
# ---------------------------------------------------------------------------


def test_two_tabs_settle_their_own_tokens(qapp):  # noqa: ARG001
    state = _make_two_tab_state()
    svc, bg = _make_service(state, EventBus())
    handles = svc._handles

    token1 = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object()
    )
    # Save tab1's on_done before tab2 overwrites it in the bg stub
    on_done_1 = bg.last_on_done

    token2 = svc.start_analyze(
        AnalyzePermit(tab_id="tab2"), analyze_params_instance=object()
    )
    on_done_2 = bg.last_on_done

    # Two distinct live handles — the second start must not clobber the first.
    assert token1 != token2
    assert handles.poll(token1) is None  # still pending
    assert handles.poll(token2) is None
    assert handles.live_count() == 2

    # Finish tab1 only: its token settles, tab2's stays live.
    r1 = MagicMock()
    r1.figure = None
    assert on_done_1 is not None
    on_done_1(r1)

    outcome1 = handles.poll(token1)
    assert outcome1 is not None and outcome1.status == "finished"
    assert handles.poll(token2) is None  # tab2 untouched
    assert state.get_tab("tab1").is_analyzing is False
    assert state.get_tab("tab2").is_analyzing is True
    assert handles.live_count() == 1

    # Finish tab2: its own (later) token settles.
    r2 = MagicMock()
    r2.figure = None
    assert on_done_2 is not None
    on_done_2(r2)

    outcome2 = handles.poll(token2)
    assert outcome2 is not None and outcome2.status == "finished"
    assert handles.live_count() == 0
    assert "tab2" not in svc._active_tokens


# ---------------------------------------------------------------------------
# Terminal slot post-processing raises — tab cleared, handle settled failed
# ---------------------------------------------------------------------------


def test_on_analyze_finished_post_processing_raise_settles_failed(qapp):  # noqa: ARG001
    state = _make_state()
    svc, bg = _make_service(state, EventBus())
    handles = svc._handles

    token = svc.start_analyze(
        AnalyzePermit(tab_id="tab1"), analyze_params_instance=object()
    )

    # Make the service-side post-processing raise (writeback compute blows up).
    boom = RuntimeError("writeback boom")
    writeback = svc._writeback
    assert isinstance(writeback, MagicMock)
    writeback.compute_items_for_tab.side_effect = boom

    failed: list = []
    svc.analyze_failed.connect(lambda tid, err: failed.append((tid, err)))

    result = MagicMock()
    result.figure = None
    # Must not raise out of the slot (would crash Qt).
    assert bg.last_on_done is not None
    bg.last_on_done(result)

    # Tab cleared, handle settled failed, failure signal emitted — mirroring the
    # worker-side _failed path.
    assert state.get_tab("tab1").is_analyzing is False
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == "failed"
    assert outcome.error == str(boom)
    assert failed == [("tab1", boom)]
    assert handles.live_count() == 0
