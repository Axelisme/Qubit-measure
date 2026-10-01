"""Unit tests for PostAnalyzeService.

Mirrors tests/gui/app/measure/services/test_analyze.py: a real State + real EventBus, with
BackgroundRunner mocked. Covers the gate (no primary analyze result), the
submit-to-bg path, and the finished/failed terminal paths.

Stage 2c: PostAnalyzeService uses _submit_with_runner. Tests drive terminal
paths via bg.last_on_done / on_error (same pattern as test_analyze.py).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.gui.app.measure.adapter import ContextReadiness
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.services.post_analyze import PostAnalyzeService
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from tests.gui._completion_helpers import on_post_analyze_failed
from tests.gui._progress_fakes import DirectProgressTransport


def _plots(figure: Figure | None = None) -> Plots:
    plots = Plots(NonPresentingHost())
    if figure is not None:
        plots.adopt("fit", figure)
    return plots


def _make_state(tab_id: str = "tab1", *, with_analyze: bool = True) -> State:
    ctx = SessionEnv(
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
        Session(adapter_name="fake", adapter=MagicMock(), cfg=MagicMock()),
    )
    state.update_tab_result(tab_id, object())
    if with_analyze:
        state.update_tab_analyze(tab_id, object(), None)
    return state


class _FakeBg:
    """Synchronous background executor stub: captures callbacks for per-test driving."""

    def __init__(self, *, fail_submit: bool = False) -> None:
        self._fail_submit = fail_submit
        self.last_work: Callable[[], Any] | None = None
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
        self.last_work = work
        self.last_on_done = on_done
        self.last_on_error = on_error


def _make_service(
    state: State, bus: EventBus, *, fail_submit: bool = False
) -> tuple[PostAnalyzeService, _FakeBg]:
    bg = _FakeBg(fail_submit=fail_submit)
    handles = OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    runner = OperationRunner(MagicMock(), handles, progress, bg, bus)  # type: ignore[arg-type]
    writeback = MagicMock()
    writeback.create_draft.return_value = None
    svc = PostAnalyzeService(state, runner, bus, handles, writeback)
    return svc, bg


def _make_two_tab_state() -> State:
    state = _make_state("tab1")
    state.add_tab(
        "tab2",
        Session(adapter_name="fake", adapter=MagicMock(), cfg=MagicMock()),
    )
    state.update_tab_result("tab2", object())
    state.update_tab_analyze("tab2", object(), None)
    return state


def test_start_post_analyze_submits_to_bg(qapp):
    state = _make_state()
    svc, bg = _make_service(state, EventBus())

    svc.start_post_analyze(
        "tab1", post_analyze_params_instance=object(), plots=_plots()
    )

    assert bg.submit_count == 1
    assert state.get_tab("tab1").is_analyzing is True


def test_start_post_analyze_emits_interaction_event(qapp):
    state = _make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))

    svc, _ = _make_service(state, bus)
    svc.start_post_analyze(
        "tab1", post_analyze_params_instance=object(), plots=_plots()
    )

    assert received == [TabInteractionFact.POST_ANALYZE_STARTED]


def test_start_post_analyze_submit_rejection_preserves_figures(qapp):
    state = _make_state()
    old_primary = _plots(Figure())
    old_primary.finish()
    old_post = _plots(Figure())
    old_post.finish()
    state.update_tab_analyze("tab1", object(), old_primary)
    state.update_tab_post_analyze("tab1", object(), old_post)
    tab = state.get_tab("tab1")
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, _ = _make_service(state, bus, fail_submit=True)

    with pytest.raises(RuntimeError, match="submit boom"):
        svc.start_post_analyze(
            "tab1", post_analyze_params_instance=object(), plots=_plots()
        )

    assert received == [TabInteractionFact.POST_ANALYZE_START_REJECTED]
    assert tab.analysis.plots is old_primary
    assert tab.post_analysis.plots is old_post
    assert tab.is_analyzing is False


def test_start_post_analyze_gates_on_missing_primary_result(qapp):
    state = _make_state(with_analyze=False)
    svc, bg = _make_service(state, EventBus())

    with pytest.raises(
        FailedPreconditionError, match="no primary analyze result"
    ) as exc_info:
        svc.start_post_analyze(
            "tab1", post_analyze_params_instance=object(), plots=_plots()
        )
    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""
    assert bg.submit_count == 0


def test_start_post_analyze_rejects_busy_tab(qapp):
    state = _make_state()
    state.set_tab_running("tab1", True)
    svc, _ = _make_service(state, EventBus())

    with pytest.raises(FailedPreconditionError, match="busy") as exc_info:
        svc.start_post_analyze(
            "tab1", post_analyze_params_instance=object(), plots=_plots()
        )

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""


def test_post_worker_receives_explicit_operation_plots(qapp):
    state = _make_state()
    svc, bg = _make_service(state, EventBus())
    plots = _plots()

    svc.start_post_analyze("tab1", post_analyze_params_instance=object(), plots=plots)

    assert bg.submit_count == 1
    assert bg.last_work is not None
    result = bg.last_work()
    adapter = state.get_tab("tab1").adapter
    adapter.post_analyze.assert_called_once()
    assert adapter.post_analyze.call_args.kwargs == {"plots": plots}
    assert result is adapter.post_analyze.return_value


def test_on_post_analyze_finished_updates_state(qapp):
    state = _make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, bg = _make_service(state, bus)

    figure = Figure()
    plots = _plots(figure)
    token = svc.start_post_analyze(
        "tab1", post_analyze_params_instance=object(), plots=plots
    )

    post_result = object()

    finished: list = []
    bus.subscribe(
        TabInteractionChangedPayload,
        lambda payload: (
            finished.append(
                (payload.tab_id, state.get_tab(payload.tab_id).post_analysis.result)
            )
            if payload.fact is TabInteractionFact.POST_ANALYZE_SUCCEEDED
            else None
        ),
    )

    assert bg.last_on_done is not None
    bg.last_on_done(post_result)

    tab = state.get_tab("tab1")
    assert tab.post_analysis.result is post_result
    assert tab.post_analysis.plots is plots
    assert tab.post_analysis.plots["fit"] is figure
    assert tab.is_analyzing is False
    assert finished == [("tab1", post_result)]
    outcome = svc._handles.poll(token)
    assert outcome is not None and outcome.status == "finished"
    assert received == [
        TabInteractionFact.POST_ANALYZE_STARTED,
        TabInteractionFact.POST_ANALYZE_SUCCEEDED,
    ]


def test_on_post_analyze_failed_resets_state(qapp):
    state = _make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, bg = _make_service(state, bus)

    token = svc.start_post_analyze(
        "tab1", post_analyze_params_instance=object(), plots=_plots()
    )

    failed: list = []
    on_post_analyze_failed(svc, lambda tid, err: failed.append((tid, err)))

    error = RuntimeError("post analysis failed")
    assert bg.last_on_error is not None
    bg.last_on_error(error)

    assert state.get_tab("tab1").is_analyzing is False
    assert len(failed) == 1
    outcome = svc._handles.poll(token)
    assert outcome is not None and outcome.status == "failed"
    assert received == [
        TabInteractionFact.POST_ANALYZE_STARTED,
        TabInteractionFact.POST_ANALYZE_FAILED,
    ]


# ---------------------------------------------------------------------------
# Concurrent tabs — no exclusion gate (ADR-0066): each settles its own token
# ---------------------------------------------------------------------------


def test_two_tabs_settle_their_own_tokens(qapp):
    state = _make_two_tab_state()
    svc, bg = _make_service(state, EventBus())
    handles = svc._handles

    token1 = svc.start_post_analyze(
        "tab1", post_analyze_params_instance=object(), plots=_plots()
    )
    on_done_1 = bg.last_on_done

    token2 = svc.start_post_analyze(
        "tab2", post_analyze_params_instance=object(), plots=_plots()
    )
    on_done_2 = bg.last_on_done

    assert token1 != token2
    assert handles.poll(token1) is None
    assert handles.poll(token2) is None
    assert handles.live_count() == 2

    r1 = object()
    assert on_done_1 is not None
    on_done_1(r1)

    outcome1 = handles.poll(token1)
    assert outcome1 is not None and outcome1.status == "finished"
    assert handles.poll(token2) is None  # tab2 untouched
    assert state.get_tab("tab1").is_analyzing is False
    assert state.get_tab("tab2").is_analyzing is True
    assert handles.live_count() == 1

    r2 = object()
    assert on_done_2 is not None
    on_done_2(r2)

    outcome2 = handles.poll(token2)
    assert outcome2 is not None and outcome2.status == "finished"
    assert handles.live_count() == 0
    assert "tab2" not in svc._active_tokens


# ---------------------------------------------------------------------------
# Terminal slot post-processing raises — tab cleared, handle settled failed
# ---------------------------------------------------------------------------


def test_on_post_analyze_finished_post_processing_raise_settles_failed(
    qapp, monkeypatch
):
    state = _make_state()
    svc, bg = _make_service(state, EventBus())
    handles = svc._handles

    token = svc.start_post_analyze(
        "tab1", post_analyze_params_instance=object(), plots=_plots()
    )

    # Make the State recording raise (post_analyze primary result vanished, etc.).
    boom = RuntimeError("update boom")
    monkeypatch.setattr(state, "update_tab_post_analyze", MagicMock(side_effect=boom))

    failed: list = []
    on_post_analyze_failed(svc, lambda tid, err: failed.append((tid, err)))

    post_result = object()
    # Must not raise out of the slot (would crash Qt).
    assert bg.last_on_done is not None
    bg.last_on_done(post_result)

    assert state.get_tab("tab1").is_analyzing is False
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == "failed"
    assert outcome.error == str(boom)
    assert failed == [("tab1", str(boom))]
    assert handles.live_count() == 0
