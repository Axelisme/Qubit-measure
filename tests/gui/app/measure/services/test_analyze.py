"""Background FIT analysis dispatch, captured inputs and terminal settlement."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import zcu_tools.experiment as experiment
from matplotlib.figure import Figure
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import ExpectedErrorCategory, FailedPreconditionError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.entry.schema import PointDocument

from tests.gui._completion_helpers import on_analyze_failed
from tests.gui.app.measure._components import QubitSchema
from tests.gui.app.measure.services._analyze_support import (
    make_plots,
    make_service,
    make_state,
    make_two_tab_state,
)
from tests.gui.session.ui._parameter_fakes import make_parameters


@pytest.mark.parametrize("phase", ["work", "record"])
@pytest.mark.parametrize("expected", [True, False])
def test_only_expected_analysis_work_failure_is_classified(qapp, phase, expected):
    state = make_state()
    adapter = MagicMock()
    state.get_tab("tab1").adapter = adapter
    error = (
        experiment.AnalysisFailed("insufficient fitting data")
        if expected
        else RuntimeError("broken analysis")
    )
    if phase == "work":
        adapter.analyze.side_effect = error
    else:
        adapter.get_writeback_items.side_effect = error
    handles = OperationHandles()
    svc, bg = make_service(state, EventBus(), handles=handles)
    token = svc.start_analyze(
        "tab1", analyze_params_instance=object(), plots=make_plots()
    )
    assert bg.last_work is not None
    assert bg.last_on_error is not None and bg.last_on_done is not None
    if phase == "work":
        with pytest.raises(type(error)) as failure:
            bg.last_work()
        bg.last_on_error(failure.value)
    else:
        bg.last_on_done(bg.last_work())
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "failed"
    assert outcome.error == str(error)
    assert outcome.failure_kind == (
        "analysis" if expected and phase == "work" else None
    )


@pytest.mark.parametrize("has_run_source", [True, False])
def test_historical_analyze_uses_run_capture_after_alias_and_parameter_edits(
    qapp, has_run_source: bool
) -> None:
    original = make_parameters({})
    point = PointDocument.model_construct(
        format="zcu.parameter-container",
        format_version="2.0",
        general=original.document().general,
        components={
            "Q1": QubitSchema(kind="qubit/fluxonium", freq=4200.0),
            "Q2": QubitSchema(kind="qubit/fluxonium", freq=5100.0),
        },
    )
    captured = ParameterContext(
        point=point, library=original.library, target=original.target, roles=None
    ).bind_aliases((AliasRow("qubit", "Q1"),), {})
    state = make_state()
    state.update_tab_result(
        "tab1",
        RunRecord(None, object()),
        parameters=captured if has_run_source else None,
    )
    point.components["Q1"] = QubitSchema(kind="qubit/fluxonium", freq=4300.0)
    current = ParameterContext(
        point=point, library=original.library, target=original.target, roles=None
    ).bind_aliases((AliasRow("qubit", "Q2"),), {})
    state.session_env = replace(state.session_env, parameters=current)
    adapter = MagicMock()
    state.get_tab("tab1").adapter = adapter
    svc, bg = make_service(state, EventBus())
    svc.start_analyze("tab1", analyze_params_instance=object(), plots=make_plots())
    assert bg.last_work is not None
    bg.last_work()
    request = adapter.analyze.call_args.args[0]
    if has_run_source:
        assert request.parameters.value("$qubit.freq") == 4200.0
        assert current.value("$qubit.freq") == 5100.0
    else:
        assert request.parameters is None


def test_start_analyze_submits_to_bg(qapp):
    state = make_state()
    bus = EventBus()
    svc, bg = make_service(state, bus)

    svc.start_analyze("tab1", analyze_params_instance=object(), plots=make_plots())

    assert bg.submit_count == 1
    assert state.get_tab("tab1").is_analyzing is True


def test_start_analyze_emits_interaction_event(qapp):
    state = make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))

    svc, _ = make_service(state, bus)
    svc.start_analyze("tab1", analyze_params_instance=object(), plots=make_plots())

    assert received == [TabInteractionFact.PRIMARY_ANALYZE_STARTED]


def test_start_analyze_submit_rejection_emits_restore_fact(qapp):
    state = make_state()
    old_plots = make_plots()
    old_plots.adopt("fit", Figure())
    old_plots.finish()
    old_post_plots = make_plots()
    old_post_plots.adopt("fit", Figure())
    old_post_plots.finish()
    state.update_tab_analyze("tab1", object(), old_plots)
    state.update_tab_post_analyze("tab1", object(), old_post_plots)
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, _ = make_service(state, bus, fail_submit=True)

    with pytest.raises(RuntimeError, match="submit boom"):
        svc.start_analyze(
            "tab1",
            analyze_params_instance=object(),
            plots=make_plots(),
        )

    assert received == [TabInteractionFact.PRIMARY_ANALYZE_START_REJECTED]
    assert state.get_tab("tab1").analysis.plots is old_plots
    assert state.get_tab("tab1").post_analysis.plots is old_post_plots
    assert state.get_tab("tab1").is_analyzing is False


def test_start_analyze_passes_operation_plots_to_worker_adapter(qapp):
    state = make_state()
    svc, bg = make_service(state, EventBus())
    plots = make_plots()

    svc.start_analyze("tab1", analyze_params_instance=object(), plots=plots)

    assert bg.submit_count == 1
    assert bg.last_work is not None
    result = bg.last_work()
    adapter = state.get_tab("tab1").adapter
    assert isinstance(adapter, MagicMock)
    adapter.analyze.assert_called_once()
    assert adapter.analyze.call_args.kwargs == {"plots": plots}
    assert result is adapter.analyze.return_value


def test_start_analyze_rejects_busy_tab(qapp):
    state = make_state()
    state.set_tab_running("tab1", running=True)
    svc, _ = make_service(state, EventBus())

    with pytest.raises(FailedPreconditionError, match="busy") as exc_info:
        svc.start_analyze(
            "tab1",
            analyze_params_instance=object(),
            plots=make_plots(),
        )

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""


def test_finished_analysis_keeps_captured_inputs_across_later_edits(qapp):
    state = make_state()
    svc, bg = make_service(state, EventBus())
    supplied = {"threshold": 0.3}
    token = svc.start_analyze(
        "tab1", analyze_params_instance=supplied, plots=make_plots()
    )
    state.update_tab_analyze_param_instance("tab1", {"threshold": 0.7})
    result = MagicMock()
    result.figure = None
    assert bg.last_on_done is not None
    bg.last_on_done(result)
    supplied["threshold"] = 0.5
    state.update_tab_analyze_param_instance("tab1", {"threshold": 0.9})
    pane = state.get_tab("tab1").analysis
    assert pane.result is result
    assert pane.source_operation_id == token
    assert pane.result_params == {"threshold": 0.3}
    assert pane.params == {"threshold": 0.9}


def test_on_analyze_finished_updates_state(qapp):
    state = make_state()
    bus = EventBus()
    handles = OperationHandles()
    svc, bg = make_service(state, bus, handles=handles)

    token = svc.start_analyze(
        "tab1", analyze_params_instance=object(), plots=make_plots()
    )

    fake_result = object()

    finished_signals: list[tuple[str, object]] = []
    bus.subscribe(
        TabInteractionChangedPayload,
        lambda payload: (
            finished_signals.append(
                (payload.tab_id, state.get_tab(payload.tab_id).analysis.result)
            )
            if payload.fact is TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED
            else None
        ),
    )

    # Trigger the on_done path (runner calls on_terminal with ok=True)
    assert bg.last_on_done is not None
    bg.last_on_done(fake_result)

    assert state.get_tab("tab1").analysis.result is fake_result
    assert state.get_tab("tab1").analysis.source_operation_id == token
    assert state.get_tab("tab1").is_analyzing is False
    assert len(finished_signals) == 1
    assert finished_signals[0] == ("tab1", fake_result)
    # Handle settles on terminal
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished"


def test_on_analyze_finished_emits_interaction_event(qapp):
    state = make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, bg = make_service(state, bus)

    svc.start_analyze("tab1", analyze_params_instance=object(), plots=make_plots())

    fake_result = object()
    assert bg.last_on_done is not None
    bg.last_on_done(fake_result)

    assert received == [
        TabInteractionFact.PRIMARY_ANALYZE_STARTED,
        TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED,
    ]


def test_background_analyze_failure_resets_state(qapp):
    state = make_state()
    bus = EventBus()
    handles = OperationHandles()
    svc, bg = make_service(state, bus, handles=handles)

    token = svc.start_analyze(
        "tab1", analyze_params_instance=object(), plots=make_plots()
    )
    assert state.get_tab("tab1").is_analyzing is True

    failed_signals: list[tuple[str, str]] = []
    on_analyze_failed(svc, lambda tid, err: failed_signals.append((tid, err)))

    error = RuntimeError("analysis failed")
    assert bg.last_on_error is not None
    bg.last_on_error(error)

    assert state.get_tab("tab1").is_analyzing is False
    assert len(failed_signals) == 1
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "failed"


def test_background_analyze_failure_emits_interaction_event(qapp):
    state = make_state()
    bus = EventBus()
    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    svc, bg = make_service(state, bus)

    svc.start_analyze("tab1", analyze_params_instance=object(), plots=make_plots())
    assert bg.last_on_error is not None
    bg.last_on_error(RuntimeError("oops"))

    assert received == [
        TabInteractionFact.PRIMARY_ANALYZE_STARTED,
        TabInteractionFact.PRIMARY_ANALYZE_FAILED,
    ]


def test_two_tabs_settle_their_own_tokens(qapp):
    state = make_two_tab_state()
    handles = OperationHandles()
    svc, bg = make_service(state, EventBus(), handles=handles)

    token1 = svc.start_analyze(
        "tab1", analyze_params_instance=object(), plots=make_plots()
    )
    # Save tab1's on_done before tab2 overwrites it in the bg stub
    on_done_1 = bg.last_on_done

    token2 = svc.start_analyze(
        "tab2", analyze_params_instance=object(), plots=make_plots()
    )
    on_done_2 = bg.last_on_done

    # Two distinct live handles — the second start must not clobber the first.
    assert token1 != token2
    assert handles.poll(token1) is None  # still pending
    assert handles.poll(token2) is None
    assert handles.live_count() == 2

    # Finish tab1 only: its token settles, tab2's stays live.
    r1 = object()
    assert on_done_1 is not None
    on_done_1(r1)

    outcome1 = handles.poll(token1)
    assert outcome1 is not None and outcome1.status == "finished"
    assert handles.poll(token2) is None  # tab2 untouched
    assert state.get_tab("tab1").is_analyzing is False
    assert state.get_tab("tab2").is_analyzing is True
    assert handles.live_count() == 1

    # Finish tab2: its own (later) token settles.
    r2 = object()
    assert on_done_2 is not None
    on_done_2(r2)

    outcome2 = handles.poll(token2)
    assert outcome2 is not None and outcome2.status == "finished"
    assert handles.live_count() == 0
    assert "tab2" not in dict(svc.active_operations())


def test_on_analyze_finished_post_processing_raise_settles_failed(qapp):
    state = make_state()
    handles = OperationHandles()
    writeback = MagicMock()
    svc, bg = make_service(state, EventBus(), handles=handles, writeback=writeback)

    token = svc.start_analyze(
        "tab1", analyze_params_instance=object(), plots=make_plots()
    )

    # Make the service-side post-processing raise (writeback compute blows up).
    boom = RuntimeError("writeback boom")
    writeback.create_draft.side_effect = boom

    failed: list[tuple[str, str]] = []
    on_analyze_failed(svc, lambda tid, err: failed.append((tid, err)))

    result = object()
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
    assert failed == [("tab1", str(boom))]
    assert handles.live_count() == 0
