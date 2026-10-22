"""Interactive plugin lifecycle, cancellation and per-tab operation isolation."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.event_bus import EventMeta, EventOrigin
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.interactive import Action, PluginDefinition
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.operation_handles import OperationHandles

from tests.gui._completion_helpers import on_analyze_failed
from tests.gui.app.measure._record_support import require_record
from tests.gui.app.measure.services._analyze_support import (
    make_plots,
    make_service,
    make_state,
    make_two_tab_state,
)
from tests.gui.session.ui._parameter_fakes import make_parameters


def test_framework_session_drives_committed_result_and_one_terminal_operation(qapp):
    state = make_state()
    captured = make_parameters({"rate": 1.0})
    state.update_tab_result(
        "tab1", require_record(state.get_tab("tab1").run), parameters=captured
    )
    state.set_context(
        replace(state.session_env, parameters=make_parameters({"rate": 2.0}))
    )
    handles = OperationHandles()
    svc, bg = make_service(state, EventBus(), handles=handles)

    def require_pick(position: int) -> None:
        if position == 0:
            raise FailedPreconditionError("pick a line")

    plugin = PluginDefinition(
        "pick",
        0,
        (),
        require_pick,
        lambda position: MagicMock(position=position),
    )

    token = svc.start_plugin(
        "tab1",
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=make_plots(),
    )
    active = svc.get_interactive("tab1")
    assert active is not None and active.plugin is plugin
    Action[int, int](lambda current, position: position).execute(active.session, 3)
    assert active.session.snapshot() == 3
    assert bg.submit_count == 0
    state.set_context(
        replace(state.session_env, parameters=make_parameters({"rate": 3.0}))
    )

    svc.finish_plugin("tab1")
    adapter = state.get_tab("tab1").adapter
    assert isinstance(adapter, MagicMock)
    assert adapter.get_writeback_items.call_args.args[0].ctx.parameters is captured
    result = state.get_tab("tab1").analysis.result
    assert result is not None
    assert result.position == 3
    assert state.get_tab("tab1").is_analyzing is False
    assert svc.get_interactive("tab1") is None
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished"
    svc.finish_plugin("tab1")
    assert state.get_tab("tab1").analysis.result is result


def test_unavailable_finish_keeps_session_active_and_cancel_discards_late_done(qapp):
    state = make_state()
    handles = OperationHandles()
    svc, _ = make_service(state, EventBus(), handles=handles)

    def require_pick(position: int) -> None:
        if position == 0:
            raise FailedPreconditionError("pick a line")

    plugin = PluginDefinition("pick", 0, (), require_pick, lambda position: MagicMock())
    token = svc.start_plugin(
        "tab1",
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=make_plots(),
    )
    active = svc.get_interactive("tab1")
    assert active is not None
    with pytest.raises(FailedPreconditionError, match="pick a line"):
        svc.finish_plugin("tab1")
    assert svc.get_interactive("tab1") is active
    assert state.get_tab("tab1").is_analyzing is True

    assert svc.cancel_interactive("tab1") is True
    svc.finish_plugin("tab1")
    assert state.get_tab("tab1").analysis.result is None
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "cancelled"
    with pytest.raises(FailedPreconditionError):
        active.session.snapshot()


def test_result_build_failure_settles_failed_without_reopening_input(qapp):
    state = make_state()
    handles = OperationHandles()
    svc, _ = make_service(state, EventBus(), handles=handles)

    def fail_result(position: int) -> object:
        raise RuntimeError("cannot build result")

    plugin = PluginDefinition("pick", 1, (), lambda position: None, fail_result)
    token = svc.start_plugin(
        "tab1",
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=make_plots(),
    )
    active = svc.get_interactive("tab1")
    assert active is not None
    svc.finish_plugin("tab1")
    assert state.get_tab("tab1").is_analyzing is False
    assert state.get_tab("tab1").analysis.result is None
    assert svc.get_interactive("tab1") is None
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "failed"
    with pytest.raises(FailedPreconditionError):
        active.session.commit(lambda current: 2)


def test_plugin_record_failure_settles_failed_and_keeps_previous_result(qapp):
    state = make_state()
    previous = MagicMock()
    previous_plots = make_plots()
    previous_plots.adopt("fit", Figure())
    previous_plots.finish()
    state.update_tab_analyze("tab1", previous, previous_plots, source_operation_id=900)
    handles = OperationHandles()
    writeback = MagicMock()
    bus = EventBus()
    svc, _ = make_service(state, bus, handles=handles, writeback=writeback)
    plugin = PluginDefinition(
        "pick",
        2,
        (),
        lambda _state: None,
        lambda value: MagicMock(value=value),
    )
    token = svc.start_plugin(
        "tab1",
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=make_plots(),
    )
    active = svc.get_interactive("tab1")
    assert active is not None
    writeback.create_draft.side_effect = RuntimeError("writeback boom")

    assert svc.finish_plugin("tab1") is True

    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "failed"
    assert state.get_tab("tab1").analysis.result is previous
    assert state.get_tab("tab1").analysis.source_operation_id == 900
    assert state.get_tab("tab1").analysis.plots is previous_plots
    assert not state.get_tab("tab1").is_analyzing
    assert svc.get_interactive("tab1") is None
    with pytest.raises(FailedPreconditionError):
        active.session.snapshot()


def test_concurrent_plugin_tabs_keep_independent_sessions_and_handles(qapp):
    state = make_two_tab_state()
    handles = OperationHandles()
    svc, _ = make_service(state, EventBus(), handles=handles)
    plugin = PluginDefinition(
        "pick",
        0,
        (),
        lambda _state: None,
        lambda position: MagicMock(position=position),
    )
    owner = ManualOwnerScheduler()
    first = svc.start_plugin(
        "tab1",
        plugin,
        owner,
        analyze_params_instance="submitted",
        plots=make_plots(),
    )
    second = svc.start_plugin(
        "tab2",
        plugin,
        owner,
        analyze_params_instance="submitted",
        plots=make_plots(),
    )
    tab1 = svc.get_interactive("tab1")
    tab2 = svc.get_interactive("tab2")
    assert tab1 is not None and tab2 is not None
    Action[int, int](lambda _current, position: position).execute(tab1.session, 5)
    assert tab2.session.snapshot() == 0
    assert svc.cancel_interactive("tab1") is True
    first_outcome = handles.poll(first)
    assert first_outcome is not None and first_outcome.status == "cancelled"
    assert handles.poll(second) is None
    assert svc.get_interactive("tab2") is tab2
    Action[int, int](lambda _current, position: position).execute(tab2.session, 8)
    assert svc.finish_plugin("tab2") is True
    result = state.get_tab("tab2").analysis.result
    assert result is not None and result.position == 8
    assert state.get_tab("tab1").analysis.result is None
    second_outcome = handles.poll(second)
    assert second_outcome is not None and second_outcome.status == "finished"


def test_start_plugin_rejects_busy_tab(qapp):
    state = make_state()
    state.set_tab_running("tab1", running=True)
    svc, _ = make_service(state, EventBus())
    plugin = PluginDefinition("pick", 0, (), lambda _state: None, lambda state: state)

    with pytest.raises(FailedPreconditionError, match="busy"):
        svc.start_plugin(
            "tab1",
            plugin,
            ManualOwnerScheduler(),
            analyze_params_instance="submitted",
            plots=make_plots(),
        )
    assert svc.get_interactive("tab1") is None


def test_interactive_start_and_finish_keep_captured_operation_origin(
    qapp,
) -> None:
    state = make_state()
    bus = EventBus()
    svc, _ = make_service(state, bus)
    observed: list[tuple[TabInteractionFact, EventMeta]] = []
    bus.subscribe_with_meta(
        TabInteractionChangedPayload,
        lambda payload, meta: observed.append((payload.fact, meta)),
    )
    plugin = PluginDefinition(
        "pick", 2, (), lambda _state: None, lambda state: MagicMock()
    )
    with bus.origin(EventOrigin(kind="agent", client_id="client-a")):
        token = svc.start_plugin(
            "tab1",
            plugin,
            ManualOwnerScheduler(),
            analyze_params_instance="submitted",
            plots=make_plots(),
        )
    svc.finish_plugin("tab1")

    assert [fact for fact, _meta in observed] == [
        TabInteractionFact.PRIMARY_ANALYZE_STARTED,
        TabInteractionFact.PRIMARY_ANALYZE_SUCCEEDED,
    ]
    assert [meta.origin for _fact, meta in observed] == [
        EventOrigin(kind="agent", client_id="client-a", operation_id=str(token)),
        EventOrigin(kind="agent", client_id="client-a", operation_id=str(token)),
    ]


def test_cancel_interactive_clears_analyzing_and_settles_cancelled(qapp):
    state = make_state()
    bus = EventBus()
    svc, _ = make_service(state, bus)
    handles = OperationHandles()
    svc, _ = make_service(state, bus, handles=handles)
    plugin = PluginDefinition(
        "pick", 0, (), lambda _state: None, lambda state: MagicMock()
    )
    token = svc.start_plugin(
        "tab1",
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=make_plots(),
    )
    assert state.get_tab("tab1").is_analyzing is True

    received: list[TabInteractionFact] = []
    bus.subscribe(TabInteractionChangedPayload, lambda p: received.append(p.fact))
    # A cancel is not a failure: analyze_failed must NOT fire (no error dialog).
    failed: list[tuple[str, str]] = []
    on_analyze_failed(svc, lambda tid, err: failed.append((tid, err)))

    cancelled = svc.cancel_interactive("tab1")

    assert cancelled is True
    # is_analyzing cleared -> the tab is no longer busy, so it can be closed.
    assert state.get_tab("tab1").is_analyzing is False
    assert state.is_tab_busy("tab1") is False
    # The handle settles cancelled (the agent's analyze-poll resolves), exactly once.
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "cancelled"
    assert handles.live_count() == 0
    assert "tab1" not in dict(svc.active_operations())
    # Interaction event fired; no failure signal.
    assert received == [TabInteractionFact.PRIMARY_ANALYZE_CANCELLED]
    assert failed == []


def test_cancelled_interaction_retains_previous_pane_operation_sources(qapp):
    state = make_state()
    primary = MagicMock(figure=None)
    post = MagicMock(figure=None)
    state.update_tab_analyze("tab1", primary, None, source_operation_id=900)
    state.update_tab_post_analyze("tab1", post, None, source_operation_id=901)
    handles = OperationHandles()
    svc, _ = make_service(state, EventBus(), handles=handles)
    plugin = PluginDefinition(
        "pick", 2, (), lambda _state: None, lambda _value: MagicMock(figure=None)
    )
    token = svc.start_plugin(
        "tab1",
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance={},
        plots=make_plots(),
    )
    assert svc.get_interactive_operation("tab1") == token
    assert svc.cancel_interactive("tab1") is True
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "cancelled"
    assert state.get_tab("tab1").analysis.source_operation_id == 900
    assert state.get_tab("tab1").post_analysis.source_operation_id == 901
    assert svc.get_interactive_operation("tab1") is None


def test_cancel_interactive_no_inflight_is_graceful_noop(qapp):
    state = make_state()
    handles = OperationHandles()
    svc, _ = make_service(state, EventBus(), handles=handles)

    # No analyze started at all -> graceful no-op, no raise, no state change.
    assert svc.cancel_interactive("tab1") is False
    assert state.get_tab("tab1").is_analyzing is False
    assert handles.live_count() == 0


def test_cancel_interactive_does_not_touch_fit_analyze(qapp):
    state = make_state()
    handles = OperationHandles()
    svc, bg = make_service(state, EventBus(), handles=handles)

    # A worker-backed FIT analyze is in flight (its callback will settle it later).
    token = svc.start_analyze(
        "tab1", analyze_params_instance=object(), plots=make_plots()
    )
    assert bg.submit_count == 1

    # cancel_interactive must not reach into a FIT analyze (settling its handle
    # while the worker callback is still pending would double-settle on finish).
    assert svc.cancel_interactive("tab1") is False
    assert state.get_tab("tab1").is_analyzing is True  # FIT analyze untouched
    assert handles.poll(token) is None  # still live, settles via its own callback


def test_cancel_interactive_is_idempotent(qapp):
    state = make_state()
    svc, _ = make_service(state, EventBus())
    plugin = PluginDefinition(
        "pick", 0, (), lambda _state: None, lambda state: MagicMock()
    )
    svc.start_plugin(
        "tab1",
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=make_plots(),
    )

    assert svc.cancel_interactive("tab1") is True
    # A second cancel finds nothing in flight -> graceful no-op.
    assert svc.cancel_interactive("tab1") is False


def test_finish_after_cancel_interactive_is_inert(qapp):
    # Defensive: if a late Done arrives after a cancel, the already-settled token
    # is gone, so finish_interactive must not resurrect the tab into analyzing.
    state = make_state()
    svc, _ = make_service(state, EventBus())
    plugin = PluginDefinition(
        "pick", 0, (), lambda _state: None, lambda state: MagicMock()
    )
    svc.start_plugin(
        "tab1",
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance="submitted",
        plots=make_plots(),
    )
    svc.cancel_interactive("tab1")
    assert svc.finish_plugin("tab1") is False
    assert state.get_tab("tab1").analysis.result is None
    assert state.get_tab("tab1").is_analyzing is False
