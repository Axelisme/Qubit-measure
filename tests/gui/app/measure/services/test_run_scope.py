"""Run owner scope identity shares the operation's cooperative stop bit."""

from contextlib import ExitStack

import pytest
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord

from ._run_support import make_inputs, make_plots, make_run_service, make_state


@pytest.mark.parametrize("selector", ["run", "acquire", "stop", "operation"])
@pytest.mark.parametrize("fails", [False, True])
def test_selector_stop_keeps_core_return_or_failure(selector: str, fails: bool) -> None:
    state, tab_id, adapter = make_state()
    service, _gate, background, handles = make_run_service(state)
    record = RunRecord[ConfigBase, object](None, object())

    def run(*_args: object, context: RunContext) -> RunRecord[ConfigBase, object]:
        with ExitStack() as scopes:
            selected = context.scope
            if selector == "acquire":
                selected = scopes.enter_context(context.scope.branch("acquire"))
            if selector == "stop":
                assert service.cancel_run()
            elif selector == "operation":
                # Unscoped remote/MCP cancellation uses the handle's cancel hook.
                handles.cancel(token)
            else:
                accepted = service.cancel_scope(token, scope=selector)
                assert accepted.cancelled
            assert selected.stop.reason is not None
            if fails:
                raise RuntimeError("selector core error")
        return record

    adapter.run.side_effect = run
    token = service.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())
    background.run_work()
    outcome = handles.poll(token)
    assert outcome is not None
    assert outcome.status == ("failed" if fails else "finished")
    assert outcome.error == ("selector core error" if fails else None)
    pane = state.get_tab(tab_id).run
    assert pane.result is (None if fails else record)
    assert pane.stop_requested
    assert pane.source_operation_id == token


def test_gui_stop_reaches_already_active_descendants() -> None:
    state, tab_id, adapter = make_state()
    service, _gate, background, _handles = make_run_service(state)
    observed: list[bool] = []

    def run(*_args: object, context: RunContext) -> RunRecord[ConfigBase, object]:
        with (
            context.scope.branch("schedule") as schedule,
            schedule.branch("acquire") as acquire,
        ):
            assert service.cancel_run()
            observed.extend([schedule.cancelled, acquire.cancelled])
        return RunRecord(None, object())

    adapter.run.side_effect = run
    service.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())
    background.run_work()
    assert observed == [True, True]


def test_gui_run_scope_matches_status_and_operation_stop_intent() -> None:
    state, tab_id, adapter = make_state()
    service, _gate, background, handles = make_run_service(state)
    token = service.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())
    background.run_work()
    context: RunContext = adapter.run.call_args.kwargs["context"]
    assert hasattr(context, "scope"), "GUI RunContext must carry an explicit scope"
    assert context.scope.active_path == "run"
    assert not context.scope.cancelled
    assert service.status_scope(token) == ""
    assert service.live_paths_scope(token) == []
    assert handles.poll(token) is not None

    token = service.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())
    assert service.status_scope(token) == "run"
    assert service.live_paths_scope(token) == ["run"]
    assert service.cancel_run()
    background.run_work()
    context = adapter.run.call_args.kwargs["context"]
    assert context.scope.cancelled
    assert context.scope.active_path == "run"
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished"
