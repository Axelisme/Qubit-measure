"""Run owner scope identity shares the operation's cooperative stop bit."""

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord

from ._run_support import make_inputs, make_plots, make_run_service, make_state


def test_gui_stop_reaches_already_active_descendants() -> None:
    state, tab_id, adapter = make_state()
    service, _gate, background, _handles = make_run_service(state)
    observed: list[bool] = []

    def run(*_args: object, context: RunContext) -> RunRecord[ExpCfgModel, object]:
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
    assert outcome is not None and outcome.status == "cancelled"
