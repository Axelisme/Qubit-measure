"""Run owner scope identity shares the operation's cooperative stop bit."""

from zcu_tools.experiment.context import RunContext

from ._run_support import make_inputs, make_plots, make_run_service, make_state


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
    assert handles.poll(token) is not None

    token = service.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())
    assert service.status_scope(token) == "run"
    assert service.cancel_run()
    background.run_work()
    context = adapter.run.call_args.kwargs["context"]
    assert context.scope.cancelled
    assert context.scope.active_path == "run"
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "cancelled"
