"""Run owner exposes forks and resolves selectors without selecting the latest."""

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord

from ._run_support import make_inputs, make_plots, make_run_service, make_state


def test_run_scope_reports_fork_candidates_and_exact_leaf_selection() -> None:
    state, tab_id, adapter = make_state()
    service, _gate, background, handles = make_run_service(state)

    def run(*_args: object, context: RunContext) -> RunRecord[ConfigBase, object]:
        with (
            context.scope.branch("operation", name="T1") as first,
            context.scope.branch("operation", name="T1") as second,
        ):
            assert service.status_scope(token) == "run"
            paths = ["run/operation:T1#1", "run/operation:T1#2"]
            assert service.live_paths_scope(token) == paths
            rejected = service.cancel_scope(token, "operation:T1")
            assert (rejected.cancelled, rejected.reason) == (False, "ambiguous")
            assert rejected.candidates == paths
            assert not first.cancelled and not second.cancelled
            assert service.cancel_scope(token, paths[1]).cancelled
            assert not first.cancelled and second.cancelled
            assert service.cancel_scope(token, "operation:T1#1").cancelled
        return RunRecord(None, object())

    adapter.run.side_effect = run
    token = service.start_run(make_inputs(state, tab_id, adapter), plots=make_plots())
    background.run_work()
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished", outcome
