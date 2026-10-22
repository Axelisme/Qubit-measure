"""Workflow scoped cancel delivers partial facts and replaces iteration at commit."""

from pathlib import Path

import pytest
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.workflows import Live, WorkflowEnv

from ._engine_fakes import Plan, Point, Rig, Tunables, completed


@pytest.mark.parametrize("scope", ["acquire", "schedule", "run", "iteration"])
def test_workflow_scope_ack_status_and_partial_facts(
    tmp_path: Path, scope: str
) -> None:
    rig = Rig(tmp_path)
    outcomes: list[tuple[bool, bool]] = []
    acquisitions: list[bool] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        def acquire(ctx: CallContext) -> int:
            schedule = ctx.scope.branch("schedule")
            first = schedule.branch("acquire")
            try:
                status = rig.engine.status()
                assert status is not None
                assert status.active_path == (
                    "workflow:example/iteration#1/operation#1/schedule#1/acquire#1"
                )
                result = rig.engine.cancel("run", scope=scope, expected=None)
                assert result.cancelled
                assert result.reason == "scope cancellation requested"
                assert result.active_path == status.active_path
                acquisitions.append(first.cancelled)
            finally:
                first.release()
            second = schedule.branch("acquire")
            acquisitions.append(second.cancelled)
            second.release()
            schedule.release()
            return 11

        iterations = env.iterations()
        iteration = next(iterations)
        first = completed(await iteration.call(acquire))
        outcomes.append((first.partial_result, first.facts.stopped))
        second = completed(await iteration.call(lambda ctx: ctx.scope.stop.is_set()))
        outcomes.append((second.partial_result, second.facts.stopped))
        iteration.commit(Point(value=11))
        iteration = next(iterations)
        third = completed(await iteration.call(lambda ctx: ctx.scope.stop.is_set()))
        outcomes.append((third.partial_result, third.facts.stopped))
        iteration.commit(Point(value=12))
        iterations.close()

    rig.start(script)
    try:
        final = rig.engine.execute("run")
        assert final.lifecycle == ("cancelled" if scope == "workflow" else "done"), (
            final.reason
        )
        assert acquisitions == [True, scope != "acquire"]
        assert outcomes == [
            (True, scope in {"run", "iteration", "workflow"}),
            (scope in {"iteration", "workflow"}, scope in {"iteration", "workflow"}),
            (scope == "workflow", scope == "workflow"),
        ]
        records = rig.engine.records("run")
        assert [
            [operation.outcome for operation in record.operations] for record in records
        ] == [
            [
                "partial",
                "partial" if scope in {"iteration", "workflow"} else "completed",
            ],
            ["partial" if scope == "workflow" else "completed"],
        ]
    finally:
        rig.close()
