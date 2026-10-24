"""Async authors retain injected Plots and own their figure outputs."""

from pathlib import Path

import numpy as np
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.workflows import Live, WorkflowEnv

from tests._plotting import create_trace

from ._engine_fakes import Plan, Point, Rig, Tunables, completed


def test_engine_reuses_plots_across_commits_and_saves_figure(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    seen: list[object] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            viewer = create_trace(env.plots, "trace", "Time", "Signal")
            for index in range(plan.count):
                seen.append(env.plots)

                def acquire(ctx: CallContext, index: int = index) -> int:
                    viewer.update((np.array([0.0, 1.0]), np.array([1.0, 3.0]) + index))
                    return index

                result = completed(await iteration.call(acquire))
                directory = iteration.dir
                env.plots.save("trace", directory / "trace.png")
                iteration.commit(Point(value=result.value))
                iteration = next(iterations)
        finally:
            iterations.close()

    try:
        rig.start(script)
        final = rig.engine.execute("run")
        assert final.lifecycle == "done", final.reason
        assert final.committed_seq == 2
        assert seen == [rig.plots, rig.plots]
        np.testing.assert_array_equal(
            rig.plots["trace"].axes[0].lines[0].get_ydata(), [2.0, 4.0]
        )
        saved = sorted(rig.paths.data_root.rglob("trace.png"))
        assert len(saved) == 2
        assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in saved)
    finally:
        rig.close()


def test_projection_error_is_not_a_recoverable_acquire_failure(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    cause = ValueError("plot callback failed")
    delivered: list[object] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            viewer = create_trace(env.plots, "trace", "x", "y")

            def project(ctx: CallContext) -> None:
                viewer.update((np.array([0.0]), np.array([2.0])))
                raise cause

            delivered.append(await iteration.call(project))
            iteration.commit(Point(value=1))
            iteration = next(iterations)
        finally:
            iterations.close()

    try:
        rig.start(script)
        final = rig.engine.execute("run")
        assert final.lifecycle == "failed"
        assert final.reason == "ValueError: plot callback failed"
        assert final.committed_seq == 0
        assert delivered == []
    finally:
        rig.close()
