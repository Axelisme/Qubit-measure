"""env.run isolates core plotting lifetimes without clearing workflow trends."""

from pathlib import Path
from threading import Event

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.workflows import Completed, Failed, Live, WorkflowEnv
from zcu_tools.plotting import LiveUpdate
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests._plotting import TraceData, create_trace

from ._engine_fakes import (
    Cfg,
    Plan,
    Point,
    Rig,
    Tunables,
    completed,
    paused,
    running,
    until,
)


class Presentation(NonPresentingHost):
    """Observe the injected host's public final-refresh and release boundary."""

    def __init__(self) -> None:
        self.final: list[Figure] = []
        self.released: list[Figure] = []

    def refresh(self, figure: Figure, *, final: bool = False) -> None:
        if final:
            self.final.append(figure)

    def release(self, figure: Figure) -> None:
        self.released.append(figure)


def test_consecutive_core_operations_keep_workflow_plots_across_commits(
    tmp_path: Path,
) -> None:
    host = Presentation()
    rig = Rig(tmp_path, native=True, plot_host=host)
    attempts: list[Plots] = []
    delivered: list[int] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        attempts.append(context.plots)
        viewer = create_trace(context.plots, "measurement", "Delay", "Signal")
        viewer.update((np.array([0.0, 1.0]), np.array([1.0, 2.0])))
        return cfg.value

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            trend = create_trace(env.plots, "trend", "Iteration", "Value")
            for index in range(2):
                outcome = completed(await iteration.run(acquire, Cfg(value=7)))
                delivered.append(outcome.value.result)
                trend.update((np.array([0.0]), np.array([float(index)])))
                iteration.commit(Point(value=index))
                iteration = next(iterations)
        finally:
            iterations.close()

    try:
        rig.start(script)
        final = rig.engine.execute("run")
        assert final.lifecycle == "done", final.reason
        assert delivered == [7, 7]
        assert len(attempts) == 2
        assert attempts[0] is not attempts[1]
        assert all(plots is not rig.plots for plots in attempts)
        assert all(plots["measurement"] in host.final for plots in attempts)
        assert tuple(rig.plots) == ("trend",)
        assert attempts[0]["measurement"] in host.released
        np.testing.assert_array_equal(
            rig.plots["trend"].axes[0].lines[0].get_ydata(), [1.0]
        )
    finally:
        rig.close()


def test_pause_resume_reenters_core_factory_with_fresh_plots_and_same_cfg(
    tmp_path: Path,
) -> None:
    host = Presentation()
    rig = Rig(tmp_path, native=True, plot_host=host)
    entered = Event()
    attempts: list[Plots] = []
    configs: list[Cfg] = []
    delivered: list[int] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        attempts.append(context.plots)
        configs.append(cfg.model_copy(deep=True))
        create_trace(context.plots, "measurement", "Delay", "Signal")
        if len(attempts) == 1:
            entered.set()
            assert context.scope.stop.event.wait(10), "Pause must stop acquisition"
            cfg.value = -1
            return -1
        return cfg.value

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            create_trace(env.plots, "trend", "Iteration", "Value")
            result = completed(await iteration.run(acquire, Cfg(value=9)))
            delivered.append(result.value.result)
            iteration.commit(Point(value=result.value.result))
            iteration = next(iterations)
        finally:
            iterations.close()

    try:
        rig.start(script)
        with running(rig) as future:
            until(lambda: entered.is_set() or future.done(), "core operation entry")
            assert entered.is_set(), future.result() if future.done() else None
            rig.engine.pause("run")
            paused(rig)
            assert delivered == []
            rig.engine.resume("run")
            final = future.result(timeout=10)
        assert final.lifecycle == "done", final.reason
        assert delivered == [9]
        assert configs == [Cfg(value=9), Cfg(value=9)]
        assert all(plots["measurement"] in host.final for plots in attempts)
        assert attempts[0] is not attempts[1]
        assert attempts[0]["measurement"] in host.released
        assert tuple(rig.plots) == ("trend",)
        assert [op.outcome for op in rig.engine.records("run")[0].operations] == [
            "discarded",
            "completed",
        ]
    finally:
        rig.close()


@pytest.mark.parametrize("outcome", ["completed", "failed", "raised"])
def test_operation_finishes_before_return_or_failure_delivery(
    tmp_path: Path, outcome: str
) -> None:
    rig = Rig(tmp_path, native=True)
    handles: list[LiveUpdate[TraceData]] = []
    attempts: list[Plots] = []
    delivered: list[Completed[RunRecord[Cfg, int]] | Failed] = []
    errors: list[str] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        attempts.append(context.plots)
        handles.append(create_trace(context.plots, "measurement", "x", "y"))
        if outcome == "raised":
            raise ValueError("core failed")
        if outcome == "failed":
            context.scope.stop.set_error(
                "failed", "acquire failed", OSError("acquire failed")
            )
        return 3

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            try:
                delivered.append(await iteration.run(acquire, Cfg()))
            except ValueError as error:
                errors.append(str(error))
            # The old producer is no longer allowed to update even before run ends.
            with pytest.raises(RuntimeError, match="finished"):
                handles[0].update((np.array([0.0]), np.array([1.0])))
            create_trace(env.plots, "trend", "x", "y")
        finally:
            iterations.close()

    try:
        rig.start(script)
        final = rig.engine.execute("run")
        assert final.lifecycle == "done", final.reason
        assert attempts[0] is not rig.plots
        if outcome == "failed":
            assert delivered == [Failed("acquire failed")]
        elif outcome == "raised":
            assert errors == ["core failed"]
        else:
            assert completed(delivered[0]).value.result == 3
    finally:
        rig.close()


@pytest.mark.parametrize("registration", ["adopt", "subplots", "liveplot"])
def test_workflow_registration_rejects_operation_namespace(
    tmp_path: Path, registration: str
) -> None:
    rig = Rig(tmp_path)

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        if registration == "adopt":
            with pytest.raises(ValueError, match="operation/"):
                env.plots.adopt("operation/measurement", Figure())
        elif registration == "subplots":
            with pytest.raises(ValueError, match="operation/"):
                env.plots.subplots("operation/measurement")
        else:
            with pytest.raises(ValueError, match="operation/"):
                create_trace(env.plots, "operation/measurement", "x", "y")
        env.plots.subplots("trend")

    try:
        rig.start(script)
        final = rig.engine.execute("run")
        assert final.lifecycle == "done", final.reason
        assert tuple(rig.plots) == ("trend",)
    finally:
        rig.close()


@pytest.mark.parametrize("core_raises", [False, True])
def test_finish_failure_propagates_without_masking_core_error(
    tmp_path: Path, core_raises: bool
) -> None:
    class FailingHost(NonPresentingHost):
        fail = True

        def refresh(self, figure: Figure, *, final: bool = False) -> None:
            if final and self.fail:
                raise OSError("final refresh failed")

    host = FailingHost()
    rig = Rig(tmp_path, native=True, plot_host=host)
    errors: list[Exception] = []
    deliveries: list[object] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        create_trace(context.plots, "measurement", "x", "y")
        if core_raises:
            raise ValueError("core failed")
        return 1

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            try:
                deliveries.append(await iteration.run(acquire, Cfg()))
            except (ValueError, OSError) as error:
                errors.append(error)
        finally:
            iterations.close()

    try:
        rig.start(script)
        final = rig.engine.execute("run")
        assert final.lifecycle == "done", final.reason
        assert deliveries == []
        assert len(errors) == 1
        if core_raises:
            assert isinstance(errors[0], ValueError)
            assert str(errors[0]) == "core failed"
            assert "final refresh failed" in str(errors[0].__context__) or any(
                "final refresh failed" in note
                for note in getattr(errors[0], "__notes__", ())
            )
        else:
            assert isinstance(errors[0], OSError)
            assert str(errors[0]) == "final refresh failed"
    finally:
        host.fail = False
        rig.close()
