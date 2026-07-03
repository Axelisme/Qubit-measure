from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner import (
    Acquirer,
    ComposedMeasurementBundle,
    MeasurementTask,
    MultiMeasurementExecutor,
    ResultTree,
    ResultUpdateEvent,
    Schedule,
    ScheduleStep,
    TaskPersister,
    TaskPlotter,
)
from zcu_tools.experiment.v2.utils import Result

FakeResult: TypeAlias = dict[str, NDArray[np.float64]]


class ExecCfg(ExpCfgModel):
    pass


@dataclass
class ExecEnv:
    current_index: int = -1


class FakePlotter:
    def __init__(self) -> None:
        self.refresh_count = 0
        self.entered = False
        self.exited = False

    def __enter__(self) -> FakePlotter:
        self.entered = True
        return self

    def __exit__(self, *_exc: object) -> None:
        self.exited = True

    def refresh(self) -> None:
        self.refresh_count += 1


class FakeTask(MeasurementTask[ExecCfg, ExecEnv, FakeResult, Any, NDArray[np.float64]]):
    def __init__(
        self,
        *,
        offset: float = 0.0,
        fail_once_at: int | None = None,
        fail_always_at: int | None = None,
        stop_at: int | None = None,
    ) -> None:
        self.offset = offset
        self.fail_once_at = fail_once_at
        self.fail_always_at = fail_always_at
        self.stop_at = stop_at
        self.init_count = 0
        self.cleanup_count = 0
        self.run_indices: list[int] = []
        self.plot_events: list[tuple[str, int | None, bool, NDArray[np.float64]]] = []
        self._failed_indices: set[int] = set()

    def init(self, dynamic_pbar: bool = False) -> None:
        self.init_count += 1

    def run(self, state: ScheduleStep[ExecCfg, Any, ExecEnv]) -> None:
        index = state.env.current_index
        self.run_indices.append(index)
        if self.stop_at == index:
            raise KeyboardInterrupt
        if self.fail_always_at == index:
            raise RuntimeError("permanent failure")
        if self.fail_once_at == index and index not in self._failed_indices:
            self._failed_indices.add(index)
            raise RuntimeError("temporary failure")
        state.set_data(
            {"value": np.array(float(index) + self.offset, dtype=np.float64)},
            flush=True,
        )

    def cleanup(self) -> None:
        self.cleanup_count += 1

    def get_default_result(self) -> FakeResult:
        return {"value": np.array(np.nan, dtype=np.float64)}

    def num_axes(self) -> dict[str, int]:
        return {"value": 1}

    def make_plotter(self, name: str, axs: dict[str, Any]) -> dict[str, object]:
        return {"value": object()}

    def update_plotter(
        self,
        plotters: Any,
        event: ResultUpdateEvent[ExecEnv, FakeResult],
        result: FakeResult,
    ) -> None:
        self.plot_events.append(
            (
                event.measurement_name,
                event.outer_index,
                event.flush,
                result["value"].copy(),
            )
        )

    def save(
        self,
        filepath: str,
        axis_values: NDArray[np.float64],
        result: FakeResult,
        comment: str | None,
        prefix_tag: str,
    ) -> None:
        raise AssertionError("save is not used by executor lifecycle tests")


class FakeExecutor(
    MultiMeasurementExecutor[FakeTask, ExecCfg, ExecEnv, NDArray[np.float64]]
):
    def __init__(self, outer_values: NDArray[np.float64]) -> None:
        super().__init__()
        self.outer_values = outer_values
        self.fake_plotter = FakePlotter()
        self.last_fig: Figure | None = None

    def make_plotter(self) -> Any:
        fig = plt.figure()
        self.last_fig = fig
        plotters_map = {
            name: task.make_plotter(name, {})
            for name, task in self.measurements.items()
        }
        return fig, self.fake_plotter, plotters_map, None

    def run(self, retry_time: int = 0) -> dict[str, FakeResult]:
        env = ExecEnv()
        cfg = ExecCfg()

        def run_loop(sched: Schedule[ExecCfg, ExecEnv]) -> None:
            for index, outer_step in sched.scan("outer", self.outer_values):
                env.current_index = int(index)
                self._run_measurement_batch(outer_step, retry_time)

        return cast(
            dict[str, FakeResult],
            dict(
                self._run(
                    cfg=cfg,
                    env=env,
                    outer_values=self.outer_values,
                    run_loop=run_loop,
                )
            ),
        )


class SplitAcquirer(Acquirer[ExecCfg, ExecEnv, FakeResult]):
    def __init__(self) -> None:
        self.init_count = 0
        self.cleanup_count = 0

    def init(self, dynamic_pbar: bool = False) -> None:
        self.init_count += 1

    def run(self, state: ScheduleStep[ExecCfg, Any, ExecEnv]) -> None:
        state.set_data(self.get_default_result(), flush=True)

    def cleanup(self) -> None:
        self.cleanup_count += 1

    def get_default_result(self) -> FakeResult:
        return {"value": np.array(3.0, dtype=np.float64)}


class SplitPlotter(TaskPlotter[ExecEnv, FakeResult, Any]):
    def __init__(self) -> None:
        self.updated = False

    def num_axes(self) -> dict[str, int]:
        return {"value": 1}

    def make_plotter(self, name: str, axs: dict[str, Any]) -> dict[str, object]:
        return {"value": object()}

    def update_plotter(
        self,
        plotters: Any,
        event: ResultUpdateEvent[ExecEnv, FakeResult],
        result: FakeResult,
    ) -> None:
        self.updated = True


class SplitPersister(TaskPersister[FakeResult, NDArray[np.float64]]):
    def __init__(self) -> None:
        self.saved = False

    def save(
        self,
        filepath: str,
        axis_values: NDArray[np.float64],
        result: FakeResult,
        comment: str | None,
        prefix_tag: str,
    ) -> None:
        self.saved = True


def test_multi_executor_template_lifecycle_and_per_node_plot_update() -> None:
    first = FakeTask(offset=0.0)
    second = FakeTask(offset=10.0)
    executor = FakeExecutor(np.array([0.0, 1.0], dtype=np.float64)).add_measurements(
        {"first": first, "second": second}
    )

    result = executor.run()

    np.testing.assert_allclose(result["first"]["value"], np.array([0.0, 1.0]))
    np.testing.assert_allclose(result["second"]["value"], np.array([10.0, 11.0]))
    assert first.init_count == second.init_count == 1
    assert first.cleanup_count == second.cleanup_count == 1
    assert [event[0] for event in first.plot_events] == ["first", "first"]
    assert [event[1] for event in first.plot_events] == [0, 1]
    assert [event[2] for event in first.plot_events] == [True, True]
    assert executor.fake_plotter.entered is True
    assert executor.fake_plotter.exited is True
    assert executor.fake_plotter.refresh_count == 4
    assert executor.last_fig is not None
    assert not plt.fignum_exists(executor.last_fig.number)


def test_multi_executor_retries_measurement_and_reinitializes_task() -> None:
    task = FakeTask(fail_once_at=0)
    executor = FakeExecutor(np.array([0.0], dtype=np.float64)).add_measurements(
        {"task": task}
    )

    result = executor.run(retry_time=1)

    np.testing.assert_allclose(result["task"]["value"], np.array([0.0]))
    assert task.run_indices == [0, 0]
    assert task.init_count == 2
    assert task.cleanup_count == 2
    assert len(task.plot_events) == 1
    assert executor.last_run_outcome is not None
    assert executor.last_run_outcome.status == "completed"


def test_multi_executor_stop_keeps_partial_result_and_cleans_up() -> None:
    task = FakeTask(stop_at=1)
    executor = FakeExecutor(
        np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ).add_measurements({"task": task})

    result = executor.run()

    np.testing.assert_allclose(
        result["task"]["value"],
        np.array([0.0, np.nan, np.nan]),
        equal_nan=True,
    )
    assert task.run_indices == [0, 1]
    assert task.cleanup_count == 1
    assert [event[1] for event in task.plot_events] == [0]
    assert executor.last_run_outcome is not None
    assert executor.last_run_outcome.status == "interrupted"


def test_multi_executor_retry_exhaustion_keeps_partial_result_and_cleans_up() -> None:
    task = FakeTask(fail_always_at=1)
    executor = FakeExecutor(
        np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ).add_measurements({"task": task})

    result = executor.run(retry_time=1)

    np.testing.assert_allclose(
        result["task"]["value"],
        np.array([0.0, np.nan, np.nan]),
        equal_nan=True,
    )
    assert task.run_indices == [0, 1, 1]
    assert task.init_count == 2
    assert task.cleanup_count == 2
    assert [event[1] for event in task.plot_events] == [0]
    assert executor.last_run_outcome is not None
    assert executor.last_run_outcome.status == "failed"
    assert isinstance(executor.last_run_outcome.exception, RuntimeError)


def test_composed_measurement_bundle_delegates_components() -> None:
    acquirer = SplitAcquirer()
    plotter = SplitPlotter()
    persister = SplitPersister()
    bundle = ComposedMeasurementBundle[
        ExecCfg, ExecEnv, FakeResult, Any, NDArray[np.float64]
    ](acquirer=acquirer, plotter=plotter, persister=persister)
    data = cast(
        list[dict[str, Result]],
        [{"bundle": {"value": np.array(np.nan, dtype=np.float64)}}],
    )
    tree = ResultTree[ExecEnv](data, outer_values=np.array([0.0], dtype=np.float64))
    env = ExecEnv(current_index=0)

    bundle.init(dynamic_pbar=True)
    with Schedule(ExecCfg(), tree, env=env) as sched:
        _, outer_step = next(sched.scan("outer", np.array([0.0], dtype=np.float64)))
        bundle.run(outer_step.child("bundle"))
    bundle.cleanup()

    result = cast(FakeResult, tree.measurement_result("bundle"))
    event = ResultUpdateEvent(
        measurement_name="bundle",
        outer_index=0,
        outer_value=0.0,
        env=env,
        node=tree.measurement_node("bundle"),
        result=result,
        flush=True,
    )
    plotters = bundle.make_plotter("bundle", {})
    bundle.update_plotter(plotters, event, result)
    bundle.save("unused", np.array([0.0], dtype=np.float64), result, None, "bundle")

    assert acquirer.init_count == 1
    assert acquirer.cleanup_count == 1
    assert plotter.updated is True
    assert persister.saved is True
    np.testing.assert_allclose(result["value"], np.array([3.0]))
