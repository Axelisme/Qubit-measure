from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, Self

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import TypeVar

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.runtime.result_tree import ResultTree, ResultUpdateEvent
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    ScheduleOutcome,
    ScheduleStep,
)
from zcu_tools.experiment.v2.runtime.task import MeasurementBundle, TaskLivePlot
from zcu_tools.experiment.v2.utils.helper import Result
from zcu_tools.plotting.plots import MovieRecording, Plots
from zcu_tools.utils.debug import log_current_exception

T_Cfg = TypeVar("T_Cfg", bound=ExpCfgModel)
T_Env = TypeVar("T_Env")
T_Axis = TypeVar("T_Axis", bound=Sequence[Any] | NDArray[Any])
T_Measurement = TypeVar(
    "T_Measurement", bound=MeasurementBundle[Any, Any, Any, Any, Any]
)

logger = logging.getLogger(__name__)


class MultiMeasurementExecutor(Generic[T_Measurement, T_Cfg, T_Env, T_Axis]):
    """Shared base for executors that run several measurements with a combined
    live plot, optionally recording an FFmpeg animation of the figure.

    Subclasses build cfg/env and provide the outer-loop policy; this base owns
    layout, plotter, recording, ``Schedule`` lifecycle, and final result folding.
    """

    def __init__(self) -> None:
        self.record_path: Path | None = None
        self.measurements: OrderedDict[str, T_Measurement] = OrderedDict()
        self.last_run_outcome: ScheduleOutcome | None = None

    def add_measurements(self, measurements: Mapping[str, T_Measurement]) -> Self:
        for name, measurement in measurements.items():
            if name in self.measurements:
                raise ValueError(f"Measurement {name} already exists")
            self.measurements[name] = measurement

        return self

    def record_animation(self, path: str) -> Self:
        if self.record_path is not None:
            raise ValueError("Animation recording path already set")
        self.record_path = Path(path)
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def make_ax_layout(
        self, *, plots: Plots, figure_name: str
    ) -> tuple[Figure, dict[str, dict[str, list[Axes]]]]:
        if not self.measurements:
            raise ValueError("No measurements added")

        num_axes_map = {
            ms_name: dict(sorted(ms.num_axes().items(), key=lambda x: -x[1]))
            for ms_name, ms in self.measurements.items()
        }

        total_num_axes = sum(
            sum(num_axes.values()) for num_axes in num_axes_map.values()
        )

        if total_num_axes < 1:
            raise ValueError("Measurements require at least one plot axis")
        n_row = int(total_num_axes**0.5)
        n_col = int(np.ceil(total_num_axes / n_row))
        fig, _ = plots.subplots(
            figure_name,
            nrows=n_row,
            ncols=n_col,
            squeeze=False,
            figsize=(min(14, 3.5 * n_col), min(8, 2.5 * n_row)),
        )

        # collect axes into dict
        axs_map: dict[str, dict[str, list[Axes]]] = defaultdict(dict)
        axes = iter(fig.axes)
        for ms_name, num_axes in num_axes_map.items():
            for ax_name, ax_num in num_axes.items():
                for _ in range(ax_num):
                    axs_map[ms_name].setdefault(ax_name, []).append(next(axes))

        return fig, axs_map

    def make_plotter(
        self, *, plots: Plots, figure_name: str
    ) -> tuple[dict[str, Mapping[str, TaskLivePlot]], MovieRecording | None]:
        _, axs_map = self.make_ax_layout(plots=plots, figure_name=figure_name)
        plotters_map = {
            ms_name: ms.make_plotter(
                ms_name, axs_map[ms_name], plots=plots, figure_name=figure_name
            )
            for ms_name, ms in self.measurements.items()
        }
        writer = (
            plots.record_animation(figure_name, self.record_path)
            if self.record_path is not None
            else None
        )
        return plotters_map, writer

    def _default_batch_result(self) -> dict[str, Result]:
        return {name: ms.get_default_result() for name, ms in self.measurements.items()}

    def _make_result_tree(  # noqa: PLR0913 - explicit result, presentation and recording owners
        self,
        data: list[dict[str, Result]],
        *,
        env: T_Env,
        outer_values: T_Axis,
        plots: Plots,
        figure_name: str,
        plotters_map: Mapping[str, Mapping[str, TaskLivePlot]],
        writer: MovieRecording | None,
    ) -> ResultTree[T_Env]:
        tree: ResultTree[T_Env] = ResultTree(data, outer_values=outer_values, env=env)

        def make_callback(
            name: str,
            measurement: T_Measurement,
        ) -> Callable[[ResultUpdateEvent[T_Env, Any]], None]:
            def update(event: ResultUpdateEvent[T_Env, Any]) -> None:
                measurement.update_plotter(plotters_map[name], event, event.result)
                if writer is not None:
                    writer.grab_frame()
                plots.refresh(figure_name)

            return update

        for name, measurement in self.measurements.items():
            tree.measurement_node(name).subscribe(make_callback(name, measurement))

        return tree

    def _run(
        self,
        *,
        cfg: T_Cfg,
        env: T_Env,
        stop: StopSignal,
        plots: Plots,
        outer_values: T_Axis,
        run_loop: Callable[[Schedule[T_Cfg, T_Env]], None],
    ) -> Mapping[str, Result]:
        if len(self.measurements) == 0:
            raise ValueError("No measurements added")

        init_result = [self._default_batch_result() for _ in range(len(outer_values))]

        figure_name = "measurement"
        plotters_map, writer = self.make_plotter(plots=plots, figure_name=figure_name)
        try:
            result_tree = self._make_result_tree(
                init_result,
                env=env,
                outer_values=outer_values,
                plots=plots,
                figure_name=figure_name,
                plotters_map=plotters_map,
                writer=writer,
            )
            with Schedule(cfg, result_tree, env=env, stop=stop) as sched:
                try:
                    for measurement in self.measurements.values():
                        measurement.init(dynamic_pbar=True)
                    run_loop(sched)
                except KeyboardInterrupt as exc:
                    sched._mark_interrupted(exc)
                except Exception:
                    log_current_exception(logger, "measurement executor failed")
                    raise
                finally:
                    for measurement in self.measurements.values():
                        measurement.cleanup()
        finally:
            if writer is not None:
                writer.finish()

        signals_dict = {
            name: result_tree.measurement_result(name)
            for name in self.measurements.keys()
        }

        self.last_cfg = cfg
        self.last_result = signals_dict
        self.last_run_outcome = sched.outcome

        return signals_dict

    def _run_measurement_batch(
        self,
        step: ScheduleStep[Any, Any, Any],
        retry_time: int,
    ) -> None:
        step.batch(
            {
                name: lambda child, measurement=measurement: (
                    self._run_measurement_with_retries(measurement, child, retry_time)
                )
                for name, measurement in self.measurements.items()
            }
        )

    def _run_measurement_with_retries(
        self,
        measurement: T_Measurement,
        state: ScheduleStep[Any, Any, Any],
        retry_time: int,
    ) -> None:
        if retry_time < 0:
            raise ValueError("retry_time must be non-negative")
        for attempt in range(retry_time + 1):
            try:
                measurement.run(state)
            except KeyboardInterrupt as exc:
                state.schedule._mark_interrupted(exc)
                break
            except Exception as exc:
                if attempt == retry_time:
                    state.schedule._mark_failed(exc)
                    break
                measurement.cleanup()
                measurement.init(dynamic_pbar=True)
                continue
            break
