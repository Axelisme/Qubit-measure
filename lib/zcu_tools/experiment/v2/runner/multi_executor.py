from __future__ import annotations

import shutil
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Generic, Protocol, Self, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner.schedule import (
    ResultBuffer,
    Schedule,
    ScheduleStep,
    StopSignal,
    current_stop_signal,
)
from zcu_tools.experiment.v2.utils import Result, merge_result_list
from zcu_tools.liveplot import AbsLivePlot, MultiLivePlot, make_plot_frame
from zcu_tools.liveplot.backend.jupyter import grab_frame_with_instant_plot
from zcu_tools.utils.debug import print_traceback

T_Cfg = TypeVar("T_Cfg", bound=ExpCfgModel)
T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)


class PlottableMeasurement(Protocol):
    """Structural interface the multi-measurement executor needs from a task.

    Each app keeps its own concrete ``MeasurementTask`` ABC (they differ by cfg
    generics); this Protocol only captures the plotting-related contract shared
    by the base executor, so the base does not force-merge the two ABCs.
    """

    def num_axes(self) -> dict[str, int]: ...

    def make_plotter(
        self, name: str, axs: dict[str, list[Axes]], /
    ) -> Mapping[str, AbsLivePlot]: ...

    def update_plotter(
        self,
        plotters: Any,
        ctx: ScheduleStep[Any, Any],
        results: Any,
        /,
    ) -> None: ...

    def init(self, dynamic_pbar: bool = False) -> None: ...

    def run(self, state: ScheduleStep[Any, Any], /) -> None: ...

    def cleanup(self) -> None: ...

    def get_default_result(self) -> Result: ...


T_Measurement = TypeVar("T_Measurement", bound=PlottableMeasurement)


class MultiMeasurementExecutor(Generic[T_Measurement, T_Cfg]):
    """Shared base for executors that run several measurements with a combined
    live plot, optionally recording an FFmpeg animation of the figure.

    Subclasses own their ``run()`` (different outer drivers + cfg/env preamble)
    but reuse the layout / plotter / recording machinery here via
    ``_run_with_plotting``.
    """

    def __init__(self) -> None:
        self.record_path: Path | None = None
        self.measurements: OrderedDict[str, T_Measurement] = OrderedDict()

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

    def make_ax_layout(self) -> tuple[Figure, dict[str, dict[str, list[Axes]]]]:
        assert len(self.measurements) > 0

        num_axes_map = {
            ms_name: dict(sorted(ms.num_axes().items(), key=lambda x: -x[1]))
            for ms_name, ms in self.measurements.items()
        }

        total_num_axes = sum(
            sum(num_axes.values()) for num_axes in num_axes_map.values()
        )

        n_row = int(total_num_axes**0.5)
        n_col = int(np.ceil(total_num_axes / n_row))
        fig, axs = make_plot_frame(
            n_row,
            n_col,
            plot_instant=True,
            figsize=(min(14, 3.5 * n_col), min(8, 2.5 * n_row)),
        )

        # collect axes into dict
        axs_map: dict[str, dict[str, list[Axes]]] = defaultdict(dict)
        i, j = 0, 0
        for ms_name, num_axes in num_axes_map.items():
            for ax_name, ax_num in num_axes.items():
                for _ in range(ax_num):
                    axs_map[ms_name].setdefault(ax_name, []).append(axs[i][j])
                    j += 1
                    if j == n_col:
                        j = 0
                        i += 1

        return fig, axs_map

    def make_plotter(
        self,
    ) -> tuple[
        Figure,
        MultiLivePlot[tuple[str, str]],
        Callable[[ScheduleStep[Any, Any]], None],
        FFMpegWriter | None,
    ]:
        fig, axs_map = self.make_ax_layout()

        if self.record_path is not None:
            if shutil.which("ffmpeg") is None:
                raise RuntimeError(
                    "FFmpeg is not found. Please install FFmpeg and add it to your "
                    "PATH to record animations."
                )
            writer: FFMpegWriter | None = FFMpegWriter(fps=30)
            assert writer is not None
            writer.setup(fig, str(self.record_path), dpi=200)
        else:
            writer = None

        plotters_map = {
            ms_name: ms.make_plotter(ms_name, axs_map[ms_name])
            for ms_name, ms in self.measurements.items()
        }

        T = TypeVar("T")

        def flatten_dict(d: Mapping[str, Mapping[str, T]]) -> dict[tuple[str, str], T]:
            return {(n1, n2): v for n1, d2 in d.items() for n2, v in d2.items()}

        plotter = MultiLivePlot(fig, flatten_dict(plotters_map))

        def plot_fn(ctx: ScheduleStep[Any, Any]) -> None:
            if len(ctx.path) < 2:
                cur_tasks = list(self.measurements.keys())
            else:
                cur_task = ctx.path[1]
                if not isinstance(cur_task, str):
                    raise TypeError(
                        f"Expected measurement name at path[1], got {type(cur_task)}"
                    )
                cur_tasks = [cur_task]

            result_data = ctx.schedule.data
            if not isinstance(result_data, list):
                raise TypeError(
                    f"Expected executor result data to be list, got {type(result_data)}"
                )
            for cur_task in cur_tasks:
                task_rows: list[Any] = []
                for row in result_data:
                    if not isinstance(row, Mapping):
                        raise TypeError(
                            "Expected executor result data rows to be mappings, "
                            f"got {type(row)}"
                        )
                    task_rows.append(row[cur_task])
                task_result = merge_result_list(task_rows)
                self.measurements[cur_task].update_plotter(
                    plotters_map[cur_task], ctx, task_result
                )

            if self.record_path is not None:
                assert writer is not None
                grab_frame_with_instant_plot(writer)

            plotter.refresh()

        return fig, plotter, plot_fn, writer

    def _run_with_plotting(
        self,
        init_result: T_RootResult,
        cfg: T_Cfg,
        env_dict: dict[str, Any],
        run_fn: Callable[[Schedule[T_Cfg]], None],
    ) -> T_RootResult:
        fig, plotter, plot_fn, writer = self.make_plotter()
        stop = current_stop_signal() or StopSignal()
        result_buffer = ResultBuffer(init_result, on_update=plot_fn)

        with Schedule(
            cfg,
            result_buffer,
            env_dict=env_dict,
            stop=stop,
        ) as sched:
            with plotter:
                try:
                    for measurement in self.measurements.values():
                        measurement.init(dynamic_pbar=True)
                    run_fn(sched)
                except KeyboardInterrupt:
                    sched.set_stop()
                except Exception:
                    print_traceback()
                    raise
                finally:
                    for measurement in self.measurements.values():
                        measurement.cleanup()
                    if self.record_path is not None:
                        assert writer is not None
                        writer.finish()
        plt.close(fig)

        return result_buffer.data

    def _default_batch_result(self) -> dict[str, Result]:
        return {name: ms.get_default_result() for name, ms in self.measurements.items()}

    def _run_measurement_batch(
        self,
        step: ScheduleStep[Any, Any],
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
        state: ScheduleStep[Any, Any],
        retry_time: int,
    ) -> None:
        if retry_time < 0:
            raise ValueError("retry_time must be non-negative")
        for attempt in range(retry_time + 1):
            try:
                measurement.run(state)
            except KeyboardInterrupt:
                state.set_stop()
                break
            except Exception:
                if attempt == retry_time:
                    raise
                measurement.cleanup()
                measurement.init(dynamic_pbar=True)
                continue
            break
