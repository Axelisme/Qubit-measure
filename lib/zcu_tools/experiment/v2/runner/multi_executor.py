from __future__ import annotations

import shutil
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, Self

import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import TypeVar

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner.result_tree import ResultTree, ResultUpdateEvent
from zcu_tools.experiment.v2.runner.schedule import (
    Schedule,
    ScheduleStep,
    StopSignal,
    current_stop_signal,
)
from zcu_tools.experiment.v2.runner.task import MeasurementBundle
from zcu_tools.experiment.v2.utils import Result
from zcu_tools.liveplot import AbsLivePlot, MultiLivePlot, make_plot_frame
from zcu_tools.liveplot.backend.jupyter import grab_frame_with_instant_plot
from zcu_tools.utils.debug import print_traceback

T_Cfg = TypeVar("T_Cfg", bound=ExpCfgModel)
T_Env = TypeVar("T_Env")
T_Axis = TypeVar("T_Axis", bound=Sequence[Any] | NDArray[Any])
T_Measurement = TypeVar(
    "T_Measurement", bound=MeasurementBundle[Any, Any, Any, Any, Any]
)


class MultiMeasurementExecutor(Generic[T_Measurement, T_Cfg, T_Env, T_Axis]):
    """Shared base for executors that run several measurements with a combined
    live plot, optionally recording an FFmpeg animation of the figure.

    Subclasses build cfg/env and provide the outer-loop policy; this base owns
    layout, plotter, recording, ``Schedule`` lifecycle, and final result folding.
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
        dict[str, Mapping[str, AbsLivePlot]],
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
        return fig, plotter, plotters_map, writer

    def _default_batch_result(self) -> dict[str, Result]:
        return {name: ms.get_default_result() for name, ms in self.measurements.items()}

    def _make_result_tree(
        self,
        data: list[dict[str, Result]],
        *,
        env: T_Env,
        outer_values: T_Axis,
        plotter: MultiLivePlot[tuple[str, str]],
        plotters_map: Mapping[str, Mapping[str, AbsLivePlot]],
        writer: FFMpegWriter | None,
    ) -> ResultTree[T_Env]:
        tree: ResultTree[T_Env] = ResultTree(data, outer_values=outer_values, env=env)

        def make_callback(
            name: str,
            measurement: T_Measurement,
        ) -> Callable[[ResultUpdateEvent[T_Env, Any]], None]:
            def update(event: ResultUpdateEvent[T_Env, Any]) -> None:
                measurement.update_plotter(plotters_map[name], event, event.result)
                if self.record_path is not None:
                    assert writer is not None
                    grab_frame_with_instant_plot(writer)
                plotter.refresh()

            return update

        for name, measurement in self.measurements.items():
            tree.measurement_node(name).subscribe(make_callback(name, measurement))

        return tree

    def _run(
        self,
        *,
        cfg: T_Cfg,
        env: T_Env,
        outer_values: T_Axis,
        run_loop: Callable[[Schedule[T_Cfg, T_Env]], None],
    ) -> Mapping[str, Result]:
        if len(self.measurements) == 0:
            raise ValueError("No measurements added")

        init_result = [self._default_batch_result() for _ in range(len(outer_values))]

        fig, plotter, plotters_map, writer = self.make_plotter()
        stop = current_stop_signal() or StopSignal()
        result_tree = self._make_result_tree(
            init_result,
            env=env,
            outer_values=outer_values,
            plotter=plotter,
            plotters_map=plotters_map,
            writer=writer,
        )

        try:
            with Schedule(cfg, result_tree, env=env, stop=stop) as sched:
                with plotter:
                    try:
                        for measurement in self.measurements.values():
                            measurement.init(dynamic_pbar=True)
                        run_loop(sched)
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
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

        signals_dict = {
            name: result_tree.measurement_result(name)
            for name in self.measurements.keys()
        }

        self.last_cfg = cfg
        self.last_result = signals_dict

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
