from __future__ import annotations

import shutil
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Hashable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import Any, Generic, Protocol, Self, TypeVar, cast, overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import DTypeLike, NDArray

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner.schedule import (
    SignalBuffer,
    StopSignal,
    current_stop_signal,
)
from zcu_tools.experiment.v2.utils import Result, merge_result_list
from zcu_tools.liveplot import AbsLivePlot, MultiLivePlot, make_plot_frame
from zcu_tools.liveplot.backend.jupyter import grab_frame_with_instant_plot
from zcu_tools.utils.debug import print_traceback
from zcu_tools.utils.func_tools import min_interval

T_Cfg = TypeVar("T_Cfg", bound=ExpCfgModel)
T_ChildCfg = TypeVar("T_ChildCfg", bound=ExpCfgModel)
T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)
T_ChildResult = TypeVar("T_ChildResult", bound=Result)
T_MappingResult = TypeVar("T_MappingResult", bound=Mapping[Any, Any])


@dataclass
class MeasurementContext(Generic[T_Result, T_RootResult, T_Cfg]):
    root_data: T_RootResult
    cfg: T_Cfg
    env: dict[str, Any] = field(default_factory=dict)
    on_update: Callable[[MeasurementContext[Any, T_RootResult, Any]], object] | None = (
        None
    )
    path: tuple[int | Hashable, ...] = field(default_factory=tuple)
    stop: StopSignal = field(default_factory=StopSignal)

    def is_stop(self) -> bool:
        return self.stop.is_stop()

    def set_stop(self) -> None:
        self.stop.set_stop()

    @overload
    def child(
        self: MeasurementContext[list[T_ChildResult], T_RootResult, T_Cfg],
        addr: int,
        child_type: None = None,
    ) -> MeasurementContext[T_ChildResult, T_RootResult, T_Cfg]: ...

    @overload
    def child(
        self: MeasurementContext[T_MappingResult, T_RootResult, T_Cfg],
        addr: Hashable,
        child_type: type[T_ChildResult],
    ) -> MeasurementContext[T_ChildResult, T_RootResult, T_Cfg]: ...

    @overload
    def child(
        self: MeasurementContext[T_MappingResult, T_RootResult, T_Cfg],
        addr: Hashable,
        child_type: None = None,
    ) -> MeasurementContext[Any, T_RootResult, T_Cfg]: ...

    def child(
        self,
        addr: int | Hashable,
        child_type: type[T_ChildResult] | None = None,
    ) -> MeasurementContext[Any, T_RootResult, T_Cfg]:
        return MeasurementContext(
            root_data=self.root_data,
            cfg=deepcopy(self.cfg),
            env=self.env,
            on_update=self.on_update,
            path=self.path + (addr,),
            stop=self.stop,
        )

    @overload
    def child_with_cfg(
        self: MeasurementContext[list[T_ChildResult], T_RootResult, T_Cfg],
        addr: int,
        new_cfg: T_ChildCfg,
        child_type: None = None,
    ) -> MeasurementContext[T_ChildResult, T_RootResult, T_ChildCfg]: ...

    @overload
    def child_with_cfg(
        self: MeasurementContext[T_MappingResult, T_RootResult, T_Cfg],
        addr: Hashable,
        new_cfg: T_ChildCfg,
        child_type: type[T_ChildResult],
    ) -> MeasurementContext[T_ChildResult, T_RootResult, T_ChildCfg]: ...

    @overload
    def child_with_cfg(
        self: MeasurementContext[T_MappingResult, T_RootResult, T_Cfg],
        addr: Hashable,
        new_cfg: T_ChildCfg,
        child_type: None = None,
    ) -> MeasurementContext[Any, T_RootResult, T_ChildCfg]: ...

    def child_with_cfg(
        self,
        addr: int | Hashable,
        new_cfg: T_ChildCfg,
        child_type: type[T_ChildResult] | None = None,
    ) -> MeasurementContext[Any, T_RootResult, T_ChildCfg]:
        return MeasurementContext(
            root_data=self.root_data,
            cfg=deepcopy(new_cfg),
            env=self.env,
            on_update=self.on_update,
            path=self.path + (addr,),
            stop=self.stop,
        )

    @property
    def value(self) -> T_Result:
        return cast("T_Result", self._get_target())

    def set_value(self, value: T_Result) -> None:
        target = self._get_target()
        if isinstance(target, dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected dict, got {type(value)}")
            target.update(value)
        elif isinstance(target, list):
            if not isinstance(value, list):
                raise ValueError(f"Expected list, got {type(value)}")
            target.clear()
            target.extend(value)
        elif isinstance(target, np.ndarray):
            if isinstance(value, np.ndarray):
                np.copyto(dst=target, src=value)
            elif isinstance(value, Number):
                np.copyto(dst=target, src=np.asarray(value))
            else:
                raise ValueError(f"Expected NDArray or number, got {type(value)}")
        else:
            raise ValueError(f"Expected Mapping, list, or NDArray, got {type(target)}")
        self.trigger_update()

    def trigger_update(self) -> None:
        if self.on_update is not None:
            snapshot = MeasurementContext[Result, T_RootResult, T_Cfg](
                root_data=self.root_data,
                cfg=self.cfg,
                env=self.env,
                path=self.path,
                stop=self.stop,
            )
            self.on_update(snapshot)

    def _get_target(self) -> Result:
        target: Result = self.root_data
        for seg in self.path:
            if isinstance(target, Mapping):
                target = target[seg]  # type: ignore[index]
            elif isinstance(target, list):
                if not isinstance(seg, int):
                    raise ValueError(f"Expected int index for list, got {type(seg)}")
                target = target[seg]
            else:
                raise ValueError(f"Expected Mapping or list, got {type(target)}")
        return target


def context_signal_buffer(
    ctx: MeasurementContext[NDArray[Any], Any, Any],
    shape: int | tuple[int, ...],
    *,
    dtype: DTypeLike = np.complex128,
) -> SignalBuffer:
    return SignalBuffer(
        shape,
        dtype=dtype,
        on_update=lambda data: ctx.set_value(data),
        update_interval=None,
    )


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
        ctx: MeasurementContext[Any, Any, Any],
        results: Any,
        /,
    ) -> None: ...

    def init(self, dynamic_pbar: bool = False) -> None: ...

    def run(self, state: MeasurementContext[Any, Any, Any], /) -> None: ...

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
        Callable[[MeasurementContext[Any, Any, T_Cfg]], None],
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

        def plot_fn(ctx: MeasurementContext[Any, Any, T_Cfg]) -> None:
            if len(ctx.path) < 2:
                cur_tasks = list(self.measurements.keys())
            else:
                assert isinstance(ctx.path[1], str)
                cur_tasks = [ctx.path[1]]

            assert isinstance(ctx.root_data, list)
            results = merge_result_list(ctx.root_data)

            assert isinstance(results, dict)
            for cur_task in cur_tasks:
                self.measurements[cur_task].update_plotter(
                    plotters_map[cur_task], ctx, results[cur_task]
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
        run_fn: Callable[[MeasurementContext[T_RootResult, T_RootResult, T_Cfg]], None],
    ) -> T_RootResult:
        fig, plotter, plot_fn, writer = self.make_plotter()
        stop = current_stop_signal() or StopSignal()
        state: MeasurementContext[T_RootResult, T_RootResult, T_Cfg] = (
            MeasurementContext(
                root_data=init_result,
                cfg=deepcopy(cfg),
                env=env_dict,
                on_update=min_interval(plot_fn, 0.1),
                stop=stop,
            )
        )

        with plotter:
            try:
                for measurement in self.measurements.values():
                    measurement.init(dynamic_pbar=True)
                run_fn(state)
            except KeyboardInterrupt:
                state.set_stop()
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

        return state.root_data

    def _default_batch_result(self) -> dict[str, Result]:
        return {name: ms.get_default_result() for name, ms in self.measurements.items()}

    def _run_measurement_with_retries(
        self,
        measurement: T_Measurement,
        state: MeasurementContext[Any, Any, T_Cfg],
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
