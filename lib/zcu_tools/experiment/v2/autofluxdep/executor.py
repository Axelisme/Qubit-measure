from __future__ import annotations

import shutil
from abc import abstractmethod
from collections import OrderedDict, UserDict, defaultdict
from copy import deepcopy
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    Self,
    TypedDict,
    TypeVar,
)

from zcu_tools.device import DeviceInfo, GlobalDeviceManager
from zcu_tools.experiment.utils import set_flux_in_dev_cfg
from zcu_tools.experiment.v2.runner import (
    AbsTask,
    BatchTask,
    Result,
    TaskState,
    run_task,
    run_with_retries,
)
from zcu_tools.experiment.v2.utils import merge_result_list
from zcu_tools.liveplot import (
    AbsLivePlotter,
    MultiLivePlotter,
    grab_frame_with_instant_plot,
    make_plot_frame,
)
from zcu_tools.simulate.fluxonium import FluxoniumPredictor

T_PlotterDict = TypeVar("T_PlotterDict", bound=Mapping[str, AbsLivePlotter])


T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)


class MeasurementTask(
    AbsTask[T_Result, T_RootResult], Generic[T_Result, T_RootResult, T_PlotterDict]
):
    @abstractmethod
    def num_axes(self) -> dict[str, int]: ...

    @abstractmethod
    def make_plotter(self, name: str, axs: dict[str, list[Axes]]) -> T_PlotterDict: ...

    @abstractmethod
    def update_plotter(
        self, plotters: T_PlotterDict, ctx: TaskState, signals: T_Result
    ) -> None: ...

    @abstractmethod
    def save(
        self,
        filepath: str,
        flux_values: NDArray[np.float64],
        result: T_Result,
        comment: Optional[str],
        prefix_tag: str,
    ) -> None: ...


class FluxDepCfg(TypedDict):
    dev: dict[str, DeviceInfo]


class FluxDepInfoDict(UserDict):
    def __init__(self, initialdata: Optional[Mapping[str, Any]] = None) -> None:
        self.first_info: dict[str, Any] = {}
        self.last_info: dict[str, Any] = {}
        super().__init__(initialdata)

    @property
    def last(self) -> dict[str, Any]:
        return self.last_info

    @property
    def first(self) -> dict[str, Any]:
        return self.first_info

    def __setitem__(self, key: str, item: Any) -> None:
        super().__setitem__(key, item)
        self.first_info.setdefault(key, deepcopy(item))
        self.last_info[key] = deepcopy(item)


class FluxDepBatchTask(BatchTask):
    def __init__(self, tasks, retry_time: int = 0) -> None:
        self.retry_time = retry_time

        super().__init__(tasks)

    def run(self, ctx) -> None:
        if self.dynamic_pbar:
            self.task_pbar = self.make_pbar(leave=False)
        else:
            assert self.task_pbar is not None
            self.task_pbar.reset()

        for name, task in self.tasks.items():
            self.task_pbar.set_description(desc=f"Task [{str(name)}]")

            run_with_retries(
                task,
                ctx.child(name),
                self.retry_time,
                dynamic_pbar=True,
                raise_error=False,
            )

            self.task_pbar.update()

        if self.dynamic_pbar:
            self.task_pbar.close()
            self.task_pbar = None


class FluxDepExecutor:
    def __init__(self, flux_values: NDArray[np.float64]) -> None:
        super().__init__()

        self.flux_values = flux_values

        self.record_path = None
        self.measurements: dict[str, MeasurementTask] = OrderedDict()

    def add_measurements(self, measurements: dict[str, MeasurementTask]) -> Self:
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
            n_row, n_col, figsize=(min(14, 3.5 * n_col), min(8, 2.5 * n_row))
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
        MultiLivePlotter[tuple[str, str]],
        Callable[[TaskState], None],
        Optional[FFMpegWriter],
    ]:
        fig, axs_map = self.make_ax_layout()

        if self.record_path is not None:
            if shutil.which("ffmpeg") is None:
                raise RuntimeError(
                    "FFmpeg is not found. Please install FFmpeg and add it to your PATH "
                    "to record animations."
                )
            writer = FFMpegWriter(fps=30)
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

        plotter = MultiLivePlotter(fig, flatten_dict(plotters_map))

        def plot_fn(ctx: TaskState) -> None:
            if len(ctx.path) < 2:
                cur_tasks = list(self.measurements.keys())
            else:
                assert isinstance(ctx.path[1], str)
                cur_tasks = [ctx.path[1]]

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

    def run(
        self,
        dev_cfg: dict[str, DeviceInfo],
        predictor: FluxoniumPredictor,
        env_dict: Optional[dict[str, Any]] = None,
        retry_time: int = 3,
    ) -> Mapping[str, Result]:
        if len(self.measurements) == 0:
            raise ValueError("No measurements added")

        if env_dict is None:
            env_dict = {}

        cfg = FluxDepCfg(dev=dev_cfg)

        env_dict.update(
            flux_values=self.flux_values,
            predictor=predictor,
            info=FluxDepInfoDict(),
        )

        def update_fn(i: int, ctx: TaskState, flux: float) -> None:
            info: FluxDepInfoDict = ctx.env["info"]
            predictor: FluxoniumPredictor = ctx.env["predictor"]

            info.clear()  # clear current info dict

            info["flux_value"] = flux
            info["flux_idx"] = i

            info["cur_m"] = predictor.predict_matrix_element(flux)
            info["m_ratio"] = info["cur_m"] / info.first["cur_m"]

            set_flux_in_dev_cfg(ctx.cfg["dev"], flux, label="flux_dev")

        set_flux_in_dev_cfg(cfg["dev"], self.flux_values[0], label="flux_dev")
        GlobalDeviceManager.setup_devices(cfg["dev"], progress=True)

        with matplotlib.rc_context(
            {"font.size": 6, "xtick.major.size": 6, "ytick.major.size": 6}
        ):
            fig, plotter, plot_fn, writer = self.make_plotter()

            with plotter:
                try:
                    results = run_task(
                        task=FluxDepBatchTask(
                            self.measurements,
                            retry_time=retry_time,
                        ).scan(
                            "flux",
                            self.flux_values.tolist(),
                            before_each=update_fn,
                        ),
                        init_cfg=cfg,
                        env_dict=env_dict,
                        on_update=plot_fn,
                    )

                finally:
                    if self.record_path is not None:
                        assert writer is not None
                        writer.finish()
            plt.close(fig)

        signals_dict = merge_result_list(results)

        self.last_cfg = cfg
        self.last_result = signals_dict

        return signals_dict

    def analyze(
        self, *args, result: Optional[Mapping[str, Result]] = None, **kwargs
    ) -> None:
        raise NotImplementedError("Not implemented")

    def save(
        self,
        filepath: str,
        result: Optional[Mapping[str, Result]] = None,
        comment: Optional[str] = None,
        prefix_tag: str = "autoflux_dep",
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        _filepath = Path(filepath)
        for ms_name, ms in self.measurements.items():
            ms.save(
                str(_filepath.with_name(_filepath.name + f"_{ms_name}")),
                self.flux_values,
                result[ms_name],
                comment,
                prefix_tag + f"/{ms_name}",
            )

    def load(self, filepath: str, **kwargs) -> Mapping[str, Result]:
        raise NotImplementedError("Not implemented")
