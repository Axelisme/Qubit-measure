from __future__ import annotations

import shutil
from collections import OrderedDict, UserDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Tuple, TypeVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg
from zcu_tools.experiment.v2.runner import (
    AbsTask,
    BatchTask,
    Result,
    SoftTask,
    TaskConfig,
    TaskContextView,
    run_task,
)
from zcu_tools.experiment.v2.utils import merge_result_list
from zcu_tools.liveplot import (
    AbsLivePlotter,
    MultiLivePlotter,
    grab_frame_with_instant_plot,
    make_plot_frame,
)
from zcu_tools.simulate.fluxonium import FluxoniumPredictor

T_PlotterDictType = TypeVar("T_PlotterDictType", bound=Mapping[str, AbsLivePlotter])


T_ResultType = TypeVar("T_ResultType", bound=Result)
T_RootResultType = TypeVar("T_RootResultType", bound=Result)
T_TaskConfigType = TypeVar("T_TaskConfigType", bound=TaskConfig)


class MeasurementTask(
    AbsTask[T_ResultType, T_RootResultType, T_TaskConfigType],
    Generic[T_ResultType, T_RootResultType, T_TaskConfigType, T_PlotterDictType],
):
    def num_axes(self) -> Dict[str, int]: ...

    def make_plotter(
        self, name: str, axs: Dict[str, List[Axes]]
    ) -> T_PlotterDictType: ...

    def update_plotter(
        self,
        plotters: T_PlotterDictType,
        ctx: TaskContextView,
        signals: T_ResultType,
    ) -> None: ...

    def save(
        self,
        filepath: str,
        flx_values: NDArray[np.float64],
        result: T_ResultType,
        comment: Optional[str],
        prefix_tag: str,
    ) -> None: ...

    def load(self, filepath: str, **kwargs) -> T_ResultType: ...


class FluxDepTaskConfig(TaskConfig):
    dev: Mapping[str, DeviceInfo]


class FluxDepInfoDict(UserDict):
    def __init__(self, initialdata: Optional[Mapping[str, Any]] = None) -> None:
        self.first_info: Dict[str, Any] = {}
        self.last_info: Dict[str, Any] = {}
        super().__init__(initialdata)

    @property
    def last(self) -> Dict[str, Any]:
        return self.last_info

    @property
    def first(self) -> Dict[str, Any]:
        return self.first_info

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self.first_info.setdefault(key, deepcopy(value))
        self.last_info[key] = deepcopy(value)


class FluxDepExecutor(AbsExperiment):
    def __init__(self, flx_values: NDArray[np.float64]) -> None:
        super().__init__()

        self.flx_values = flx_values

        self.record_path = None
        self.measurements: Dict[str, MeasurementTask] = OrderedDict()

    def add_measurement(
        self, name: str, measurement: MeasurementTask
    ) -> FluxDepExecutor:
        if name in self.measurements:
            raise ValueError(f"Measurement {name} already exists")
        self.measurements[name] = measurement

        return self

    def record_animation(self, path: str) -> FluxDepExecutor:
        if self.record_path is not None:
            raise ValueError("Animation recording path already set")
        self.record_path = Path(path)
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def make_ax_layout(self) -> Tuple[Figure, Dict[str, Dict[str, List[Axes]]]]:
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
        axs_map: Dict[str, Dict[str, List[Axes]]] = defaultdict(dict)
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
    ) -> Tuple[
        Figure,
        MultiLivePlotter[Tuple[str, str]],
        Callable[[TaskContextView], None],
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

        def flatten_dict(d: Mapping[str, Mapping[str, T]]) -> Dict[Tuple[str, str], T]:
            return {(n1, n2): v for n1, d2 in d.items() for n2, v in d2.items()}

        plotter = MultiLivePlotter(fig, flatten_dict(plotters_map))

        def plot_fn(ctx: TaskContextView) -> None:
            if len(ctx._addr_stack) < 2:
                cur_tasks = list(self.measurements.keys())
            else:
                assert isinstance(ctx._addr_stack[1], str)
                cur_tasks = [ctx._addr_stack[1]]

            results = ctx.get_data()
            assert isinstance(results, list)
            results = merge_result_list(results)

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
        dev_cfg: Mapping[str, DeviceInfo],
        predictor: FluxoniumPredictor,
        env_dict: Optional[Dict[str, Any]] = None,
    ) -> Mapping[str, Result]:
        if len(self.measurements) == 0:
            raise ValueError("No measurements added")

        if env_dict is None:
            env_dict = {}

        cfg = FluxDepTaskConfig(dev=dev_cfg)

        env_dict.update(
            flx_values=self.flx_values,
            predictor=predictor,
            info=FluxDepInfoDict(),
        )

        def update_fn(i: int, ctx: TaskContextView, flx: float) -> None:
            info: FluxDepInfoDict = ctx.env_dict["info"]
            predictor: FluxoniumPredictor = ctx.env_dict["predictor"]

            info.clear()  # clear current info dict

            info["flx_value"] = flx
            info["flx_idx"] = i

            info["cur_m"] = predictor.predict_matrix_element(flx)
            info["m_ratio"] = info["cur_m"] / info.first["cur_m"]

            set_flux_in_dev_cfg(ctx.cfg["dev"], flx, label="flux_dev")

        with matplotlib.rc_context(
            {"font.size": 6, "xtick.major.size": 6, "ytick.major.size": 6}
        ):
            fig, plotter, plot_fn, writer = self.make_plotter()

            with plotter:
                try:
                    results = run_task(
                        task=SoftTask(
                            sweep_name="flux",
                            sweep_values=self.flx_values.tolist(),
                            update_cfg_fn=update_fn,
                            sub_task=BatchTask(self.measurements),
                        ),
                        init_cfg=cfg,
                        env_dict=env_dict,
                        update_hook=plot_fn,
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

    def analyze(self, result: Optional[Mapping[str, Result]] = None) -> None:
        raise NotImplementedError("Not implemented")

    def save(
        self,
        filepath: str,
        results: Optional[Mapping[str, Result]] = None,
        comment: Optional[str] = None,
        prefix_tag: str = "autoflux_dep",
    ) -> None:
        if results is None:
            results = self.last_result
        assert results is not None, "no result found"

        _filepath = Path(filepath)
        for ms_name, ms in self.measurements.items():
            ms.save(
                str(_filepath.with_name(_filepath.name + f"_{ms_name}")),
                self.flx_values,
                results[ms_name],
                comment,
                prefix_tag + f"/{ms_name}",
            )

    def load(self, filepath: str, **kwargs) -> Mapping[str, Result]:
        _filepath = Path(filepath)

        loaded: Dict[str, Result] = {}
        for ms_name, ms in self.measurements.items():
            ms_filepath = str(_filepath.with_name(_filepath.name + f"_{ms_name}"))
            loaded[ms_name] = ms.load(ms_filepath, **kwargs)

        self.last_cfg = None
        self.last_result = loaded

        return loaded
