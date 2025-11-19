from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.utils import set_flux_in_dev_cfg
from zcu_tools.experiment.v2.runner import (
    AbsTask,
    BatchTask,
    ResultType,
    Runner,
    SoftTask,
    T_ResultType,
    TaskContext,
)
from zcu_tools.experiment.v2.utils import merge_result_list
from zcu_tools.liveplot import AbsLivePlotter, MultiLivePlotter, make_plot_frame
from zcu_tools.utils.func_tools import MinIntervalFunc

T_PlotterDictType = TypeVar("T_PlotterDictType", bound=Dict[str, AbsLivePlotter])


class MeasurementTask(AbsTask, Generic[T_ResultType, T_PlotterDictType]):
    def num_axes(self) -> Dict[str, int]: ...

    def make_plotter(
        self, name: str, axs: Dict[str, List[plt.Axes]]
    ) -> T_PlotterDictType: ...

    def update_plotter(
        self,
        plotters: T_PlotterDictType,
        ctx: TaskContext,
        signals: T_ResultType,
    ) -> None: ...

    def save(
        self,
        filepath: str,
        flx_values: np.ndarray,
        result: T_ResultType,
        comment: Optional[str],
        prefix_tag: str,
    ) -> None: ...


class SmartBatchTask(BatchTask[str]):
    def __init__(self, tasks: Dict[str, MeasurementTask]) -> None:
        super().__init__(tasks)

    def run(self, ctx: TaskContext) -> None:
        if self.dynamic_pbar:
            self.task_pbar = self.make_pbar(leave=False)
        else:
            assert self.task_pbar is not None
            self.task_pbar.reset()

        for name, task in self.tasks.items():
            self.task_pbar.set_description(desc=f"Task [{str(name)}]")

            cur_ctx = ctx(addr=name)

            task.run(cur_ctx)
            self.task_pbar.update()
            with MinIntervalFunc.force_execute():
                cur_ctx.update_hook(cur_ctx)  # force refresh current task data

            cur_result = cur_ctx.get_current_data()
            assert isinstance(cur_result, dict)

            if not cur_result["success"]:
                self.task_pbar.set_description(desc=f"Task [{str(name)}] failed")
                break

        if self.dynamic_pbar:
            self.task_pbar.close()
            self.task_pbar = None


class FluxDepExecutor:
    def __init__(
        self,
        flx_values: np.ndarray,
        flux_dev_name: str,
        flux_dev_cfg: DeviceInfo,
        env_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.flx_values = flx_values
        self.flux_dev_name = flux_dev_name
        self.flux_dev_cfg = flux_dev_cfg
        self.env_dict = {} if env_dict is None else env_dict

        self.env_dict["flx_values"] = self.flx_values

        self.measurements: Dict[str, MeasurementTask] = OrderedDict()

    def add_measurement(
        self, name: str, measurement: MeasurementTask
    ) -> FluxDepExecutor:
        if name in self.measurements:
            raise ValueError(f"Measurement {name} already exists")
        self.measurements[name] = measurement

        return self

    def make_ax_layout(self) -> Tuple[plt.Figure, Dict[str, Dict[str, List[plt.Axes]]]]:
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
        axs_map: Dict[str, Dict[str, List[plt.Axes]]] = defaultdict(dict)
        i, j = 0, 0
        for ms_name, num_axes in num_axes_map.items():
            for ax_name, ax_num in num_axes.items():
                for _ in range(ax_num):
                    axs_map[ms_name].setdefault(ax_name, []).append(axs[i, j])
                    j += 1
                    if j == n_col:
                        j = 0
                        i += 1

        return fig, axs_map

    def make_plotter(
        self,
    ) -> Tuple[plt.Figure, MultiLivePlotter[str], Callable[[TaskContext], None]]:
        fig, axs_map = self.make_ax_layout()

        plotters_map = {
            ms_name: ms.make_plotter(ms_name, axs_map[ms_name])
            for ms_name, ms in self.measurements.items()
        }

        T = TypeVar("T")

        def flatten_dict(d: Dict[str, Dict[str, T]]) -> Dict[Tuple[str, str], T]:
            return {(n1, n2): v for n1, d2 in d.items() for n2, v in d2.items()}

        plotter = MultiLivePlotter(fig, flatten_dict(plotters_map))

        def plot_fn(ctx: TaskContext) -> None:
            if len(ctx.addr_stack) < 2:
                cur_tasks = list(self.measurements.keys())
            else:
                cur_tasks = [ctx.addr_stack[1]]

            results = merge_result_list(ctx.get_data())

            for cur_task in cur_tasks:
                self.measurements[cur_task].update_plotter(
                    plotters_map[cur_task], ctx, results[cur_task]
                )

            plotter.refresh()

        return fig, plotter, plot_fn

    def run(self) -> Tuple[Dict[str, ResultType], plt.Figure]:
        if len(self.measurements) == 0:
            raise ValueError("No measurements added")

        def update_fn(i: int, ctx: TaskContext, flx: float) -> None:
            set_flux_in_dev_cfg(ctx.cfg["dev"], flx)

            ctx.env_dict["flx_idx"] = i
            ctx.env_dict["flx_value"] = flx

        with matplotlib.rc_context(
            {"font.size": 6, "xtick.major.size": 6, "ytick.major.size": 6}
        ):
            fig, plotter, plot_fn = self.make_plotter()

            with plotter:
                results = Runner(
                    task=SoftTask(
                        sweep_name="flux",
                        sweep_values=self.flx_values,
                        update_cfg_fn=update_fn,
                        sub_task=SmartBatchTask(self.measurements),
                    ),
                    update_hook=plot_fn,
                ).run(
                    init_cfg={
                        "dev": {
                            self.flux_dev_name: self.flux_dev_cfg,
                        },
                    },
                    env_dict=self.env_dict,
                )
                signals_dict = merge_result_list(results)

        self.last_result = signals_dict

        return signals_dict, fig

    def save(
        self,
        filepath: str,
        results: Dict[str, ResultType] = None,
        comment: Optional[str] = None,
        prefix_tag: str = "autoflux_dep",
    ) -> None:
        if results is None:
            results = self.last_result
        assert results is not None, "no result found"

        filepath = Path(filepath)
        for ms_name, ms in self.measurements.items():
            ms.save(
                filepath.with_name(filepath.name + f"_{ms_name}"),
                self.flx_values,
                results[ms_name],
                comment,
                prefix_tag + f"/{ms_name}",
            )
