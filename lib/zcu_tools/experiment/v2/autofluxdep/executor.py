from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment.utils import set_flux_in_dev_cfg
from zcu_tools.experiment.v2.runner import (
    AbsTask,
    BatchTask,
    ResultType,
    Runner,
    SoftTask,
    TaskContext,
)
from zcu_tools.experiment.v2.utils import merge_result_list
from zcu_tools.liveplot import AbsLivePlotter, MultiLivePlotter, make_plot_frame
from zcu_tools.simulate.fluxonium import FluxoniumPredictor


class AutoMeasurement:
    def num_axes(self) -> Dict[str, int]: ...

    def make_plotter(
        self, name: str, axs: Dict[str, List[plt.Axes]]
    ) -> Dict[str, AbsLivePlotter]: ...

    def generate_update_kwargs(
        self,
        plotters: Dict[str, AbsLivePlotter],
        flx_values: np.ndarray,
        signals: ResultType,
    ) -> Dict[str, tuple]: ...

    def writeback_cfg(self, i: int, ctx: TaskContext, flx_value: float) -> None: ...

    def make_task(self, soccfg, soc) -> AbsTask: ...

    def save(
        self, filepath: str, result: ResultType, comment: Optional[str], prefix_tag: str
    ) -> None: ...


T = TypeVar("T")


class AutoFluxDepExecutor:
    def __init__(
        self,
        soccfg,
        soc,
        flx_values: np.ndarray,
        predictor: FluxoniumPredictor,
        ref_flux: float,
    ) -> None:
        self.soccfg = soccfg
        self.soc = soc
        self.flx_values = flx_values
        self.predictor = predictor
        self.ref_flux = ref_flux

        self.auto_measurements: Dict[str, AutoMeasurement] = OrderedDict()

    def add_measurement(self, name: str, auto_measurement: AutoMeasurement) -> None:
        if name in self.auto_measurements:
            raise ValueError(f"Measurement {name} already exists")
        self.auto_measurements[name] = auto_measurement

    def make_ax_layout(self) -> Tuple[plt.Figure, Dict[str, Dict[str, List[plt.Axes]]]]:
        assert len(self.auto_measurements) > 0

        num_axes_map = {
            ms_name: ms.num_axes() for ms_name, ms in self.auto_measurements.items()
        }

        max_height = max(
            n for num_axes in num_axes_map.values() for n in num_axes.values()
        )
        if len(self.auto_measurements) > 3:
            max_height = max(3, max_height)

        # calculate layout
        layout: List[List[Tuple[str, str]]] = []
        for ms_name, num_axes in num_axes_map.items():
            for ax_name, ax_num in num_axes.items():
                for stack in layout:
                    if len(stack) + ax_num <= max_height:
                        cur_stack = stack
                        break
                else:
                    cur_stack = []  # new column
                    layout.append(cur_stack)
                cur_stack.extend([(ms_name, ax_name)] * ax_num)

        n_row, n_col = len(layout), max(len(stack) for stack in layout)
        fig, axs = make_plot_frame(n_row, n_col)

        # collect axes into dict
        axs_map: Dict[str, Dict[str, List[plt.Axes]]] = defaultdict(dict)
        for i, stack in enumerate(layout):
            for j, (ms_name, ax_name) in enumerate(stack):
                axs_map[ms_name][ax_name] = axs[i, j]

        return fig, axs_map

    def run(
        self, init_cfg: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, ResultType], plt.Figure]:
        assert len(self.auto_measurements) > 0

        def update_cfg_fn(i: int, ctx: TaskContext, flx_value: float) -> None:
            set_flux_in_dev_cfg(ctx.cfg["dev"], flx_value)
            for m in self.auto_measurements.values():
                m.writeback_cfg(i, ctx, flx_value)

        fig, axs_map = self.make_ax_layout()

        plotters_map = {
            ms_name: ms.make_plotter(ms_name, axs_map[ms_name])
            for ms_name, ms in self.auto_measurements.items()
        }

        T = TypeVar("T")

        def flatten_dict(d: Dict[str, Dict[str, T]]) -> Dict[Tuple[str, str], T]:
            return {(n1, n2): v for n1, d2 in d.items() for n2, v in d2.items()}

        with MultiLivePlotter(fig, flatten_dict(plotters_map)) as viewer:

            def plot_fn(ctx: TaskContext) -> None:
                signals_dict = merge_result_list(ctx.get_data())

                plot_kwargs = {
                    ms_name: ms.generate_update_kwargs(
                        plotters_map[ms_name],
                        self.flx_values,
                        signals_dict[ms_name],
                    )
                    for ms_name, ms in self.auto_measurements.items()
                }

                viewer.update(flatten_dict(plot_kwargs))

            results = Runner(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=self.flx_values,
                    update_cfg_fn=update_cfg_fn,
                    sub_task=BatchTask(
                        tasks={
                            name: ms.make_task(self.soccfg, self.soc)
                            for name, ms in self.auto_measurements.items()
                        }
                    ),
                ),
                update_hook=plot_fn,
            ).run(init_cfg)
            signals_dict = merge_result_list(results)

            return self.flx_values, signals_dict, fig

    def save(
        self,
        filepath: str,
        results: Tuple[np.ndarray, Dict[str, ResultType]],
        comment: Optional[str] = None,
        prefix_tag: str = "autoflux_dep",
    ) -> None:
        filepath = Path(filepath)
        for name, ms in self.auto_measurements.items():
            ms.save(
                filepath.with_name(filepath.name + f"_{name}"),
                results[name],
                comment,
                prefix_tag + f"/{name}",
            )
