from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Tuple, TypeVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.v2.runner import (
    AbsTask,
    BatchTask,
    run_with_retries,
    Result,
    RepeatOverTime,
    TaskConfig,
    TaskContextView,
    run_task,
)
from zcu_tools.experiment.v2.utils import merge_result_list
from zcu_tools.liveplot import AbsLivePlotter, MultiLivePlotter, make_plot_frame

T_PlotterDictType = TypeVar("T_PlotterDictType", bound=Mapping[str, AbsLivePlotter])


T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)
T_TaskConfig = TypeVar("T_TaskConfig", bound=TaskConfig)


class MeasurementTask(
    AbsTask[T_Result, T_RootResult, T_TaskConfig],
    Generic[T_Result, T_RootResult, T_TaskConfig, T_PlotterDictType],
):
    def num_axes(self) -> Dict[str, int]: ...

    def make_plotter(
        self, name: str, axs: Dict[str, List[Axes]]
    ) -> T_PlotterDictType: ...

    def update_plotter(
        self,
        plotters: T_PlotterDictType,
        ctx: TaskContextView,
        signals: T_Result,
    ) -> None: ...

    def save(
        self,
        filepath: str,
        iters: NDArray[np.int64],
        result: T_Result,
        comment: Optional[str],
        prefix_tag: str,
    ) -> None: ...

    def load(self, filepath: str, **kwargs) -> T_Result: ...

    def analyze(
        self, name: str, iters: NDArray[np.int64], result: T_Result, **kwargs
    ) -> None: ...


class OvernightTaskConfig(TaskConfig): ...


class OvernightBatchTask(BatchTask):
    def __init__(self, tasks, retry_time: int = 0) -> None:
        self.retry_time = retry_time

        super().__init__(tasks)

    def run(self, ctx: TaskContextView) -> None:
        if self.dynamic_pbar:
            self.task_pbar = self.make_pbar(leave=False)
        else:
            assert self.task_pbar is not None
            self.task_pbar.reset()

        for name, task in self.tasks.items():
            self.task_pbar.set_description(desc=f"Task [{str(name)}]")

            run_with_retries(
                task,
                ctx(addr=name),
                self.retry_time,
                dynamic_pbar=True,
                # raise_error=False,
            )

            self.task_pbar.update()

        if self.dynamic_pbar:
            self.task_pbar.close()
            self.task_pbar = None


class OvernightExecutor(AbsExperiment):
    def __init__(self, num_times: int, interval: float) -> None:
        super().__init__()

        self.num_times = num_times
        self.interval = interval

        self.measurements: Dict[str, MeasurementTask] = OrderedDict()

    def add_measurement(
        self, name: str, measurement: MeasurementTask
    ) -> OvernightExecutor:
        if name in self.measurements:
            raise ValueError(f"Measurement {name} already exists")
        self.measurements[name] = measurement

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
        Figure, MultiLivePlotter[Tuple[str, str]], Callable[[TaskContextView], None]
    ]:
        fig, axs_map = self.make_ax_layout()

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

            results = merge_result_list(ctx.data)

            assert isinstance(results, dict)
            for cur_task in cur_tasks:
                self.measurements[cur_task].update_plotter(
                    plotters_map[cur_task], ctx, results[cur_task]
                )

            plotter.refresh()

        return fig, plotter, plot_fn

    def run(
        self, fail_retry: int = 3, env_dict: Optional[Dict[str, Any]] = None
    ) -> Mapping[str, Result]:
        if len(self.measurements) == 0:
            raise ValueError("No measurements added")

        if env_dict is None:
            env_dict = {}

        env_dict.update(iters=np.arange(self.num_times))

        cfg = OvernightTaskConfig()

        with matplotlib.rc_context(
            {"font.size": 6, "xtick.major.size": 6, "ytick.major.size": 6}
        ):
            fig, plotter, plot_fn = self.make_plotter()

            with plotter:
                results = run_task(
                    task=RepeatOverTime(
                        name="Iter",
                        num_times=self.num_times,
                        interval=self.interval,
                        task=OvernightBatchTask(
                            self.measurements, retry_time=fail_retry
                        ),
                    ),
                    init_cfg=cfg,
                    env_dict=env_dict,
                    update_hook=plot_fn,
                )

            plt.close(fig)

        signals_dict = merge_result_list(results)

        self.last_cfg = cfg
        self.last_result = signals_dict

        return signals_dict

    def analyze(
        self,
        result: Optional[Mapping[str, Result]] = None,
        task_kwargs: Optional[Dict[str, dict]] = None,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        iters = np.arange(self.num_times)

        if task_kwargs is None:
            task_kwargs = {}

        for ms_name, ms in self.measurements.items():
            ms.analyze(ms_name, iters, result[ms_name], **task_kwargs.get(ms_name, {}))

    def save(
        self,
        filepath: str,
        results: Optional[Mapping[str, Result]] = None,
        comment: Optional[str] = None,
        prefix_tag: str = "overnight",
    ) -> None:
        if results is None:
            results = self.last_result
        assert results is not None, "no result found"

        iters = np.arange(self.num_times)

        _filepath = Path(filepath)
        for ms_name, ms in self.measurements.items():
            ms.save(
                str(_filepath.with_name(_filepath.name + f"_{ms_name}")),
                iters,
                results[ms_name],
                comment,
                prefix_tag + f"/{ms_name}",
            )

    def load(self, filepath: str, **kwargs) -> Mapping[str, Result]:
        raise NotImplementedError("Not implemented")
