from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
    Sequence,
    TypeVar,
)

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner import (
    AbsTask,
    BatchTask,
    Result,
    TaskState,
    run_task,
    run_with_retries,
)
from zcu_tools.experiment.v2.utils import merge_result_list
from zcu_tools.liveplot import AbsLivePlot, MultiLivePlot, make_plot_frame

T_PlotDict = TypeVar("T_PlotDict", bound=Mapping[str, AbsLivePlot])


T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)


class OvernightCfg(ExpCfgModel):
    pass


class MeasurementTask(
    AbsTask[T_Result, T_RootResult, OvernightCfg],
    Generic[T_Result, T_RootResult, T_PlotDict],
):
    @abstractmethod
    def num_axes(self) -> dict[str, int]: ...

    @abstractmethod
    def make_plotter(self, name: str, axs: dict[str, list[Axes]]) -> T_PlotDict: ...

    @abstractmethod
    def update_plotter(
        self,
        plotters: T_PlotDict,
        ctx: TaskState,
        signals: T_Result,
    ) -> None: ...

    @abstractmethod
    def save(
        self,
        filepath: str,
        iters: NDArray[np.int64],
        result: T_Result,
        comment: Optional[str],
        prefix_tag: str,
    ) -> None: ...


class OvernightBatchTask(BatchTask[str, Result, list[dict[str, Result]], OvernightCfg]):
    def __init__(self, tasks, retry_time: int = 0) -> None:
        self.retry_time = retry_time

        super().__init__(tasks)

    def run(self, ctx) -> None:
        if self.dynamic_pbar:
            self.task_pbar = self._build_pbar(leave=False)
        else:
            assert self.task_pbar is not None
            self.task_pbar.reset()

        for name, task in self.tasks.items():
            assert self.task_pbar is not None
            self.task_pbar.set_description(f"Task [{str(name)}]")

            run_with_retries(
                task,
                ctx.child(name, child_type=Result),
                self.retry_time,
                dynamic_pbar=True,
                # raise_error=False,
            )

            self.task_pbar.update()

        if self.dynamic_pbar:
            self.task_pbar.close()
            self.task_pbar = None


class OvernightExecutor:
    def __init__(self, num_times: int, interval: float) -> None:
        super().__init__()

        self.num_times = num_times
        self.interval = interval

        self.measurements: OrderedDict[
            str,
            MeasurementTask[Any, Sequence[Mapping[str, Result]], Any],
        ] = OrderedDict()

    def add_measurements(
        self,
        measurements: dict[
            str, MeasurementTask[Any, Sequence[Mapping[str, Result]], Any]
        ],
    ) -> Self:
        for name, measurement in measurements.items():
            if name in self.measurements:
                raise ValueError(f"Measurement {name} already exists")
            self.measurements[name] = measurement

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
        Callable[[TaskState[Result, list[dict[str, Result]], OvernightCfg]], None],
    ]:
        fig, axs_map = self.make_ax_layout()

        plotters_map = {
            ms_name: ms.make_plotter(ms_name, axs_map[ms_name])
            for ms_name, ms in self.measurements.items()
        }

        T = TypeVar("T")

        def flatten_dict(d: Mapping[str, Mapping[str, T]]) -> dict[tuple[str, str], T]:
            return {(n1, n2): v for n1, d2 in d.items() for n2, v in d2.items()}

        plotter = MultiLivePlot(fig, flatten_dict(plotters_map))

        def plot_fn(
            ctx: TaskState[Result, list[dict[str, Result]], OvernightCfg],
        ) -> None:
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

            plotter.refresh()

        return fig, plotter, plot_fn

    @matplotlib.rc_context(
        {"font.size": 6, "xtick.major.size": 6, "ytick.major.size": 6}
    )
    def run(
        self, fail_retry: int = 3, env_dict: Optional[dict[str, Any]] = None
    ) -> Mapping[str, Result]:
        if len(self.measurements) == 0:
            raise ValueError("No measurements added")

        if env_dict is None:
            env_dict = {}

        env_dict.update(iters=np.arange(self.num_times))

        cfg = OvernightCfg()

        fig, plotter, plot_fn = self.make_plotter()

        with plotter:
            results = run_task(
                task=OvernightBatchTask(
                    self.measurements, retry_time=fail_retry
                ).repeat("Iter", self.num_times, self.interval),
                init_cfg=cfg,
                env_dict=env_dict,
                on_update=plot_fn,
            )

        plt.close(fig)

        signals_dict = merge_result_list(results)

        self.last_cfg = cfg
        self.last_result = signals_dict

        return signals_dict

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
