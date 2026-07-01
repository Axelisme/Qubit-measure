from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Generic, TypeVar

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner.multi_executor import (
    MeasurementContext,
    MultiMeasurementExecutor,
)
from zcu_tools.experiment.v2.utils import Result, merge_result_list
from zcu_tools.liveplot import AbsLivePlot
from zcu_tools.progress_bar import make_pbar
from zcu_tools.utils.func_tools import MinIntervalFunc

T_PlotDict = TypeVar("T_PlotDict", bound=Mapping[str, AbsLivePlot])


T_Result = TypeVar("T_Result", bound=Result)
T_RootResult = TypeVar("T_RootResult", bound=Result)


class OvernightCfg(ExpCfgModel):
    pass


class MeasurementTask(
    ABC,
    Generic[T_Result, T_RootResult, T_PlotDict],
):
    def init(self, dynamic_pbar: bool = False) -> None: ...

    @abstractmethod
    def run(
        self, state: MeasurementContext[T_Result, T_RootResult, OvernightCfg]
    ) -> None: ...

    def cleanup(self) -> None: ...

    @abstractmethod
    def get_default_result(self) -> T_Result: ...

    @abstractmethod
    def num_axes(self) -> dict[str, int]: ...

    @abstractmethod
    def make_plotter(self, name: str, axs: dict[str, list[Axes]]) -> T_PlotDict: ...

    @abstractmethod
    def update_plotter(
        self,
        plotters: T_PlotDict,
        ctx: MeasurementContext[Any, Any, Any],
        results: T_Result,
    ) -> None: ...

    @abstractmethod
    def save(
        self,
        filepath: str,
        iters: NDArray[np.int64],
        result: T_Result,
        comment: str | None,
        prefix_tag: str,
    ) -> None: ...


class OvernightExecutor(
    MultiMeasurementExecutor[
        "MeasurementTask[Any, list[dict[str, Result]], Any]", OvernightCfg
    ]
):
    def __init__(self, num_times: int, interval: float) -> None:
        super().__init__()

        if num_times < 0:
            raise ValueError("num_times must be non-negative")
        if interval < 0.0:
            raise ValueError("interval must be non-negative")

        self.num_times = num_times
        self.interval = interval

    @matplotlib.rc_context(
        {"font.size": 6, "xtick.major.size": 6, "ytick.major.size": 6}
    )
    def run(
        self, fail_retry: int = 3, env_dict: dict[str, Any] | None = None
    ) -> Mapping[str, Result]:
        if len(self.measurements) == 0:
            raise ValueError("No measurements added")

        if env_dict is None:
            env_dict = {}

        env_dict.update(iters=np.arange(self.num_times))

        cfg = OvernightCfg()

        init_result = [self._default_batch_result() for _ in range(self.num_times)]

        def run_loop(
            root_ctx: MeasurementContext[
                list[dict[str, Result]], list[dict[str, Result]], OvernightCfg
            ],
        ) -> None:
            iter_pbar = make_pbar(
                total=self.num_times, smoothing=0, desc="Iter", leave=True
            )
            time_pbar = make_pbar(
                total=self.interval,
                smoothing=0,
                desc="Passing Time",
                leave=True,
                miniters=0.2,
                bar_format="{desc}: {bar} {n:.1f}/{total:.1f} s",
                disable=self.interval == 0.0,
            )
            start_t = time.time() - 2 * self.interval
            try:
                for i in range(self.num_times):
                    if root_ctx.is_stop():
                        break
                    while time.time() - start_t < self.interval:
                        if root_ctx.is_stop():
                            break
                        passed_time = round(time.time() - start_t, 1)
                        time_pbar.update(passed_time - time_pbar.n)
                        time.sleep(0.1)
                    time_pbar.reset()
                    if root_ctx.is_stop():
                        break

                    start_t = time.time()
                    root_ctx.env["repeat_idx"] = i
                    iter_ctx = root_ctx.child(i)
                    for name, measurement in self.measurements.items():
                        if root_ctx.is_stop():
                            break
                        measurement_ctx = iter_ctx.child(name)
                        self._run_measurement_with_retries(
                            measurement, measurement_ctx, fail_retry
                        )
                    iter_pbar.update()
                    with MinIntervalFunc.force_execute():
                        root_ctx.trigger_update()
            finally:
                iter_pbar.close()
                time_pbar.close()

        results = self._run_with_plotting(init_result, cfg, env_dict, run_loop)

        signals_dict = merge_result_list(results)

        self.last_cfg = cfg
        self.last_result = signals_dict

        return signals_dict

    def save(
        self,
        filepath: str,
        results: Mapping[str, Result] | None = None,
        comment: str | None = None,
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
