from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Generic, TypeVar

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner import AbsTask, Result, RetryBatchTask, TaskState
from zcu_tools.experiment.v2.runner.multi_executor import MultiMeasurementExecutor
from zcu_tools.experiment.v2.utils import merge_result_list
from zcu_tools.liveplot import AbsLivePlot

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

        batch: RetryBatchTask[str, Result, list[dict[str, Result]], OvernightCfg] = (
            RetryBatchTask(self.measurements, retry_time=fail_retry)
        )
        task = batch.repeat("Iter", self.num_times, self.interval)

        results = self._run_with_plotting(task, cfg, env_dict)

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
