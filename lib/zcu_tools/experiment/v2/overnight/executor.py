from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Generic, TypeVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.v2.runner import (
    MultiMeasurementExecutor,
    Schedule,
    ScheduleStep,
    StopSignal,
    current_stop_signal,
)
from zcu_tools.experiment.v2.utils import Result, merge_result_list
from zcu_tools.liveplot import AbsLivePlot
from zcu_tools.utils.debug import print_traceback
from zcu_tools.utils.func_tools import MinIntervalFunc

from .env import OvernightDeps, OvernightEnv

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
    def run(self, state: ScheduleStep[OvernightCfg, Any, OvernightEnv]) -> None: ...

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
        ctx: ScheduleStep[Any, Any, OvernightEnv],
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
        "MeasurementTask[Any, list[dict[str, Result]], Any]",
        OvernightCfg,
        OvernightEnv,
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
    def run(self, deps: OvernightDeps, fail_retry: int = 3) -> Mapping[str, Result]:
        if len(self.measurements) == 0:
            raise ValueError("No measurements added")

        env = OvernightEnv(
            soc=deps.soc,
            soccfg=deps.soccfg,
            iters=np.arange(self.num_times),
        )

        cfg = OvernightCfg()

        init_result = [self._default_batch_result() for _ in range(self.num_times)]

        fig, plotter, plot_fn, writer = self.make_plotter()
        stop = current_stop_signal() or StopSignal()
        result_buffer = self._make_result_buffer(init_result, plot_fn)

        try:
            with Schedule(cfg, result_buffer, env=env, stop=stop) as sched:
                with plotter:
                    try:
                        for measurement in self.measurements.values():
                            measurement.init(dynamic_pbar=True)

                        for _, iter_step in sched.repeat(
                            "Iter", self.num_times, self.interval
                        ):
                            self._run_measurement_batch(iter_step, fail_retry)
                            with MinIntervalFunc.force_execute():
                                iter_step.trigger_update()
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
            plt.close(fig)

        results = result_buffer.data

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
