"""Synthetic acquisition at real Schedule/program/executor interruption seams."""

from collections.abc import Callable
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils.device import setup_devices
from zcu_tools.experiment.v2.runtime.multi_executor import MultiMeasurementExecutor
from zcu_tools.experiment.v2.runtime.result_tree import ResultUpdateEvent
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    ScheduleStep,
    SignalBuffer,
)
from zcu_tools.experiment.v2.runtime.task import MeasurementTask
from zcu_tools.plotting.plots import LinePlot, Plots
from zcu_tools.program.acquisition import CancelFlagProtocol
from zcu_tools.program.v2 import ProgramV2Cfg

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult


class PartialProgram:
    """Return one scalar sample, or interrupt before this point's first round."""

    def __init__(self, value: float, *, interrupt: bool) -> None:
        self.cfg_model = ProgramV2Cfg(rounds=1)
        self.value = value
        self.interrupt = interrupt

    def acquire(
        self,
        soc: object,
        *,
        round_hook: Callable[[int, NDArray[np.complex128], CancelFlagProtocol], object],
        cancel_flag: CancelFlagProtocol,
        progress: bool = False,
        **kwargs: object,
    ) -> NDArray[np.complex128]:
        """Emulate program output; the interrupt path returns no completed round."""
        if self.interrupt:
            raise KeyboardInterrupt("acquire stopped")
        return np.array(self.value, dtype=np.complex128)

    def acquire_decimated(self, soc: object, **kwargs: object) -> object:
        """Reject decimated acquisition, which these contract tests do not use."""
        raise NotImplementedError


class RoundInterruptedProgram:
    """Emit one completed-round partial, then interrupt the same acquire.

    cfg_model declares two rounds. The completed first round contains one
    measured voltage and two unmeasured NaNs; acquire never normally returns.
    """

    def __init__(self) -> None:
        self.cfg_model = ProgramV2Cfg(rounds=2)

    def acquire(
        self,
        soc: object,
        *,
        round_hook: Callable[[int, NDArray[np.complex128], CancelFlagProtocol], object],
        cancel_flag: CancelFlagProtocol,
        progress: bool = False,
        **kwargs: object,
    ) -> NDArray[np.complex128]:
        """Deliver the partial through the caller's hook, then interrupt."""
        partial = np.array([1.0, np.nan, np.nan], dtype=np.complex128)
        round_hook(1, partial, cancel_flag)
        raise KeyboardInterrupt("acquire stopped after completed round")

    def acquire_decimated(self, soc: object, **kwargs: object) -> object:
        """Reject decimated acquisition, which this contract test does not use."""
        raise NotImplementedError


def identity_signal(raw: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Keep synthetic program signals in working volts."""
    return raw


class InterruptedCore(SavingCore):
    """Saveable three-point scan interrupted in its body or program acquire.

    mode selects software-body, scalar program, completed-round program, or
    pre-Schedule device setup. context retains the latest RunContext for public signal observation.
    """

    def __init__(
        self, mode: Literal["software", "program", "program-round", "setup"]
    ) -> None:
        self.mode = mode
        self.context: RunContext | None = None

    def run(self, config: SavingCfg, *, context: RunContext | None) -> SavingResult:
        """Acquire through the real runtime; setup failure precedes its scope."""
        if context is None:
            raise ValueError("RunContext required")
        self.context = context
        if self.mode == "setup":
            setup_devices(config, context.devices, cancel_signal=context.cancel_signal)
        buffer = SignalBuffer(3)
        with Schedule(config, buffer, stop=context.cancel_signal) as sched:
            if self.mode == "program-round":
                sched.prog_builder(
                    context.soc, context.soccfg, cfg=ProgramV2Cfg()
                ).run_program(RoundInterruptedProgram(), raw2signal_fn=identity_signal)
            else:
                for index, step in sched.scan("point", range(3)):
                    if self.mode == "software" and index == 1:
                        raise KeyboardInterrupt("scan body stopped")
                    if self.mode == "program":
                        step.prog_builder(
                            context.soc, context.soccfg, cfg=ProgramV2Cfg()
                        ).run_program(
                            PartialProgram(float(index + 1), interrupt=index == 1),
                            raw2signal_fn=identity_signal,
                        )
                    else:
                        buffer[step].set(np.array(index + 1, dtype=np.complex128))
        return SavingResult(
            frequency=np.array([1000.0, 1001.0, 1002.0]),
            signal=np.asarray(buffer.data, dtype=np.complex128),
        )


class InterruptedMeasurement(
    MeasurementTask[
        SavingCfg,
        RunContext,
        NDArray[np.complex128],
        dict[str, LinePlot],
        NDArray[np.float64],
    ]
):
    """Executor leaf completes point zero and interrupts at point one."""

    def init(self, dynamic_pbar: bool = False) -> None:
        """No synthetic hardware needs initialization."""

    def cleanup(self) -> None:
        """No synthetic hardware needs cleanup."""

    def run(self, state: ScheduleStep[SavingCfg, object, RunContext]) -> None:
        """Write a scalar voltage, or interrupt before writing point one."""
        index = state.path[0]
        if index == 1:
            raise KeyboardInterrupt("executor acquire stopped")
        state.set_data(np.array(1.0, dtype=np.complex128), flush=True)

    def get_default_result(self) -> NDArray[np.complex128]:
        """Initialize each unmeasured scalar to NaN volts."""
        return np.array(np.nan, dtype=np.complex128)

    def num_axes(self) -> dict[str, int]:
        """The synthetic signal has one plot axis."""
        return {"signal": 1}

    def make_plotter(
        self, name: str, axs: dict[str, list[Axes]], *, plots: Plots, figure_name: str
    ) -> dict[str, LinePlot]:
        """Create the runtime-owned voltage line on its assigned axis."""
        return {
            "signal": plots.liveplot_1d(
                figure_name, "point", "V", axes=axs["signal"][0]
            )
        }

    def update_plotter(
        self,
        plotters: dict[str, LinePlot],
        event: ResultUpdateEvent[RunContext, NDArray[np.complex128]],
        result: NDArray[np.complex128],
    ) -> None:
        """Update the executor's merged partial voltage magnitude."""
        plotters["signal"].update(
            np.arange(result.size, dtype=np.float64), np.abs(result), refresh=False
        )

    def save(
        self,
        filepath: str,
        axis_values: NDArray[np.float64],
        result: NDArray[np.complex128],
        comment: str | None,
        prefix_tag: str,
    ) -> None:
        """Unused leaf persistence; the bound Notebook core owns save_run."""
        raise NotImplementedError


class InterruptedExecutorCore(
    MultiMeasurementExecutor[
        InterruptedMeasurement, SavingCfg, RunContext, NDArray[np.float64]
    ],
    SavingCore,
):
    """Saveable core using the real executor's result and stop lifecycle."""

    def __init__(self) -> None:
        super().__init__()
        self.context: RunContext | None = None
        self.add_measurements({"scan": InterruptedMeasurement()})

    def run(self, config: SavingCfg, *, context: RunContext | None) -> SavingResult:
        """Run three executor points, retaining the interrupted leaf's partial."""
        if context is None:
            raise ValueError("RunContext required")
        self.context = context
        frequency = np.array([1000.0, 1001.0, 1002.0])

        def run_loop(sched: Schedule[SavingCfg, RunContext]) -> None:
            for _, step in sched.scan("point", frequency):
                self._run_measurement_batch(step, retry_time=0)

        results = self._run(
            cfg=config,
            env=context,
            stop=context.cancel_signal,
            plots=context.plots,
            outer_values=frequency,
            run_loop=run_loop,
        )
        return SavingResult(
            frequency=frequency,
            signal=np.asarray(results["scan"], dtype=np.complex128),
        )
