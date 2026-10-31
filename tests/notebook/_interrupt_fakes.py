"""Synthetic acquisition at real Schedule/program interruption seams."""

import signal
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.device import FakeDevice
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.program.acquisition import CancelFlagProtocol

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult


class PartialProgram:
    """Return a scalar sample, requesting stop before the selected point."""

    def __init__(self, value: float, *, interrupt: bool) -> None:

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
            signal.raise_signal(signal.SIGINT)
        return np.array(self.value, dtype=np.complex128)

    def acquire_decimated(self, soc: object, **kwargs: object) -> object:
        """Reject decimated acquisition, which these contract tests do not use."""
        raise NotImplementedError


class RoundInterruptedProgram:
    """Emit one completed-round partial, then request cooperative stop.

    The caller requests two rounds. The completed first round contains one
    measured voltage and two unmeasured NaNs; acquire returns the available data.
    """

    def acquire(
        self,
        soc: object,
        *,
        round_hook: Callable[[int, NDArray[np.complex128], CancelFlagProtocol], object],
        cancel_flag: CancelFlagProtocol,
        progress: bool = False,
        **kwargs: object,
    ) -> NDArray[np.complex128]:
        """Deliver partial data through the hook, then request stop and return."""
        partial = np.array([1.0, np.nan, np.nan], dtype=np.complex128)
        round_hook(1, partial, cancel_flag)
        signal.raise_signal(signal.SIGINT)
        return partial

    def acquire_decimated(self, soc: object, **kwargs: object) -> object:
        """Reject decimated acquisition, which this contract test does not use."""
        raise NotImplementedError


def identity_signal(raw: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Keep synthetic program signals in working volts."""
    return raw


class InterruptedCore(SavingCore):
    """Saveable three-point scan interrupted in its body or program acquire.

    mode selects software-body, scalar program, completed-round program, or
    pre-Schedule device operation. context retains the latest RunContext for public signal observation.
    """

    def __init__(
        self, mode: Literal["software", "program", "program-round", "device"]
    ) -> None:
        self.mode = mode
        self.context: RunContext | None = None

    def run(self, config: SavingCfg, *, context: RunContext | None) -> SavingResult:
        """Acquire through the real runtime; device failure precedes its scope."""
        if context is None:
            raise ValueError("RunContext required")
        self.context = context
        if self.mode == "device" and config.dev is not None:
            device = context.devices[config.dev["probe"]]
            if not isinstance(device, FakeDevice):
                raise TypeError("This synthetic experiment requires FakeDevice")
            device.set_value(0.5)
        buffer = SignalBuffer(3)
        with Schedule(config, buffer, scope=context.scope) as sched:
            if self.mode == "program-round":
                sched.prog_builder(context.soc, context.soccfg).run_program(
                    RoundInterruptedProgram(), rounds=2, raw2signal_fn=identity_signal
                )
            else:
                for index, step in sched.scan("point", range(3)):
                    if self.mode == "software" and index == 1:
                        signal.raise_signal(signal.SIGINT)
                        break
                    if self.mode == "program":
                        step.prog_builder(context.soc, context.soccfg).run_program(
                            PartialProgram(float(index + 1), interrupt=index == 1),
                            raw2signal_fn=identity_signal,
                        )
                    else:
                        buffer[step].set(np.array(index + 1, dtype=np.complex128))
        return SavingResult(
            frequency=np.array([1000.0, 1001.0, 1002.0]),
            signal=np.asarray(buffer.data, dtype=np.complex128),
        )
