"""Builder and callback defects propagate instead of becoming acquire failures."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.runtime import AcquireOptions, Schedule, SignalBuffer
from zcu_tools.program.acquisition import CancelFlagProtocol
from zcu_tools.program.v2 import ProgramV2Cfg

from ._flow_cfg import make_flow_cfg
from ._flow_fakes import FakeModule


@pytest.mark.parametrize("failure_site", ["build", "round", "final", "update"])
@pytest.mark.parametrize("decimated", [False, True])
def test_non_acquire_failures_raise_without_retry_or_failed_outcome(
    failure_site: str, decimated: bool
) -> None:
    cause = RuntimeError(f"broken {failure_site}")
    builds: list[int] = []
    acquires: list[int] = []

    class Program:
        def __init__(
            self, soccfg: object, cfg: ProgramV2Cfg, **kwargs: object
        ) -> None:
            builds.append(1)
            if failure_site == "build":
                raise cause
            self.cfg_model = cfg

        def acquire(
            self,
            soc: object,
            *,
            progress: bool,
            round_hook: Callable[
                [int, NDArray[np.float64], CancelFlagProtocol], object
            ],
            cancel_flag: CancelFlagProtocol,
            **kwargs: object,
        ) -> NDArray[np.float64]:
            acquires.append(1)
            if failure_site != "final":
                round_hook(1, np.array([1.0]), cancel_flag)
            return np.array([2.0])

        def acquire_decimated(
            self,
            soc: object,
            *,
            progress: bool,
            round_hook: Callable[
                [int, NDArray[np.float64], CancelFlagProtocol], object
            ],
            cancel_flag: CancelFlagProtocol,
            **kwargs: object,
        ) -> NDArray[np.float64]:
            return self.acquire(
                soc, progress=progress, round_hook=round_hook, cancel_flag=cancel_flag, **kwargs
            )

    def convert(raw: NDArray[np.float64]) -> NDArray[np.float64]:
        if failure_site in ("round", "final"):
            raise cause
        return raw

    def update(data: NDArray[np.complex128]) -> None:
        if failure_site == "update":
            raise cause

    signal = StopSignal()
    buffer = SignalBuffer((1,), on_update=update, update_interval=None)
    schedule = Schedule(make_flow_cfg(rounds=1), buffer, stop=signal)
    with pytest.raises(RuntimeError, match=f"broken {failure_site}") as caught:
        with schedule:
            builder = schedule.prog_builder("soc", "soccfg", program_cls=Program).add(
                FakeModule("readout")
            )
            acquire = (
                builder.build_and_acquire_decimated
                if decimated
                else builder.build_and_acquire
            )
            acquire(raw2signal_fn=convert, options=AcquireOptions(retry=2))
    assert caught.value is cause
    assert schedule.outcome.status == "completed"
    assert signal.error is None
    assert builds == [1]
    assert acquires == ([] if failure_site == "build" else [1])
