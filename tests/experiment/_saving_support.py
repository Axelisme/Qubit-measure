"""Typed fake acquisition and native figures shared by save contract tests."""

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    ExpCfgModel,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.context import RunContext


class SavingCfg(ExpCfgModel):
    """Synthetic gain in volts per unit signal."""

    gain: float = 1.0


@dataclass
class SavingResult:
    """Frequency samples in MHz and the corresponding complex volts."""

    frequency: NDArray[np.float64]
    signal: NDArray[np.complex128]


class SavingCore(PersistableExperiment[SavingResult, SavingCfg]):
    """Acquire a typed three-point signal without consulting hardware."""

    AXES_SPEC = AxesSpec(
        axes=(Axis("frequency", "frequency", "Hz", scale=1e6),),
        z=ZSpec("signal", "signal", "V"),
        result_type=SavingResult,
        cfg_type=SavingCfg,
        tag="test/saving",
    )

    def run(self, config: SavingCfg, *, context: RunContext | None) -> SavingResult:
        """Return deterministic data; None is allowed for this offline fake."""
        return SavingResult(
            frequency=np.array([1000.0, 1001.0, 1002.0]),
            signal=np.asarray(
                np.array([1.0 + 2.0j, 3.0 - 4.0j, 5.0 + 6.0j]) * config.gain,
                dtype=np.complex128,
            ),
        )


def saving_figure() -> Figure:
    """Build a native figure without a pyplot manager or GUI host."""
    figure = Figure()
    figure.subplots().plot([1.0, 2.0], [3.0, 4.0])
    return figure
