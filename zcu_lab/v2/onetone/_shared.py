"""Qt-free results and frequency grids for one-tone experiments."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import PulseReadoutCfg, SweepCfg


@dataclass(frozen=True)
class FreqResult:
    """Hardware-rounded absolute freqs in MHz and mean complex IQ, both shape (P,)."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def linear_freqs(
    freq: SweepCfg,
    readout: PulseReadoutCfg,
    *,
    context: RunContext,
) -> NDArray[np.float64]:
    """Expand the absolute MHz sweep onto the readout's hardware frequency grid.

    freq supplies the MHz sweep; readout supplies generator/ADC channels.
    Neither input is changed.
    context supplies cached SoC configuration. Invalid sweeps propagate errors.
    """
    return sweep2array(
        freq,
        "freq",
        {"soccfg": context.soccfg, "gen_ch": readout.gen_ch, "ro_ch": readout.ro_ch},
    )
