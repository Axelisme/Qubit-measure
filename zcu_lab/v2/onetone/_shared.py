"""Qt-free cfg and frequency grids for one-tone experiments."""

from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import PulseReadoutCfg, ResetCfg, SweepCfg


@dataclass(frozen=True)
class FreqResult:
    """Hardware-rounded absolute freqs in MHz and mean complex IQ, both shape (P,)."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class LinearFreqCfg(ConfigBase):
    """One-tone acquisition settings, averaging, delays and absolute frequency grid."""

    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        PulseReadoutCfg,
        FieldMeta(
            label="Readout",
            description="Resonator readout 設定與 timing（us）。",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Frequency",
            unit="MHz",
            description="Absolute MHz 掃描起訖值。",
        ),
    ]
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ]
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]


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
