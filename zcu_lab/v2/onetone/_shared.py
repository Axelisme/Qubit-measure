"""Qt-free cfg, frequency grids and spectrum buffers for one-tone experiments."""

from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseReadoutCfg,
    ResetCfg,
    SweepCfg,
)


@dataclass(frozen=True)
class FreqResult:
    """Hardware-rounded absolute freqs in MHz and mean complex IQ, both shape (P,)."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class FreqModuleCfg(ConfigBase):
    """Optional reset followed by the pulse readout's generator and ADC settings."""

    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        PulseReadoutCfg,
        FieldMeta(
            label="Readout",
            description="填入 resonator readout 設定。先用 lookback 取得 TOF，再以 onetone 取得 kappa，確認 trigger offset 與 waveform duration，單位 us。",
        ),
    ]


class FreqSweepCfg(ConfigBase):
    """Absolute MHz sweep; the hardware owns both generator and ADC frequencies."""

    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Frequency",
            unit="MHz",
            description="填入 absolute MHz 起訖值。range_find 不會從已知中心自動設定掃描窗。",
        ),
    ]


class LinearFreqCfg(ProgramV2Cfg, ExpCfgModel):
    """One-tone acquisition settings, averaging, delays and absolute frequency grid."""

    modules: FreqModuleCfg
    sweep: FreqSweepCfg


def linear_freqs(config: LinearFreqCfg, *, context: RunContext) -> NDArray[np.float64]:
    """Expand the absolute MHz sweep onto the readout's hardware frequency grid.

    config supplies the sweep and generator/ADC channels without being changed.
    context supplies cached SoC configuration. Invalid sweeps propagate errors.
    """
    readout = config.modules.readout
    return sweep2array(
        config.sweep.freq,
        "freq",
        {"soccfg": context.soccfg, "gen_ch": readout.gen_ch, "ro_ch": readout.ro_ch},
    )


def spectrum_buffer(freqs: NDArray[np.float64], *, plots: Plots) -> SignalBuffer:
    """Create a complex NaN buffer and a live magnitude plot for (P,) MHz freqs.

    plots owns the measurement figure. Buffer updates draw the entire spectrum;
    acquisition and cancellation remain the caller's responsibility.
    """
    viewer = plots.liveplot_1d("measurement", "Frequency (MHz)", "Amplitude")
    return SignalBuffer(
        (len(freqs),), on_update=lambda data: viewer.update(freqs, np.abs(data))
    )
