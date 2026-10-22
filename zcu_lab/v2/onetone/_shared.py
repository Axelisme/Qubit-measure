"""Qt-free configuration and linear hardware acquisition for one-tone spectra."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)


@dataclass(frozen=True)
class FreqResult:
    """Hardware-rounded absolute freqs in MHz and mean complex IQ, both shape (P,)."""

    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class FreqModuleCfg(ConfigBase):
    """Optional reset followed by the pulse readout's generator and ADC settings."""

    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[PulseReadoutCfg, FieldMeta(label="Readout")]


class FreqSweepCfg(ConfigBase):
    """Absolute MHz sweep; the hardware owns both generator and ADC frequencies."""

    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]


class LinearFreqCfg(ProgramV2Cfg, ExpCfgModel):
    """One-tone acquisition settings, averaging, delays and absolute frequency grid."""

    modules: FreqModuleCfg
    sweep: FreqSweepCfg


def run_linear(config: LinearFreqCfg, *, context: RunContext) -> FreqResult:
    """Reset and read out at each FPGA-swept absolute MHz frequency.

    config supplies pulse readout, sweep and averaging; its values stay unchanged.
    context supplies SoC, cached config, plots and cancellation. Return rounded
    MHz and mean complex IQ. Cancellation retains unacquired NaNs in the buffer;
    acquisition errors propagate to the host.
    """
    cfg = deepcopy(config)
    readout = cfg.modules.readout
    freqs = sweep2array(
        cfg.sweep.freq,
        "freq",
        {"soccfg": context.soccfg, "gen_ch": readout.gen_ch, "ro_ch": readout.ro_ch},
    )
    viewer = context.plots.liveplot_1d("measurement", "Frequency (MHz)", "Amplitude")
    signals = SignalBuffer(
        (len(freqs),), on_update=lambda data: viewer.update(freqs, np.abs(data))
    )
    with Schedule(cfg, signals, stop=context.cancel_signal) as sched:
        cfg = sched.cfg
        modules = cfg.modules
        freq_sweep = cfg.sweep.freq
        modules.readout.set_param("freq", sweep2param("freq", freq_sweep))
        _ = (
            sched.prog_builder(context.soc, context.soccfg)
            .add_reset("reset", modules.reset)
            .add(PulseReadout("readout", modules.readout))
            .declare_sweep("freq", freq_sweep)
            .build_and_acquire()
        )
    return FreqResult(freqs=freqs, signals=signals.array)
