from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program.v2 import (
    Join,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class TwoToneResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: TwotoneCfg | None = None


def twotone_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class TwoToneModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    flux_pulse: PulseCfg
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class TwoToneSweepCfg(ConfigBase):
    gain: SweepCfg
    freq: SweepCfg


class TwotoneCfg(ProgramV2Cfg, ExpCfgModel):
    modules: TwoToneModuleCfg
    sweep: TwoToneSweepCfg


class TwoToneExp(PersistableExperiment[TwoToneResult, TwotoneCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=MHZ_TO_HZ; outer gains -> IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("gains", "Flux Pulse Gain", "a.u.", scale=IDENTITY),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=TwoToneResult,
        cfg_type=TwotoneCfg,
        tag="fastflux/twotone",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: TwotoneCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> TwoToneResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        # uniform in square space
        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.flux_pulse.ch},
        )
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        with LivePlot2D("Flux Pulse Gain (a.u.)", "Frequency (MHz)") as viewer:
            signals_buffer = SignalBuffer(
                (len(gains), len(freqs)),
                on_update=lambda data: viewer.update(
                    gains, freqs, twotone_signal2real(data)
                ),
            )
            with Schedule(cfg, signals_buffer) as sched:
                modules = sched.cfg.modules
                modules.flux_pulse.set_param(
                    "gain", sweep2param("gain", sched.cfg.sweep.gain)
                )
                modules.qub_pulse.set_param(
                    "freq", sweep2param("freq", sched.cfg.sweep.freq)
                )
                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", modules.reset),
                        Join(
                            Pulse("flux_pulse", modules.flux_pulse),
                            Pulse("qub_pulse", modules.qub_pulse),
                        ),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("gain", sched.cfg.sweep.gain)
                    .declare_sweep("freq", sched.cfg.sweep.freq)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )
                signals = signals_buffer.array

        return TwoToneResult(gains, freqs, signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(self, result: TwoToneResult | None = None) -> Figure:
        assert result is not None, "No result found"

        gains, freqs, signals2D = result.gains, result.freqs, result.signals

        real_signals = twotone_signal2real(signals2D)

        fig, ax = plt.subplots()

        ax.imshow(
            real_signals.T,
            extent=(
                float(gains[0]),
                float(gains[-1]),
                float(freqs[0]),
                float(freqs[-1]),
            ),
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax.set_xlabel("Flux Pulse Gain (a.u.)")
        ax.set_ylabel("Frequency (MHz)")

        fig.tight_layout()

        return fig
