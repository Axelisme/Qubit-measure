from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime.schedule import Schedule
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
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

    def run(self, config: TwotoneCfg, *, context: RunContext) -> TwoToneResult:
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
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

        viewer = context.plots.liveplot_2d(
            "measurement", "Flux Pulse Gain (a.u.)", "Frequency (MHz)"
        )
        signals_buffer = SignalBuffer(
            (len(gains), len(freqs)),
            on_update=lambda data: viewer.update(
                gains, freqs, twotone_signal2real(data)
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
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
                .build_and_acquire()
            )

        return TwoToneResult(gains, freqs, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[TwotoneCfg, TwoToneResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        del options  # This analysis has no configurable options.
        result = source.result

        gains, freqs, signals2D = result.gains, result.freqs, result.signals

        real_signals = twotone_signal2real(signals2D)

        fig, ax = plots.subplots("fit")

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
