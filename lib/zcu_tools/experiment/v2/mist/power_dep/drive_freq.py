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
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
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


@dataclass(frozen=True)
class DriveFreqResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def drivefreq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = signals - np.mean(signals[:, :avg_len])

    return np.abs(mist_signals)


class DriveFreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    probe_pulse: PulseCfg
    readout: ReadoutCfg


class DriveFreqSweepCfg(ConfigBase):
    freq: SweepCfg
    gain: SweepCfg


class DriveFreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: DriveFreqModuleCfg
    sweep: DriveFreqSweepCfg


class DriveFreqExp(PersistableExperiment[DriveFreqResult, DriveFreqCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=MHZ_TO_HZ; outer gains -> IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Pulse frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("gains", "Pulse gain", "a.u.", scale=IDENTITY),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=DriveFreqResult,
        cfg_type=DriveFreqCfg,
        tag="mist/",
    )

    def run(
        self,
        config: DriveFreqCfg,
        *,
        context: RunContext,
    ) -> DriveFreqResult:
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal.event,
        )
        modules = cfg.modules

        freq_sweep = cfg.sweep.freq
        gain_sweep = cfg.sweep.gain

        probe_pulse = modules.probe_pulse
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": probe_pulse.ch}
        )
        gains = sweep2array(
            gain_sweep, "gain", {"soccfg": soccfg, "gen_ch": probe_pulse.ch}
        )

        viewer = context.plots.liveplot_2d(
            "measurement", "Pulse frequency (MHz)", "Pulse gain (a.u.)"
        )
        signals_buffer = SignalBuffer(
            (len(freqs), len(gains)),
            on_update=lambda data: viewer.update(
                freqs, gains, drivefreq_signal2real(data)
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules
            modules.probe_pulse.set_param(
                "freq", sweep2param("freq", sched.cfg.sweep.freq)
            )
            modules.probe_pulse.set_param(
                "gain", sweep2param("gain", sched.cfg.sweep.gain)
            )
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Pulse("probe_pulse", modules.probe_pulse),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("freq", sched.cfg.sweep.freq)
                .declare_sweep("gain", sched.cfg.sweep.gain)
                .build_and_acquire()
            )

        return DriveFreqResult(gains=gains, freqs=freqs, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[DriveFreqCfg, DriveFreqResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        del options
        result = source.result

        freqs, gains, signals = result.freqs, result.gains, result.signals

        real_signals = drivefreq_signal2real(signals)

        _, ax = plots.subplots("fit", figsize=config.figsize)

        ax.imshow(
            real_signals.T,
            extent=(freqs[0], freqs[-1], gains[0], gains[-1]),
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax.set_xlabel("Pulse frequency (MHz)", fontsize=14)
        ax.set_ylabel("Pulse gain (a.u.)", fontsize=14)
