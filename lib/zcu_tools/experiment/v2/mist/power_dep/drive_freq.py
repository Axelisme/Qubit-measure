from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
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
    config,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
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
    cfg_snapshot: DriveFreqCfg | None = None


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

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: DriveFreqCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> DriveFreqResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
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

        with LivePlot2D("Pulse frequency (MHz)", "Pulse gain (a.u.)") as viewer:
            signals_buffer = SignalBuffer(
                (len(freqs), len(gains)),
                on_update=lambda data: viewer.update(
                    freqs, gains, drivefreq_signal2real(data)
                ),
            )
            with Schedule(cfg, signals_buffer) as sched:
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
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )
                signals = signals_buffer.array

        return DriveFreqResult(
            gains=gains, freqs=freqs, signals=signals, cfg_snapshot=orig_cfg
        )

    @retrieve_result
    def analyze(self, result: DriveFreqResult | None = None) -> Figure:
        assert result is not None, "no result found"

        freqs, gains, signals = result.freqs, result.gains, result.signals

        real_signals = drivefreq_signal2real(signals)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(ax, Axes)

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

        return fig
