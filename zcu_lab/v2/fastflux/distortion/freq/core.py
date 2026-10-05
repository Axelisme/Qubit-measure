from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fitting import fitlor
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    US_TO_S,
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
    SoftDelay,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class FreqResult:
    lengths: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def freq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


def get_resonance_freq(
    xs: NDArray[np.float64], freqs: NDArray[np.float64], amps: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s_xs = []
    s_freqs = []

    for x, amp in zip(xs, amps, strict=False):
        if np.any(np.isnan(amp)):
            continue

        param, _ = fitlor(freqs, amp)
        curr_freq = param[3]

        s_xs.append(x)
        s_freqs.append(curr_freq)

    return np.array(s_xs), np.array(s_freqs)


class FreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    flux_pulse: PulseCfg
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class FreqSweepCfg(ConfigBase):
    length: SweepCfg
    freq: SweepCfg


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    readout_t: float
    sweep: FreqSweepCfg


@dataclass(frozen=True)
class FreqAnalysis:
    lengths_us: NDArray[np.float64]
    freqs_mhz: NDArray[np.float64]


class FreqExp(PersistableExperiment[FreqResult, FreqCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=MHZ_TO_HZ;
    # outer lengths stores us on disk (disk s) -> scale=US_TO_S
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("lengths", "Wait Time", "s", scale=US_TO_S),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        tag="fastflux/distortion/freq",
    )

    def run(self, config: FreqCfg, *, context: RunContext) -> FreqResult:
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
        modules = cfg.modules

        length_sweep = cfg.sweep.length
        freq_sweep = cfg.sweep.freq

        qub_pulse = modules.qub_pulse

        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": qub_pulse.ch}
        )

        viewer = context.plots.liveplot_2d(
            "measurement", "Time (us)", "Frequency (MHz)"
        )
        signals_buffer = SignalBuffer(
            (len(lengths), len(freqs)),
            on_update=lambda data: viewer.update(
                lengths, freqs, freq_signal2real(data)
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules
            length_param = sweep2param("length", sched.cfg.sweep.length)
            modules.qub_pulse.set_param(
                "freq", sweep2param("freq", sched.cfg.sweep.freq)
            )
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Join(
                        Pulse("flux_pulse", modules.flux_pulse),
                        [
                            SoftDelay("wait_time", delay=length_param),
                            Pulse("qub_pulse", modules.qub_pulse),
                        ],
                        SoftDelay("readout_t", sched.cfg.readout_t),
                    ),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("length", sched.cfg.sweep.length)
                .declare_sweep("freq", sched.cfg.sweep.freq)
                .build_and_acquire()
            )

        return FreqResult(lengths, freqs, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: None,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        del options  # This analysis has no configurable options.
        result = source.result

        cfg = source.cfg
        if cfg is None:
            raise ValueError("Analysis requires source.cfg")
        modules = cfg.modules

        flux_pulse = modules.flux_pulse
        qub_len = float(modules.qub_pulse.waveform.length)

        lengths, freqs, signals = result.lengths, result.freqs, result.signals

        # align to center of qubit pulse
        lengths = lengths + qub_len / 2

        amps = freq_signal2real(signals)
        s_lengths, s_freqs = get_resonance_freq(lengths, freqs, amps)

        sort_idxs = np.argsort(np.abs(s_freqs - s_freqs[0]))
        mean_background = np.mean(s_freqs[sort_idxs[: int(len(s_freqs) * 0.1)]])
        mean_topdetune = np.mean(s_freqs[sort_idxs[int(len(s_freqs) * 0.9) :]])

        start_t = float(flux_pulse.pre_delay)
        end_t = start_t + float(flux_pulse.waveform.length)

        ideal_lengths = np.linspace(lengths[0], lengths[-1], 1000)
        ideal_curve = np.full_like(ideal_lengths, mean_background)
        ideal_curve[(ideal_lengths >= start_t) & (ideal_lengths <= end_t)] = (
            mean_topdetune
        )

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        ax.imshow(
            amps.T,
            extent=(lengths[0], lengths[-1], freqs[0], freqs[-1]),
            aspect="auto",
            interpolation="none",
            cmap="RdBu_r",
            origin="lower",
        )

        # Plot the resonance frequencies and fitted curve
        ax.plot(s_lengths, s_freqs, ".", c="k")
        ax.plot(ideal_lengths, ideal_curve, "g-", label="Ideal")

        ax.axvspan(
            start_t - qub_len / 2, start_t + qub_len / 2, color="gray", alpha=0.3
        )
        ax.axvspan(end_t - qub_len / 2, end_t + qub_len / 2, color="gray", alpha=0.3)

        ax.set_ylabel("Qubit Frequency (MHz)", fontsize=14)
        ax.set_xlabel("Time (us)", fontsize=14)
        ax.legend(fontsize="x-large")

        fig.tight_layout()

        return FreqAnalysis(lengths_us=s_lengths, freqs_mhz=s_freqs)
