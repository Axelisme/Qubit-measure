from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import fitlor
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Join,
    Pulse,
    Readout,
    Reset,
    SoftDelay,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2._support import live_heatmap


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


class FreqCfg(ConfigBase):
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
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
    ] = 1.0
    reset: Reset.Cfg | None = None
    flux_pulse: Pulse.Cfg
    qub_pulse: Pulse.Cfg
    readout: Readout.Cfg
    readout_t: float
    length: SweepCfg
    freq: SweepCfg


@dataclass(frozen=True)
class FreqAnalysis:
    lengths_us: NDArray[np.float64]
    freqs_mhz: NDArray[np.float64]


class FreqExp(Experiment[FreqResult, FreqCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=1e6;
    # outer lengths stores us on disk (disk s) -> scale=1e-6
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),
            AxesSpec.Axis("lengths", "Wait Time", "s", scale=1e-6),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        cfg_schema_version="3.0",
        tag="fastflux/distortion/freq",
    )

    def run(self, config: FreqCfg, *, context: RunContext) -> FreqResult:
        """Probe the qubit resonance at different times during a flux transient.

        After reset, run flux, wait→frequency-swept qubit probe, and readout_t
        in parallel. Read out after the longest branch: readout_t sets a minimum
        duration in us, not an exact read time. Return probe waits in us,
        frequencies in MHz and mean complex IQ shaped (wait, frequency).
        Resonance fits in analyze trace the flux-induced frequency response.
        """
        # Quantize probe timing and frequency on the configured hardware.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg

        length_sweep = cfg.length
        freq_sweep = cfg.freq

        qub_pulse = modules.qub_pulse

        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": qub_pulse.ch}
        )

        # Keep the delayed spectroscopy map in wait-major order.
        viewer = live_heatmap(
            context.plots, "measurement", "Time (us)", "Frequency (MHz)"
        )
        signals_buffer = SignalBuffer(
            (len(lengths), len(freqs)),
            on_update=lambda data: viewer.update(
                lengths, freqs, freq_signal2real(data)
            ),
        )

        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg
            length_param = sweep2param("length", sched.cfg.length)
            modules.qub_pulse.set_param("freq", sweep2param("freq", sched.cfg.freq))

            # Probe while flux is scheduled independently; read after all branches.
            _ = (
                sched.prog_builder(
                    soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
                )
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
                .declare_sweep("length", sched.cfg.length)
                .declare_sweep("freq", sched.cfg.freq)
                .build_and_acquire(rounds=cfg.rounds)
            )

        return FreqResult(lengths, freqs, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: None,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        """Fit each finite spectroscopy row to obtain resonance versus time.

        Return probe-center times in us and Lorentzian centers in MHz, skipping
        rows containing NaN. Require source.cfg for probe/flux timing, otherwise
        raise ValueError; the ideal curve is only a timing reference.
        """
        del options  # This analysis has no configurable options.
        result = source.result

        cfg = source.cfg
        if cfg is None:
            raise ValueError("Analysis requires source.cfg")
        modules = cfg

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
