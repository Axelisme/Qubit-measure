from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import fitlor
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_heatmap


@dataclass(frozen=True)
class FreqResult:
    lengths: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


@dataclass(frozen=True)
class FreqAnalysis:
    lengths_us: NDArray[np.float64]
    freqs_mhz: NDArray[np.float64]


def freq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return the real-valued measurement view of complex IQ without changing the input."""
    return rotate2real(signals).real


class FreqCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Choose the reset module before each shot, or leave it disabled.",
        ),
    ] = None
    flux_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Flux pulse",
            description="Enter the flux transient to characterize; its pre-delay places the leading edge.",
        ),
    ]
    qub_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit probe",
            description="Enter the spectroscopy pulse; Qubit probe frequency supplies its swept carrier.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the calibrated resonator readout used after both drives finish.",
        ),
    ]
    readout_t: Annotated[
        float,
        FieldMeta(
            label="Minimum readout wait",
            unit="µs",
            description="Enter the minimum time before the final readout branch can start.",
        ),
    ]
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="Qubit probe wait",
            unit="µs",
            description="Enter the waiting time before the spectroscopy pulse starts.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Qubit probe frequency",
            unit="MHz",
            description="Enter the absolute RF carrier scan that spans the qubit resonance.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the number of shots per scan point in each round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of times to acquire and average the full scan.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the time for relaxation after each shot.",
        ),
    ]


class FreqExp(Experiment[FreqResult, FreqCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=1e6;
    # outer lengths stores us on disk (disk s) -> scale=1e-6
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),
            AxesSpec.Axis("lengths", label="Wait Time", unit="s", scale=1e-6),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
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
        # 1. Quantize probe timing and frequency on the configured hardware.
        cfg = config
        soc, soccfg = context.soc, context.soccfg

        length_sweep = cfg.length
        freq_sweep = cfg.freq

        qub_pulse = cfg.qub_pulse

        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": qub_pulse.ch}
        )

        # 2. Keep the delayed spectroscopy map in wait-major order.
        centers = lengths + float(qub_pulse.waveform.length) / 2
        fig, ax = context.plots.subplots(num="measurement", layout="constrained")
        fig.suptitle("Fast-flux transient · Qubit · Resonance")
        ax.set(xlabel="Probe-center time (µs)", ylabel="Qubit probe frequency (MHz)")
        ax.set_title(f"0/{centers.size * freqs.size} scan points with IQ")
        ax.grid(False)
        viewer = live_heatmap(ax)
        (image,) = viewer.set(
            centers, freqs, np.full((centers.size, freqs.size), np.nan)
        )
        fig.colorbar(image, ax=ax, label="Mean IQ real projection (a.u.)")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Display available IQ on the scan grid without changing the buffer."""
            count = np.count_nonzero(np.isfinite(data))
            title = ax.set_title(f"{count}/{data.size} scan points with IQ")
            return (*viewer.set(centers, freqs, freq_signal2real(data)), title)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(lengths), len(freqs)),
            on_update=update,
        )

        with Schedule(signals_buffer, scope=context.scope) as sched:
            # 3. Probe during the flux transient and read out after all timing branches.
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            length = builder.add_sweep("length", cfg.length)
            freq = builder.add_sweep("freq", cfg.freq)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.fork("default", into=["flux", "probe", "readout_wait"])
            builder.emit(Pulse("flux_pulse", cfg.flux_pulse), timeline="flux")
            builder.timeline_inc(length, timeline="probe")
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse, freq=freq), timeline="probe")
            builder.timeline_inc(cfg.readout_t, timeline="readout_wait")
            builder.join("flux", "probe", "readout_wait", into="default")
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(prog, soc, rounds=cfg.rounds)

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
        # 1. Require saved pulse timing to interpret the delayed spectroscopy map.
        del options  # This analysis has no configurable options.
        result = source.result

        cfg = source.cfg
        if cfg is None:
            raise ValueError("Analysis requires source.cfg")

        flux_pulse = cfg.flux_pulse
        qub_len = float(cfg.qub_pulse.waveform.length)

        lengths, freqs, signals = result.lengths, result.freqs, result.signals

        # 2. Locate each spectroscopy sample at the qubit-probe center.
        lengths = lengths + qub_len / 2

        amps = freq_signal2real(signals)
        # 3. Recover the resonance from each complete delay trace.
        fit_lengths = []
        fit_freqs = []
        fit_errors = []

        for x, amp in zip(lengths, amps, strict=False):
            if np.any(np.isnan(amp)):
                continue

            param, covariance = fitlor(freqs, amp)
            fit_lengths.append(x)
            fit_freqs.append(param[3])
            variance = covariance[3, 3]
            fit_errors.append(
                np.sqrt(variance) if np.isfinite(variance) and variance >= 0 else np.nan
            )

        s_lengths, s_freqs = np.array(fit_lengths), np.array(fit_freqs)

        # 4. Use low/high detuning bands to construct a flux-timing reference.
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

        # 5. Compare resonance tracks with ideal timing and finite-probe edge windows.
        fig, ax = plots.subplots(num="fit", layout="constrained")
        fig.suptitle("Fast-flux transient · Qubit · Resonance")
        ax.set(xlabel="Probe-center time (µs)", ylabel="Qubit probe frequency (MHz)")
        ax.grid(False)
        (image,) = live_heatmap(ax).set(lengths, freqs, amps)
        fig.colorbar(image, ax=ax, label="Mean IQ real projection (a.u.)")
        scan_ylim = ax.get_ylim()

        ax.errorbar(
            s_lengths,
            s_freqs,
            yerr=fit_errors,
            fmt="k.",
            label="Lorentzian center ± fit SE",
        )
        ax.plot(
            ideal_lengths, ideal_curve, color="green", label="Rectangular reference"
        )
        ax.axvspan(
            start_t - qub_len / 2,
            start_t + qub_len / 2,
            color="gray",
            alpha=0.25,
            label="Probe overlaps edge",
        )
        ax.axvspan(end_t - qub_len / 2, end_t + qub_len / 2, color="gray", alpha=0.25)
        ax.set_ylim(scan_ylim)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncols=2)

        return FreqAnalysis(lengths_us=s_lengths, freqs_mhz=s_freqs)
