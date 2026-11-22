from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from zcu_tools.analysis.fitting import fit_decay
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
class T1Result:
    gains: NDArray[np.float64]
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]


def t1_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return the real-valued measurement view of complex IQ without changing the input."""
    return rotate2real(signals).real


class T1Cfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Choose the reset module before each shot, or leave it disabled.",
        ),
    ] = None
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="π pulse",
            description="Enter a calibrated qubit π pulse to excite the qubit before the flux pulse.",
        ),
    ]
    flux_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Flux pulse",
            description="Enter a flux pulse with a sweepable waveform; the two sweeps set its gain and duration.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Enter the calibrated resonator readout used after the flux pulse.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Flux pulse gain",
            unit="a.u.",
            description="Enter the signed DAC gain scan of the flux pulse.",
        ),
    ]
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="Flux pulse duration",
            unit="µs",
            description="Enter the nonzero flux pulse durations over which the qubit relaxes.",
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


class T1Exp(Experiment[T1Result, T1Cfg]):
    # inner lengths stores memory us on disk (disk seconds) -> scale=1e-6;
    # outer gains -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[T1Result, T1Cfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("lengths", label="Time", unit="s", scale=1e-6),
            AxesSpec.Axis("gains", label="Flux Pulse Gain", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=T1Result,
        cfg_type=T1Cfg,
        cfg_schema_version="3.0",
        tag="fastflux/t1",
    )

    def run(self, config: T1Cfg, *, context: RunContext) -> T1Result:
        """Measure relaxation during a flux pulse at each programmed flux gain.

        Reset, excite with π, apply the gain/length-swept flux pulse, then read
        out. Return flux gains in a.u., pulse lengths in us and mean complex IQ
        shaped (gain, length). Each row is a decay trace; T1 is fitted later,
        not a directly measured population or a conversion from gain to flux.
        """
        # 1. Resolve flux gain and duration coordinates on the hardware grid.
        cfg = config
        soc, soccfg = context.soc, context.soccfg

        gain_sweep = cfg.gain
        length_sweep = cfg.length

        lf_ch = cfg.flux_pulse.ch
        gains = sweep2array(gain_sweep, "gain", {"soccfg": soccfg, "gen_ch": lf_ch})
        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg, "gen_ch": lf_ch})

        # 2. Keep the relaxation traces grouped by flux gain.
        fig, ax = context.plots.subplots(num="measurement", layout="constrained")
        fig.suptitle("Fast-flux relaxation · Qubit · Decay map")
        ax.set(xlabel="Flux pulse gain (a.u.)", ylabel="Flux pulse duration (µs)")
        ax.set_title(f"0/{gains.size * lengths.size} scan points with IQ")
        ax.grid(False)
        viewer = live_heatmap(ax)
        (image,) = viewer.set(
            gains, lengths, np.full((gains.size, lengths.size), np.nan)
        )
        fig.colorbar(image, ax=ax, label="Mean IQ real projection (a.u.)")

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Display available IQ on the scan grid without changing the buffer."""
            count = np.count_nonzero(np.isfinite(data))
            title = ax.set_title(f"{count}/{data.size} scan points with IQ")
            return (*viewer.set(gains, lengths, t1_signal2real(data)), title)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(gains), len(lengths)),
            on_update=update,
        )

        with Schedule(signals_buffer, scope=context.scope) as sched:
            # 3. Excite with pi, then vary the flux pulse gain and duration before readout.
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain = builder.add_sweep("gain", cfg.gain)
            length = builder.add_sweep("length", cfg.length)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.emit(Pulse("flux_pulse", cfg.flux_pulse, gain=gain, length=length))
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire(prog, soc, rounds=cfg.rounds)

        return T1Result(gains, lengths, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[T1Cfg, T1Result],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Fit decay at each flux gain and plot T1 with fit errors in us.

        First estimate a common baseline from accepted preliminary fits, then
        refit each row with that baseline fixed. Rejected final fits retain NaN
        T1 values; their initialized zero errors do not indicate a valid fit.
        """
        # 1. Project each flux-gain row to a real relaxation trace.
        del options  # This analysis has no configurable options.
        result = source.result

        gains, lengths, signals2D = result.gains, result.lengths, result.signals

        real_signals = t1_signal2real(signals2D)

        def fit_contrast(
            fit_signal: NDArray[np.float64], real_signal: NDArray[np.float64]
        ) -> tuple[float, float, float]:
            """Return fitted IQ span and its two existing acceptance bounds."""
            smooth_residual = gaussian_filter1d(real_signal - fit_signal, sigma=1)
            return (
                np.ptp(fit_signal).item(),
                0.5 * np.ptp(real_signal).item(),
                np.mean(np.abs(smooth_residual)).item(),
            )

        # 2. Share a median baseline from fits passing contrast/residual checks.
        prev_pOpt = None
        list_pOpt = []
        for real_signal in real_signals:
            fit = fit_decay(
                lengths,
                real_signal,
                fit_params=prev_pOpt,
            )
            span, half_measured_span, mean_smoothed_error = fit_contrast(
                fit.curve, real_signal
            )
            prev_pOpt = (
                fit.params
                if span > half_measured_span and span > mean_smoothed_error
                else None
            )
            list_pOpt.append(prev_pOpt)
        mean_y0 = np.median([pOpt[0] for pOpt in list_pOpt if pOpt is not None]).item()

        # 3. Refit at fixed baseline; rejected T1 values stay NaN with zero errors.
        t1s = np.full_like(gains, np.nan)
        t1errs = np.zeros_like(gains)
        r2s = np.full_like(gains, np.nan)
        residuals = np.full_like(gains, np.nan)
        rejected_gains: list[float] = []
        rejected_contrasts: list[tuple[float, float, float]] = []
        for i, real_signal in enumerate(real_signals):
            fit = fit_decay(
                lengths,
                real_signal,
                fit_params=list_pOpt[i],
                fixedparams=(mean_y0, None, None),
            )
            if fit.quality.r2 is not None:
                r2s[i] = fit.quality.r2
            if fit.quality.normalized_residual_rms is not None:
                residuals[i] = fit.quality.normalized_residual_rms
            span, half_measured_span, mean_smoothed_error = fit_contrast(
                fit.curve, real_signal
            )
            if span > half_measured_span and span > mean_smoothed_error:
                t1s[i] = fit.decay_time
                t1errs[i] = fit.decay_time_err
            else:
                rejected_gains.append(gains[i].item())
                rejected_contrasts.append(
                    (span, half_measured_span, mean_smoothed_error)
                )

        # 4. Show the measured map before the fitted lifetimes and diagnostics.
        fig, ax = plots.subplots(num="fit", layout="constrained")
        fig.suptitle("Fast-flux relaxation · Qubit · Decay map")
        ax.set(xlabel="Flux pulse gain (a.u.)", ylabel="Flux pulse duration (µs)")
        ax.grid(False)
        (image,) = live_heatmap(ax).set(gains, lengths, real_signals)
        fig.colorbar(image, ax=ax, label="Mean IQ real projection (a.u.)")

        fig, _ = plots.subplots(
            num="lifetime", nrows=3, sharex=True, layout="constrained"
        )
        fig.suptitle("Fast-flux relaxation · Qubit · Fitted T1")
        ax_t1, ax_r2, ax_residual = fig.axes
        ax_t1.errorbar(gains, t1s, yerr=t1errs, fmt="k.-", label="T1 ± fit SE")
        ax_t1.set(
            ylabel="T1 (µs)",
            title=f"{len(rejected_gains)}/{gains.size} fits rejected by contrast checks",
        )
        ax_t1.legend(loc="lower right")
        ax_r2.plot(gains, r2s, "k.-")
        ax_r2.set_ylabel("Fit R²")
        ax_residual.plot(gains, residuals, "k.-")
        ax_residual.set(
            xlabel="Flux pulse gain (a.u.)", ylabel="Normalized\nresidual RMS"
        )

        _plot_rejected_decays(plots, rejected_gains, rejected_contrasts)


def _plot_rejected_decays(
    plots: Plots,
    rejected_gains: list[float],
    rejected_contrasts: list[tuple[float, float, float]],
) -> None:
    """Plot failed fits' IQ spans and the measured bounds used to reject them."""
    if not rejected_gains:
        return

    fig, ax = plots.subplots(num="rejected_decay_fits", layout="constrained")
    fig.suptitle("Fast-flux relaxation · Qubit · Rejected fits")
    spans, half_spans, residual_bounds = zip(*rejected_contrasts, strict=True)
    ax.plot(rejected_gains, spans, "k.-", label="Fitted span")
    ax.plot(rejected_gains, half_spans, "C0x-", label="Half of measured span")
    ax.plot(rejected_gains, residual_bounds, "C1+-", label="Mean |smoothed residual|")
    ax.set(
        xlabel="Flux pulse gain (a.u.)",
        ylabel="IQ contrast (a.u.)",
        title="Fitted span must exceed both bounds",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncols=2)
