from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.fitting import (
    FitQuality,
    decaycos,
    expfunc,
    fit_decay,
    fit_decay_fringe,
)
from zcu_tools.analysis.shot_uncertainty import iq_shot_sem
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    config,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    Delay,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
    split_sums,
)
from zcu_tools.utils.process import find_rotate_angle

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.detune import actual_detune


@dataclass(frozen=True)
class T2EchoResult:
    """Total delays in us and mean complex IQ, one sample per delay.

    true_activate_detune is the phase-ramp frequency in MHz after hardware
    time/phase rounding, not a fitted qubit detuning. It is run-only metadata
    and is None after canonical load. iq_moments holds per-delay retained
    shot sums [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²], or None for mean-only files.
    """

    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    true_activate_detune: float | None = None
    iq_moments: NDArray[np.float64] | None = None


@validated_dataclass(frozen=True)
class T2EchoAnalyzeOptions:
    fit_method: Annotated[Literal["fringe", "decay"], FieldMeta(label="Fit method")] = (
        "decay"
    )
    fit_phase: Annotated[bool, FieldMeta(label="Fit phase offset (fringe only)")] = (
        False
    )


@dataclass(frozen=True)
class T2EchoAnalysis:
    t2e: float
    t2e_err: float
    detune: float
    detune_err: float
    fit_quality: Mapping[str, FitQuality] | None = None
    review_note: str | None = None
    """Reasons the T₂ writeback is not preselected; None if it passes existing limits."""


def _coherence_review_note(
    t2: float, t2_err: float, quality: FitQuality | None
) -> str | None:
    """Explain why an Echo fit is not preselected for writeback.

    Times/errors are in us. Initial selection requires positive finite T2,
    nonnegative finite error/T2 <=0.5, r2 >=0.9 and normalized residual RMS <=0.1.
    Missing/nonfinite required metrics or their invalid issues require review.
    Unrelated fixed-zero parameter issues do not reject the result. None means
    these review gates passed, not that a calibration is physically established.
    """
    reasons = _coherence_fit_review_reasons(quality)
    if not math.isfinite(t2) or t2 <= 0:
        reasons.append("T2 must be finite and positive")
    elif not math.isfinite(t2_err) or t2_err < 0:
        reasons.append("T2 error must be finite and nonnegative")
    elif t2_err / t2 > 0.5:
        reasons.append(f"T2 relative error={t2_err / t2:.4g} >0.5")
    return "; ".join(reasons) if reasons else None


def _coherence_fit_review_reasons(quality: FitQuality | None) -> list[str]:
    if quality is None:
        return ["fit quality unavailable"]
    reasons = [
        f"{issue.path}: {issue.reason}"
        for issue in quality.invalid
        if issue.path
        in {"r2", "normalized_residual_rms", "relative_parameter_errors.decay_time"}
    ]
    if quality.r2 is None or not math.isfinite(quality.r2):
        reasons.append("r² unavailable")
    elif quality.r2 < 0.9:
        reasons.append(f"r²={quality.r2:.4g} <0.9")
    rms = quality.normalized_residual_rms
    if rms is None or not math.isfinite(rms):
        reasons.append("normalized RMS unavailable")
    elif rms > 0.1:
        reasons.append(f"normalized RMS={rms:.4g} >0.1")
    return reasons


def _direction(signals: NDArray[np.complex128]) -> tuple[float, float]:
    """Project along the principal IQ axis, or use I if fewer than two points."""
    try:
        angle = find_rotate_angle(signals)
    except ValueError:
        return (1.0, 0.0)
    return (float(np.cos(angle)), float(np.sin(angle)))


def t2echo_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    i, q = _direction(signals)
    return signals.real * i + signals.imag * q


class T2EchoCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi2_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi/2 Pulse")]
    pi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="µs")]
    detune: float = 0.0
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Stop averaging after a completed round reaches this SNR; None disables.",
        ),
    ] = Field(default=None, gt=0)
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]


_TRACE_AXIS = VariableAxisSpec(
    label="Time",
    unit="s",
    field_name="times",
    scale=1e-6,
)
_MOMENT_COLUMNS = tuple(range(6))


def _build_result(data: GroupedLoadData[T2EchoCfg]) -> T2EchoResult:
    """Restore the measured trace and any shot moments on the same native axis."""
    trace = data.variable("data")
    (coordinates,) = trace.axes
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return T2EchoResult(coordinates, trace.z)
    columns, moment_coordinates = moments.axes
    if not np.array_equal(coordinates, moment_coordinates) or not np.array_equal(
        columns, _MOMENT_COLUMNS
    ):
        raise ValueError(
            "/vars/iq_moments: expected columns 0–5 on the /vars/data grid"
        )
    return T2EchoResult(coordinates, trace.z, iq_moments=moments.z)


class T2EchoExp(Experiment[T2EchoResult, T2EchoCfg]):
    Options: ClassVar[type[T2EchoAnalyzeOptions]] = T2EchoAnalyzeOptions

    # times stores us in memory, s on disk -> scale=1e-6
    # ADR-0069: preserve the original trace; shot moments are optional on load.
    AXES_SPEC: ClassVar[GroupedAxesSpec[T2EchoResult, T2EchoCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "data",
                (_TRACE_AXIS,),
                VariableZSpec(
                    "signals", label="Signal", unit="a.u.", dtype=np.complex128
                ),
            ),
            VariableSpec(
                "iq_moments",
                (
                    VariableAxisSpec(label="Moment", unit="", values=_MOMENT_COLUMNS),
                    _TRACE_AXIS,
                ),
                VariableZSpec("iq_moments", label="IQ moments", unit=""),
                required=False,
            ),
        ),
        result_builder=_build_result,
        labber_variables=("data",),
        result_type=T2EchoResult,
        cfg_type=T2EchoCfg,
        cfg_schema_version="4.0",
        tag="twotone/ge/t2echo",
    )

    def run(self, cfg: T2EchoCfg, *, context: RunContext) -> T2EchoResult:
        """Measure refocused coherence with a Hahn-echo delay sweep.

        After optional reset, apply pi/2, half the total delay, pi, the other
        half delay, then pi/2 and readout. cfg supplies the pulses and total
        delay sweep in us. detune in MHz ramps the final pi/2 phase with total
        delay to add fringes; it does not change the drive frequency.
        Return hardware-rounded total delays and mean complex IQ. analyze
        extracts T2echo from the decay or fringe envelope.
        """
        # 1. Round both half-delays and report the realized phase-ramp frequency.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        detune = cfg.detune

        lengths = sweep2array(cfg.delay, "time", {"soccfg": soccfg, "scaler": 0.5})

        true_detune = actual_detune(
            cfg.delay,
            detune=detune,
            gen_ch=cfg.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=0.5,
        )

        # 2. Display the IQ projection while averaging the total-delay sweep.
        fig, ax = context.plots.subplots(num="measurement")
        experiment = "Hahn echo"
        ax.set(
            title=experiment,
            xlabel="Total free-evolution delay (µs)",
            ylabel="IQ projection (a.u.)",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        viewer = live_errorbar(ax)

        @context.plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Update the projected IQ, shot SEM and acquisition count."""
            counts, mean = split_sums(moments)
            i, q = _direction(mean)
            sem = iq_shot_sem(moments, direction=(i, q))
            retained = np.nanmax(counts, initial=0)
            title = ax.set_title(f"{experiment} · {retained:.0f} retained shots/point")
            return (*viewer.set(lengths, mean.real * i + mean.imag * q, sem), title)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(lengths), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
        )
        # 3. Measure midpoint-refocused coherence with a phase-ramped final projection.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            length_sweep = cfg.delay
            length_param = builder.add_sweep("delay", length_sweep)
            detune_param = 360 * detune * length_param

            if cfg.reset is not None:
                builder.emit(cfg.reset.build("reset"))
            builder.emit(Pulse("pi2_pulse1", cfg.pi2_pulse))
            builder.emit(Delay("t2e_delay1", delay=0.5 * length_param))
            builder.delay_by_timeline()
            builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.emit(Delay("t2e_delay2", delay=0.5 * length_param))
            builder.delay_by_timeline()
            builder.emit(
                Pulse(
                    name="pi2_pulse2",
                    cfg=cfg.pi2_pulse.with_updates(
                        phase=cfg.pi2_pulse.phase + detune_param
                    ),
                )
            )
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                soc,
                reducer=shot_sums(prog, (IQ("readout", offset_corrected=True),)),
                rounds=cfg.rounds,
                options=RunOptions(
                    stop_condition=snr_checker(
                        signals_buffer.at(),
                        cfg.earlystop_snr,
                        lambda sums: t2echo_signal2real(split_sums(sums)[1]),
                    ),
                ),
            )

        # 4. Preserve the realized phase-ramp frequency with the measured echo trace.
        return T2EchoResult(
            times=lengths,
            signals=split_sums(signals_buffer.array)[1],
            iq_moments=signals_buffer.array,
            true_activate_detune=true_detune,
        )

    def analyze(
        self,
        source: RunRecord[T2EchoCfg, T2EchoResult],
        options: T2EchoAnalyzeOptions,
        *,
        plots: Plots,
    ) -> T2EchoAnalysis:
        """Fit T2echo in us from the IQ projection, excluding the first sample.

        fit_method selects a pure decay or a decaying fringe. fit_phase frees
        the fringe phase; otherwise it is fixed at zero. Decay-only fits ignore
        fit_phase and return zero detune. Fringe detune is the observed MHz
        frequency, including the applied phase ramp, not a corrected qubit
        detuning. Return fit uncertainties and quality, and draw the fit.
        Zero source detune forces decay. For fringe eligibility use cfg.detune
        when present, otherwise finite result.true_activate_detune. Missing
        detune for a requested fringe fit raises ValueError; explicit decay needs
        no detune source. Options validate fit_method as fringe or decay.
        """
        # 1. Resolve the fringe-detuning source, falling back to a decay fit at zero detuning.
        result = source.result
        fit_method = options.fit_method
        if fit_method == "fringe":
            detune_source = (
                source.cfg.detune
                if source.cfg is not None
                else result.true_activate_detune
            )
            if detune_source is None or not np.isfinite(detune_source):
                raise ValueError(
                    "Echo fringe fit requires source detune: cfg or saved finite actual_detune"
                )
            if detune_source == 0.0:
                fit_method = "decay"

        xs, signals = result.times, result.signals

        xs = xs[1:]
        signals = signals[1:]

        # 2. Estimate echo coherence from the real-IQ trace after omitting its first point.
        i, q = _direction(signals)
        real_signals = signals.real * i + signals.imag * q
        dense = np.linspace(xs.min(), xs.max(), max(200, xs.size))

        if fit_method == "fringe":
            fixedparams = None if options.fit_phase else [None, None, None, 0.0, None]
            fit = fit_decay_fringe(xs, real_signals, fixedparams=fixedparams)
            t2e, t2eerr, detune, detune_err = (
                fit.decay_time,
                fit.decay_time_err,
                fit.freq,
                fit.freq_err,
            )
            y_fit = decaycos(dense, *fit.params)
        else:
            fit = fit_decay(xs, real_signals)
            t2e, t2eerr = fit.decay_time, fit.decay_time_err
            y_fit = expfunc(dense, *fit.params)
            detune = 0.0
            detune_err = 0.0

        # 3. Assess echo decay and any fitted fringe frequency against the measured trace.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        if result.iq_moments is None:
            ax.plot(xs, real_signals, ".", label="Measured")
        else:
            sem = iq_shot_sem(result.iq_moments[1:], direction=(i, q))
            ax.errorbar(
                xs, real_signals, yerr=sem, fmt=".", label="Measured ± shot SEM"
            )
        label = "Fringe fit" if fit_method == "fringe" else "Decay fit"
        quality = fit.quality
        review_note = _coherence_review_note(t2e, t2eerr, quality)
        if quality.r2 is not None:
            label += f" (R² = {quality.r2:.3f})"
        ax.plot(dense, y_fit, label=label)
        title = f"Hahn echo: T₂ = {t2e:.2f} ± {t2eerr:.2f} µs"
        if fit_method == "fringe":
            title += f"\nFringe frequency = {detune:.4f} ± {detune_err:.4f} MHz"
        if quality.normalized_residual_rms is not None:
            title += (
                f"\nNormalized residual RMS = {quality.normalized_residual_rms:.3f}"
            )
        ax.set(
            title=title,
            xlabel="Total free-evolution delay (µs)",
            ylabel="IQ projection (a.u.)",
        )
        ax.legend()
        if review_note is not None:
            fig.text(
                0.5,
                0.02,
                "Not preselected for writeback:\n" + review_note.replace("; ", "\n"),
                ha="center",
                va="bottom",
                fontsize=8,
            )
            fig.tight_layout(
                rect=(0, min(0.31, 0.12 + 0.04 * review_note.count(";")), 1, 1)
            )
        else:
            fig.tight_layout()

        return T2EchoAnalysis(
            t2e=float(t2e),
            t2e_err=float(t2eerr),
            detune=float(detune),
            detune_err=float(detune_err),
            fit_quality={"fit": quality},
            review_note=review_note,
        )
