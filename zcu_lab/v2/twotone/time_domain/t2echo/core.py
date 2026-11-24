from __future__ import annotations

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
from zcu_tools.experiment import Experiment, config
from zcu_tools.experiment.axes_spec import (
    GroupedAxesSpec,
    GroupedLoadData,
    RunValueSpec,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
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
from zcu_tools.utils.process import find_rotate_angle, rotate2real

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.support.detune import actual_detune

_DELAY_LABEL = "Total echo delay (µs)"
_IQ_LABEL = "IQ projection (a.u.)"


@dataclass(frozen=True)
class T2EchoResult:
    """Total delays in us and mean complex IQ, one sample per delay.

    true_activate_detune is the phase-ramp frequency in MHz after hardware
    time/phase rounding, not a fitted qubit detuning. It is None if the
    applied frequency was not recorded or could not be estimated.
    iq_moments contains corrected shot sums [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]
    with shape (len(times), 6) from the same
    run as signals. It is None when only mean IQ is available, including when
    loading Native data saved without shot sums.
    """

    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    true_activate_detune: float | None = None
    iq_moments: NDArray[np.float64] | None = None


@validated_dataclass(frozen=True)
class T2EchoAnalyzeOptions:
    fit_method: Annotated[
        Literal["fringe", "decay"],
        FieldMeta(
            label="Fit method",
            description="Choose decay for the echo envelope or fringe to fit its modulation when detuned.",
        ),
    ] = "decay"
    fit_phase: Annotated[
        bool,
        FieldMeta(
            label="Fit phase offset (fringe only)",
            description="Fit the fringe phase offset instead of fixing it at 0°.",
        ),
    ] = False


@dataclass(frozen=True)
class T2EchoAnalysis:
    t2e: float
    t2e_err: float
    detune: float
    detune_err: float
    fit_quality: Mapping[str, FitQuality] | None = None


def _iq_direction(signals: NDArray[np.complex128]) -> tuple[float, float]:
    """Return the signed IQ principal direction, or I if no contrast is available."""
    measured = signals[np.isfinite(signals)]
    if measured.size < 2 or np.all(measured == measured[0]):
        return (1.0, 0.0)
    angle = find_rotate_angle(measured)
    return (float(np.cos(angle)), float(np.sin(angle)))


def t2echo_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Project the mean complex IQ used by the completed-round SNR checker."""
    return rotate2real(signals).real


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


_TIME_AXIS = VariableAxisSpec(field_name="times", label="Time", unit="s", scale=1e-6)
_MOMENT_COLUMNS = (0, 1, 2, 3, 4, 5)


def _build_result(data: GroupedLoadData[T2EchoCfg]) -> T2EchoResult:
    """Restore mean IQ and optional moments aligned to the same delays."""
    signal = data.variable("data")
    (times,) = signal.axes
    detune = data.run_values.nullable_scalar("true_activate_detune")
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return T2EchoResult(times, signal.z, true_activate_detune=detune)
    columns, moment_times = moments.axes
    if not np.array_equal(times, moment_times):
        raise ValueError("/vars/iq_moments: Time axis must equal /vars/data Time axis")
    if not np.array_equal(columns, _MOMENT_COLUMNS):
        raise ValueError(
            "/vars/iq_moments: Moment axis must contain columns 0 through 5"
        )
    return T2EchoResult(
        times, signal.z, true_activate_detune=detune, iq_moments=moments.z
    )


class T2EchoExp(Experiment[T2EchoResult, T2EchoCfg]):
    Options: ClassVar[type[T2EchoAnalyzeOptions]] = T2EchoAnalyzeOptions

    # ADR-0069: optional Native moments do not change the Labber projection.
    AXES_SPEC: ClassVar[GroupedAxesSpec[T2EchoResult, T2EchoCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "data",
                (_TIME_AXIS,),
                VariableZSpec(
                    "signals", label="Signal", unit="a.u.", dtype=np.complex128
                ),
            ),
            VariableSpec(
                "iq_moments",
                (
                    VariableAxisSpec(label="Moment", unit="", values=_MOMENT_COLUMNS),
                    _TIME_AXIS,
                ),
                VariableZSpec("iq_moments", label="IQ moments", unit=""),
                required=False,
            ),
        ),
        result_type=T2EchoResult,
        cfg_type=T2EchoCfg,
        cfg_schema_version="4.0",
        tag="twotone/ge/t2echo",
        result_builder=_build_result,
        labber_variables=("data",),
        run_values=(
            RunValueSpec(
                field_name="true_activate_detune",
                wire_name="true_activate_detune",
                unit="Hz",
                scale=1e6,
                allow_unknown=True,
            ),
        ),
    )

    def run(self, cfg: T2EchoCfg, *, context: RunContext) -> T2EchoResult:
        """Measure refocused coherence with a Hahn-echo delay sweep.

        After optional reset, apply pi/2, half the total delay, pi, the other
        half delay, then pi/2 and readout. cfg supplies the pulses and total
        delay sweep in us. detune in MHz ramps the final pi/2 phase with total
        delay to add fringes; it does not change the drive frequency.
        Return hardware-rounded total delays, mean complex IQ and the
        offset-corrected shot sums from those same completed rounds.
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
        ax.set(
            title="Spin echo",
            xlabel=_DELAY_LABEL,
            ylabel=_IQ_LABEL,
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        viewer = live_errorbar(ax)

        @context.plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Refresh the projected mean and shot SEM."""
            _, mean = split_sums(moments)
            i, q = _iq_direction(mean)
            sem = iq_shot_sem(moments, direction=(i, q))
            return viewer.set(lengths, mean.real * i + mean.imag * q, sem)

        fig.show()
        moments_buffer = SignalBuffer(
            (len(lengths), 6), dtype=np.float64, on_update=update
        )
        # 3. Measure midpoint-refocused coherence with a phase-ramped final projection.
        with Schedule(moments_buffer, scope=context.scope) as sched:
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
                        moments_buffer.at(),
                        cfg.earlystop_snr,
                        lambda sums: t2echo_signal2real(split_sums(sums)[1]),
                    ),
                ),
            )

        # 4. Return the measured trace and its shot statistics from the same rounds.
        return T2EchoResult(
            times=lengths,
            signals=split_sums(moments_buffer.array)[1],
            true_activate_detune=true_detune if np.isfinite(true_detune) else None,
            iq_moments=moments_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[T2EchoCfg, T2EchoResult],
        options: T2EchoAnalyzeOptions,
        *,
        plots: Plots,
    ) -> T2EchoAnalysis:
        """Fit T2 Echo in us from finite IQ after omitting the first delay.

        Plot the principal-axis projection and same-run shot SEM when present.
        If fitting raises, leave the measured trace without a model curve.

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
        fit_phase = options.fit_phase
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

        # 2. Build the measured trace before fitting so it remains available on failure.
        valid = np.isfinite(result.signals[1:])
        xs, signals = result.times[1:][valid], result.signals[1:][valid]
        i, q = _iq_direction(signals)
        real_signals = signals.real * i + signals.imag * q
        sem = (
            None
            if result.iq_moments is None
            else iq_shot_sem(result.iq_moments[1:][valid], direction=(i, q))
        )
        fig, ax = plots.subplots(
            num="fit", figsize=config.figsize, layout="constrained"
        )
        ax.set(title="Spin echo", xlabel=_DELAY_LABEL, ylabel=_IQ_LABEL)
        if sem is not None and np.isfinite(sem).any():
            measured, _, _ = ax.errorbar(
                xs, real_signals, yerr=sem, fmt=".-", label="Measured ± shot SEM"
            ).lines
        else:
            (measured,) = ax.plot(xs, real_signals, ".-", label="Measured")
        measured_legend = fig.legend(loc="outside lower center", ncols=2)

        # 3. Fit the same unweighted observations and overlay the model on a dense grid.
        if fit_method == "fringe":
            fixedparams = None if fit_phase else [None, None, None, 0.0, None]
            fit = fit_decay_fringe(xs, real_signals, fixedparams=fixedparams)
            model = decaycos
            label = "Fringe fit"
            detune, detune_err = fit.freq, fit.freq_err
        else:
            fit = fit_decay(xs, real_signals)
            model = expfunc
            label = "Decay fit"
            detune = detune_err = 0.0
        fit_delays = np.linspace(xs.min(), xs.max(), max(200, xs.size))
        measured.set_linestyle("none")
        ax.plot(
            fit_delays, model(fit_delays, *fit.params), color="tab:orange", label=label
        )
        measured_legend.remove()
        fig.legend(loc="outside lower center", ncols=2)

        # 4. Annotate the fitted lifetimes, frequency and quality without selection gates.
        t2e, t2eerr = fit.decay_time, fit.decay_time_err
        quality = fit.quality
        facts = [f"T₂ Echo = {t2e:.2f} ± {t2eerr:.2f} µs"]
        if fit_method == "fringe":
            facts.append(f"Fringe frequency = {detune:.4f} ± {detune_err:.4f} MHz")
        if quality.r2 is not None:
            facts.append(f"R² = {quality.r2:.4f}")
        if quality.normalized_residual_rms is not None:
            facts.append(
                f"Normalized residual RMS = {quality.normalized_residual_rms:.3g}"
            )
        ax.text(
            0.98,
            0.97,
            "\n".join(facts),
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

        return T2EchoAnalysis(
            t2e=float(t2e),
            t2e_err=float(t2eerr),
            detune=float(detune),
            detune_err=float(detune_err),
            fit_quality={"fit": quality},
        )
