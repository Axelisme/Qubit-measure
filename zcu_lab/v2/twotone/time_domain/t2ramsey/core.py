from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

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
class T2RamseyResult:
    """Free-evolution delays in us and mean complex IQ, one sample per delay.

    true_activate_detune is the phase-ramp frequency in MHz after hardware
    time/phase rounding, not a fitted qubit detuning. It is run-only metadata
    and is None after canonical load. iq_moments stores (N, 6) retained shot
    sums [count, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²], or None for a mean-only file.
    """

    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    true_activate_detune: float | None = None
    iq_moments: NDArray[np.float64] | None = None


@validated_dataclass(frozen=True)
class T2RamseyAnalyzeOptions:
    fit_fringe: Annotated[bool, FieldMeta(label="Fit fringe")] = True
    fit_phase: Annotated[bool, FieldMeta(label="Fit phase offset (fringe only)")] = True


@dataclass(frozen=True)
class T2RamseyAnalysis:
    t2r: float
    t2r_err: float
    detune: float
    detune_err: float
    fit_quality: Mapping[str, FitQuality] | None = None


def _direction(signals: NDArray[np.complex128]) -> tuple[float, float]:
    """Get the principal IQ direction, or the I axis for fewer than two points."""
    try:
        angle = find_rotate_angle(signals)
    except ValueError:
        return (1.0, 0.0)
    return (float(np.cos(angle)), float(np.sin(angle)))


def t2ramsey_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    i, q = _direction(signals)
    return signals.real * i + signals.imag * q


_TIME_AXIS = VariableAxisSpec(label="Time", unit="s", scale=1e-6, field_name="times")
_MOMENT_COLUMNS = tuple(range(6))
_DELAY_LABEL = "Free-evolution delay (µs)"
_IQ_LABEL = "IQ projection (a.u.)"


def _build_result(data: GroupedLoadData[T2RamseyCfg]) -> T2RamseyResult:
    """Restore mean IQ and optional sums on the same saved delay grid."""
    trace = data.variable("data")
    (times,) = trace.axes
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return T2RamseyResult(times, trace.z)
    columns, moment_times = moments.axes
    if not np.array_equal(times, moment_times) or not np.array_equal(
        columns, _MOMENT_COLUMNS
    ):
        raise ValueError(
            "/vars/iq_moments: expected columns 0–5 on the /vars/data delay grid"
        )
    return T2RamseyResult(times, trace.z, iq_moments=moments.z)


class T2RamseyCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    pi2_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Pi/2 Pulse")]
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


class T2RamseyExp(Experiment[T2RamseyResult, T2RamseyCfg]):
    Options: ClassVar[type[T2RamseyAnalyzeOptions]] = T2RamseyAnalyzeOptions

    # ADR-0069: keep the native mean-IQ variable and add optional shot moments.
    AXES_SPEC: ClassVar[GroupedAxesSpec[T2RamseyResult, T2RamseyCfg]] = GroupedAxesSpec(
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
        result_builder=_build_result,
        labber_variables=("data",),
        result_type=T2RamseyResult,
        cfg_type=T2RamseyCfg,
        cfg_schema_version="4.0",
        tag="twotone/ge/t2ramsey",
    )

    def run(self, cfg: T2RamseyCfg, *, context: RunContext) -> T2RamseyResult:
        """Measure Ramsey fringes and free-evolution coherence decay.

        After optional reset, apply pi/2, wait the swept delay, then apply the
        final pi/2 and read out. cfg supplies the pulses and delay sweep in us.
        detune in MHz ramps the final pi/2 phase with delay to add fringes;
        it does not change the drive frequency. Return hardware-rounded delays
        and mean complex IQ. analyze fits the envelope to estimate T2 Ramsey
        and, for a fringe fit, the observed oscillation frequency.
        """
        # 1. Prepare the delay grid and the realized phase-ramp frequency.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        detune = cfg.detune

        length_sweep = cfg.delay
        lengths = sweep2array(length_sweep, "time", {"soccfg": soccfg})

        true_detune = actual_detune(
            cfg.delay,
            detune=detune,
            gen_ch=cfg.pi2_pulse.ch,
            soccfg=soccfg,
            time_scaler=1.0,
        )

        # 2. Display the IQ projection while averaging the delay sweep.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(title="Ramsey fringes", xlabel=_DELAY_LABEL, ylabel=_IQ_LABEL)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        viewer = live_errorbar(ax)

        @context.plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Refresh projected mean IQ and its shot SEM."""
            _, mean = split_sums(moments)
            i, q = _direction(mean)
            sem = iq_shot_sem(moments, direction=(i, q))
            return viewer.set(lengths, mean.real * i + mean.imag * q, sem)

        fig.show()
        signals_buffer = SignalBuffer(
            (len(lengths), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
        )
        # 3. Measure freely evolving coherence with a phase-ramped final projection.
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
            builder.emit(Delay("t2_delay", delay=length_param))
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
                        lambda sums: t2ramsey_signal2real(split_sums(sums)[1]),
                    ),
                ),
            )

        # 4. Preserve the realized phase-ramp frequency with the measured Ramsey trace.
        return T2RamseyResult(
            times=lengths,
            signals=split_sums(signals_buffer.array)[1],
            true_activate_detune=true_detune,
            iq_moments=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[T2RamseyCfg, T2RamseyResult],
        options: T2RamseyAnalyzeOptions,
        *,
        plots: Plots,
    ) -> T2RamseyAnalysis:
        """Fit T2 Ramsey in us and optionally the observed fringe frequency.

        Project source IQ onto its signal axis. fit_fringe selects a decaying
        fringe instead of a pure decay. fit_phase frees the fringe phase;
        otherwise the near-zero-delay signal selects a fixed 0/180 degree
        phase. Decay-only fits ignore fit_phase and return zero detune.
        Fringe detune is in MHz and includes the applied phase ramp, not a
        corrected qubit detuning. Return uncertainties and quality, and draw
        the fit using the same samples.
        """
        # 1. Recover real-IQ Ramsey contrast for the selected envelope and phase policy.
        result = source.result
        fit_fringe = options.fit_fringe
        fit_phase = options.fit_phase

        lengths, signals = result.times, result.signals

        i, q = _direction(signals)
        real_signals = signals.real * i + signals.imag * q
        fit_delays = np.linspace(lengths.min(), lengths.max(), max(200, lengths.size))

        # 2. Choose whether to fit oscillations as well as their decay envelope.
        if fit_fringe:
            fixedparams: list[float | None] | None = None
            if not fit_phase:
                zero_signal = real_signals[np.argmin(np.abs(lengths))]
                if zero_signal > 0.5 * (np.max(real_signals) + np.min(real_signals)):
                    init_phase = 0.0
                else:
                    init_phase = 180.0
                fixedparams = [None, None, None, init_phase, None]
            fit = fit_decay_fringe(lengths, real_signals, fixedparams=fixedparams)
            t2r, t2rerr, detune, detune_err = (
                fit.decay_time,
                fit.decay_time_err,
                fit.freq,
                fit.freq_err,
            )
            y_fit = decaycos(fit_delays, *fit.params)
        else:
            fit = fit_decay(lengths, real_signals)
            t2r, t2rerr = fit.decay_time, fit.decay_time_err
            y_fit = expfunc(fit_delays, *fit.params)
            detune = 0.0
            detune_err = 0.0

        # 3. Assess Ramsey coherence time and any fitted fringe frequency against the trace.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        if result.iq_moments is None:
            ax.plot(lengths, real_signals, ".", label="Measured")
        else:
            sem = iq_shot_sem(result.iq_moments, direction=(i, q))
            ax.errorbar(
                lengths, real_signals, yerr=sem, fmt=".", label="Measured ± shot SEM"
            )
        quality = fit.quality
        label = "Fringe fit" if fit_fringe else "Decay fit"
        if quality.r2 is not None:
            label += f" (R² = {quality.r2:.3f})"
        ax.plot(fit_delays, y_fit, label=label)
        title = f"Ramsey: T₂* = {t2r:.2f} ± {t2rerr:.2f} µs"
        if fit_fringe:
            title += f"\nFringe frequency = {detune:.4f} ± {detune_err:.4f} MHz"
        ax.set(title=title, xlabel=_DELAY_LABEL, ylabel=_IQ_LABEL)
        ax.legend()

        fig.tight_layout()

        return T2RamseyAnalysis(
            t2r=float(t2r),
            t2r_err=float(t2rerr),
            detune=float(detune),
            detune_err=float(detune_err),
            fit_quality={"fit": fit.quality},
        )
