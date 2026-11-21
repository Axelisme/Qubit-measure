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
from zcu_tools.analysis.fitting import FitQuality, cosfunc, decaycos, fit_rabi
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
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
    split_sums,
)
from zcu_tools.utils.process import find_rotate_angle

from zcu_lab.v2.support import live_errorbar


@dataclass(frozen=True)
class LenRabiResult:
    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]
    iq_moments: NDArray[np.float64] | None = None
    """Retained shot sums [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²] per point; None for mean-only files."""


@validated_dataclass(frozen=True)
class LenRabiAnalyzeOptions:
    decay: Annotated[bool, FieldMeta(label="Fit decay envelope")] = True
    fit_phase: Annotated[bool, FieldMeta(label="Fit phase offset")] = False


@dataclass(frozen=True)
class LenRabiAnalysis:
    pi_len: float
    pi_len_err: float
    pi2_len: float
    pi2_len_err: float
    rabi_f: float
    rabi_f_err: float
    fit_quality: Mapping[str, FitQuality] | None = None


def _direction(signals: NDArray[np.complex128]) -> tuple[float, float]:
    """Project along the principal IQ axis, or use I if fewer than two points."""
    try:
        angle = find_rotate_angle(signals)
    except ValueError:
        return (1.0, 0.0)
    return (float(np.cos(angle)), float(np.sin(angle)))


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    i, q = _direction(signals)
    return signals.real * i + signals.imag * q


class LenRabiCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Rabi Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="Pulse length",
            unit="µs",
            description="Sweep the drive-pulse length while its gain and carrier stay fixed.",
        ),
    ]
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
    label="Length",
    unit="s",
    field_name="lengths",
    scale=1e-6,
)
_MOMENT_COLUMNS = tuple(range(6))


def _build_result(data: GroupedLoadData[LenRabiCfg]) -> LenRabiResult:
    """Restore the measured trace and any shot moments on the same native axis."""
    trace = data.variable("data")
    (coordinates,) = trace.axes
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return LenRabiResult(coordinates, trace.z)
    columns, moment_coordinates = moments.axes
    if not np.array_equal(coordinates, moment_coordinates) or not np.array_equal(
        columns, _MOMENT_COLUMNS
    ):
        raise ValueError(
            "/vars/iq_moments: expected columns 0–5 on the /vars/data grid"
        )
    return LenRabiResult(coordinates, trace.z, iq_moments=moments.z)


class LenRabiExp(Experiment[LenRabiResult, LenRabiCfg]):
    Options: ClassVar[type[LenRabiAnalyzeOptions]] = LenRabiAnalyzeOptions

    # lengths stored in seconds on disk (mem us) -> scale=1e-6; z complex
    # ADR-0069: preserve the original trace; shot moments are optional on load.
    AXES_SPEC: ClassVar[GroupedAxesSpec[LenRabiResult, LenRabiCfg]] = GroupedAxesSpec(
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
        result_type=LenRabiResult,
        cfg_type=LenRabiCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/rabi_length",
    )

    def run(
        self,
        cfg: LenRabiCfg,
        *,
        context: RunContext,
    ) -> LenRabiResult:
        """Measure Rabi oscillations by sweeping drive length at fixed gain.

        Apply optional reset and initialization, then the length-swept qubit
        pulse and readout. Const/flat-top pulses sweep on the FPGA; other
        waveforms scan on the host and reuse each length's compiled program.
        Return quantized lengths in us and mean complex IQ at each length,
        retaining acquired data on cancellation and NaN for unmeasured points.
        Fit the oscillation with analyze() to obtain pi and pi/2 lengths.
        Reject native arb with ValueError before preparing coordinates or plots:
        its asset-derived length cannot be swept by this experiment.
        """
        # 1. Reject asset-defined pulse lengths that cannot support a duration scan.
        if cfg.qub_pulse.waveform.style == "arb":
            raise ValueError(
                "Len Rabi cannot sweep an arb waveform's asset-derived length. "
                "Select a waveform with an editable length instead."
            )

        # 2. Prepare length coordinates without changing the caller's cfg.
        cfg = deepcopy(cfg)
        soccfg = context.soccfg
        lengths = sweep2array(
            cfg.length,
            "time",
            {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
        )

        # 3. Choose hardware duration sweeps or host-rebuilt sampled waveforms.
        if cfg.qub_pulse.waveform.style in ["const", "flat_top"]:
            return self._run_for_flat(cfg, lengths, context=context)
        return self._run_for_arb(cfg, np.unique(lengths), context=context)

    def _run_for_flat(
        self, cfg: LenRabiCfg, lengths: NDArray[np.float64], *, context: RunContext
    ) -> LenRabiResult:
        """Sweep constant or flat-top pulse lengths in hardware and return mean IQ."""
        # 1. Monitor Rabi contrast across hardware-swept pulse durations.
        fig, ax = context.plots.subplots(num="measurement")
        experiment = "Length Rabi"
        ax.set(
            title=experiment, xlabel="Pulse length (µs)", ylabel="IQ projection (a.u.)"
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
        signals = SignalBuffer(
            (len(lengths), len(_MOMENT_COLUMNS)), dtype=np.float64, on_update=update
        )
        # 2. Measure duration-dependent rotations in one FPGA sweep with SNR early stopping.
        with Schedule(signals, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            length = builder.add_sweep("length", cfg.length)
            if cfg.reset is not None:
                builder.emit(cfg.reset.build("reset"))
            builder.emit(Pulse("init_pulse", cfg.init_pulse))
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse, length=length))
            builder.emit(Readout("readout", cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=shot_sums(prog, (IQ("readout", offset_corrected=True),)),
                rounds=cfg.rounds,
                options=RunOptions(
                    stop_condition=snr_checker(
                        signals.at(),
                        cfg.earlystop_snr,
                        lambda sums: rabi_signal2real(split_sums(sums)[1]),
                    ),
                ),
            )
        return LenRabiResult(
            lengths=lengths,
            signals=split_sums(signals.array)[1],
            iq_moments=signals.array,
        )

    def _run_for_arb(
        self, cfg: LenRabiCfg, lengths: NDArray[np.float64], *, context: RunContext
    ) -> LenRabiResult:
        # Historical name: editable shaped waveforms, not native arb assets.
        """Compile editable shaped lengths on the host and return round-averaged IQ."""
        # 1. Reuse each length's compiled waveform and average only acquired rounds.
        programs: dict[float, TimelineProgramV2] = {}

        def accumulated_moments(rounds: NDArray[np.float64]) -> NDArray[np.float64]:
            """Add completed rounds at each length, leaving unmeasured lengths NaN."""
            acquired = np.isfinite(rounds[..., 0])
            totals = np.sum(np.where(acquired[..., None], rounds, 0.0), axis=0)
            totals[~acquired.any(axis=0)] = np.nan
            return totals

        # 2. Monitor the round-averaged contrast and SNR of the host duration scan.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(
            title="Length Rabi · shaped pulse",
            xlabel="Pulse length (µs)",
            ylabel="IQ projection (a.u.)",
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        viewer = live_errorbar(ax)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Display projected IQ and shot SEM from completed length points."""
            moments = accumulated_moments(data)
            counts, mean = split_sums(moments)
            i, q = _direction(mean)
            sem = iq_shot_sem(moments, direction=(i, q))
            acquired = counts[np.isfinite(counts)]
            title = ax.set_title(
                f"Length Rabi · {acquired.size}/{len(lengths)} lengths\n"
                f"Up to {np.max(acquired, initial=0):.0f} retained shots/point"
            )
            return (*viewer.set(lengths, mean.real * i + mean.imag * q, sem), title)

        fig.show()
        signals = SignalBuffer(
            (cfg.rounds, len(lengths), len(_MOMENT_COLUMNS)),
            dtype=np.float64,
            on_update=update,
        )
        check_snr = snr_checker(
            signals.at(),
            cfg.earlystop_snr,
            lambda data: rabi_signal2real(split_sums(accumulated_moments(data))[1]),
        )
        # 3. Measure each sampled-waveform duration until rounds finish or SNR suffices.
        with Schedule(signals, scope=context.scope) as sched:
            for _, rep in sched.repeat("round", cfg.rounds):
                for length, step in rep.scan("length", lengths.tolist()):
                    length_key = float(length)
                    if length_key not in programs:
                        builder = TimelineBuilder(
                            context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                        )
                        if cfg.reset is not None:
                            builder.emit(cfg.reset.build("reset"))
                        builder.emit(Pulse("init_pulse", cfg.init_pulse))
                        builder.emit(Pulse("qub_pulse", cfg.qub_pulse, length=length))
                        builder.emit(Readout("readout", cfg.readout))
                        programs[length_key] = builder.build()
                    prog = programs[length_key]
                    step.run_acquire_summed(
                        prog,
                        context.soc,
                        reducer=shot_sums(
                            prog, (IQ("readout", offset_corrected=True),)
                        ),
                        rounds=1,
                    )
                if check_snr():
                    break
        moments = accumulated_moments(signals.array)
        return LenRabiResult(
            lengths=lengths, signals=split_sums(moments)[1], iq_moments=moments
        )

    def analyze(
        self,
        source: RunRecord[LenRabiCfg, LenRabiResult],
        options: LenRabiAnalyzeOptions,
        *,
        plots: Plots,
    ) -> LenRabiAnalysis:
        """Fit the measured IQ oscillation to derive pi and pi/2 lengths in us.

        Rotate source.result IQ to a real trace and discard unmeasured NaNs.
        options.decay enables a decay envelope; options.fit_phase frees the
        otherwise fixed zero phase (signed amplitude also covers 180 degrees).
        Publish the fit plot and return lengths, Rabi frequency in MHz, their
        uncertainties and fit quality. source.cfg may be None. All-NaN data
        raise ValueError; other invalid traces follow fit_rabi's failure rules.
        """
        # 1. Project mean IQ and keep only acquired length points.
        result = source.result

        lens, signals = result.lengths, result.signals

        i, q = _direction(signals)
        real_signals = signals.real * i + signals.imag * q

        nan_mask = np.isnan(real_signals)
        if np.all(nan_mask):
            raise ValueError("All data are NaN!")

        lens = lens[~nan_mask]
        real_signals = real_signals[~nan_mask]

        # 2. Fit this length scan with the user's envelope and phase choices.
        fit = fit_rabi(
            lens,
            real_signals,
            decay=options.decay,
            init_phase=None if options.fit_phase else 0.0,
        )
        pi_len, pi_len_err, pi2_len, pi2_len_err, freq, freq_err = (
            fit.pi_x,
            fit.pi_x_err,
            fit.pi2_x,
            fit.pi2_x_err,
            fit.freq,
            fit.freq_err,
        )

        # 3. Show the oscillation and the fitted calibration lengths.
        fig, ax = plots.subplots(num="fit", figsize=config.figsize)

        if result.iq_moments is None:
            ax.plot(lens, real_signals, ".", label="Measured")
        else:
            sem = iq_shot_sem(result.iq_moments[~nan_mask], direction=(i, q))
            ax.errorbar(
                lens, real_signals, yerr=sem, fmt=".", label="Measured ± shot SEM"
            )
        dense = np.linspace(lens.min(), lens.max(), max(200, lens.size))
        model = decaycos if options.decay else cosfunc
        fit_label = (
            "Fit" if fit.quality.r2 is None else f"Fit (R² = {fit.quality.r2:.3f})"
        )
        ax.plot(dense, model(dense, *fit.params), label=fit_label)
        ax.axvline(
            pi_len,
            ls="--",
            c="red",
            label=f"pi = {pi_len:.3g} ± {pi_len_err:.2g} µs",
        )
        ax.axvspan(pi_len - pi_len_err, pi_len + pi_len_err, color="red", alpha=0.2)
        ax.axvline(
            pi2_len,
            ls="--",
            c="red",
            label=f"pi/2 = {pi2_len:.3g} ± {pi2_len_err:.2g} µs",
        )
        ax.axvspan(pi2_len - pi2_len_err, pi2_len + pi2_len_err, color="red", alpha=0.2)
        ax.set_xlabel("Pulse length (µs)")
        ax.set_ylabel("IQ projection (a.u.)")
        ax.set_title(f"Length Rabi · Rabi frequency = {freq:.3f} ± {freq_err:.3f} MHz")
        ax.legend(loc="best")

        fig.tight_layout()

        # Include length and frequency uncertainties in the GUI summary as in the figure.
        return LenRabiAnalysis(
            pi_len,
            pi_len_err,
            pi2_len,
            pi2_len_err,
            freq,
            freq_err,
            fit_quality={"fit": fit.quality},
        )
