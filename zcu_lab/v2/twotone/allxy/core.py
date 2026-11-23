from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.fitting.allxy import fit_allxy_errors
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
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    ComputedPulse,
    LoadValue,
    Pulse,
    Readout,
    Reset,
    TimelineBuilder,
    shot_sums,
    split_sums,
)
from zcu_tools.utils.process import find_rotate_angle

from zcu_lab.v2.support import live_errorbar


@dataclass(frozen=True)
class AllXY_Result:
    """Mean complex IQ for the fixed 21 AllXY gate pairs, shaped (21,)."""

    signals: NDArray[np.complex128]
    iq_moments: NDArray[np.float64] | None = None
    """Retained shot sums [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²] per point; None for mean-only files."""


# Standard AllXY sequence of 21 gate pairs
ALLXY_SEQUENCE = [
    ("I", "I"),
    ("X180", "X180"),
    ("Y180", "Y180"),
    ("X180", "Y180"),
    ("Y180", "X180"),
    ("X90", "I"),
    ("Y90", "I"),
    ("X90", "Y90"),
    ("Y90", "X90"),
    ("X90", "Y180"),
    ("Y90", "X180"),
    ("X180", "Y90"),
    ("Y180", "X90"),
    ("X90", "X180"),
    ("X180", "X90"),
    ("Y90", "Y180"),
    ("Y180", "Y90"),
    ("X180", "I"),
    ("Y180", "I"),
    ("X90", "X90"),
    ("Y90", "Y90"),
]

GATE_LIST = ["I", "X90", "Y90", "X180", "Y180"]
ALLXY_GATE1_IDX = [GATE_LIST.index(g1) for g1, _ in ALLXY_SEQUENCE]
ALLXY_GATE2_IDX = [GATE_LIST.index(g2) for _, g2 in ALLXY_SEQUENCE]

_GATE_NAME = {
    "I": "I",
    "X90": "X₉₀",
    "Y90": "Y₉₀",
    "X180": "X₁₈₀",
    "Y180": "Y₁₈₀",
}
_GATE_PAIR_LABELS = [f"{_GATE_NAME[g1]}–{_GATE_NAME[g2]}" for g1, g2 in ALLXY_SEQUENCE]

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


@validated_dataclass(frozen=True)
class AllXYAnalyzeOptions:
    """ALLXY analysis has no options; g/e levels are always fitted."""


# ------------------------------------------------------------------------------
# AllXYExperiment
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class AllXYAnalysis:
    """Fitted gate errors.

    ``amplitude_error`` is the relative drive amplitude error: +0.02 means the
    pulses rotate 2% too far. ``detune_param`` is the dimensionless detuning
    parameter of the Reed thesis model.

    ``power_err``, ``detune_err`` and ``residual_rms`` are in Bloch-z units,
    where ground is +1 and excited is -1. ``power_err`` and ``detune_err`` are
    the mean absolute deviation each fitted error alone causes over the 21
    pairs; they are not gate infidelities. ``residual_rms`` is the RMS
    distance between the data and the fitted model. A residual comparable to
    the fitted deviations means the low-order model does not describe the
    data, for example because T1/T2 decay during the sequence matters.
    """

    amplitude_error: float
    detune_param: float
    power_err: float
    detune_err: float
    residual_rms: float


def _direction(signals: NDArray[np.complex128]) -> tuple[float, float]:
    """Project along the principal IQ axis, or use I if fewer than two points."""
    try:
        angle = find_rotate_angle(signals)
    except ValueError:
        return (1.0, 0.0)
    return (float(np.cos(angle)), float(np.sin(angle)))


def allxy_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    i, q = _direction(signals)
    return signals.real * i + signals.imag * q


class AllXYCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    I_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Identity Pulse")] = None
    X180_pulse: Annotated[Pulse.Cfg, FieldMeta(label="X180 Pulse")]
    X90_pulse: Annotated[Pulse.Cfg, FieldMeta(label="X90 Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
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
    label="Gate Pair Index",
    unit="",
    values=tuple(range(len(ALLXY_SEQUENCE))),
    dtype=np.int64,
)
_MOMENT_COLUMNS = tuple(range(6))


def _build_result(data: GroupedLoadData[AllXYCfg]) -> AllXY_Result:
    """Restore the measured trace and any shot moments on the same native axis."""
    trace = data.variable("data")
    (coordinates,) = trace.axes
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return AllXY_Result(trace.z)
    columns, moment_coordinates = moments.axes
    if not np.array_equal(coordinates, moment_coordinates) or not np.array_equal(
        columns, _MOMENT_COLUMNS
    ):
        raise ValueError(
            "/vars/iq_moments: expected columns 0–5 on the /vars/data grid"
        )
    return AllXY_Result(trace.z, iq_moments=moments.z)


class AllXY_Exp(Experiment[AllXY_Result, AllXYCfg]):
    Options: ClassVar[type[AllXYAnalyzeOptions]] = AllXYAnalyzeOptions

    # ADR-0069: preserve the original trace; shot moments are optional on load.
    AXES_SPEC: ClassVar[GroupedAxesSpec[AllXY_Result, AllXYCfg]] = GroupedAxesSpec(
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
        result_type=AllXY_Result,
        cfg_type=AllXYCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/allxy",
    )

    def run(self, cfg: AllXYCfg, *, context: RunContext) -> AllXY_Result:
        """Measure the 21 AllXY gate pairs to diagnose rotation and detuning errors.

        Reset, play the pair's two register-selected gates in order, then
        read out. Y gates use the corresponding X pulse shifted by 90 degrees;
        absent I uses a zero-gain X90 pulse. Return gate-pair indices and mean
        complex IQ, not per-gate fidelities. Gate pulses must satisfy
        ComputedPulse's common-channel, pre-delay and waveform-style rules.
        """
        # 1. Isolate the gate calibrations used to measure the ALLXY staircase.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        # 2. Track the staircase with gate-pair labels in acquisition order.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(
            title="AllXY gate-pair response",
            xlabel="Ordered gate pair",
            ylabel="IQ projection (a.u.)",
        )
        viewer = live_errorbar(ax)
        ax.lines[0].set_linestyle("none")
        ax.set_xticks(
            np.arange(len(ALLXY_SEQUENCE)), _GATE_PAIR_LABELS, rotation=90, fontsize=10
        )
        fig.tight_layout()

        @context.plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Refresh projected mean IQ and shot SEM per gate pair."""
            _, mean = split_sums(moments)
            i, q = _direction(mean)
            sem = iq_shot_sem(moments, direction=(i, q))
            return viewer.set(
                np.arange(len(ALLXY_SEQUENCE), dtype=np.float64),
                mean.real * i + mean.imag * q,
                sem,
            )

        fig.show()
        signals_buffer = SignalBuffer(
            (len(ALLXY_SEQUENCE), len(_MOMENT_COLUMNS)),
            dtype=np.float64,
            on_update=update,
        )
        # 3. Measure all 21 gate pairs using the calibrated X rotations and derived Y rotations.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            # Derive Y rotations from X calibrations and retain an identity time slot.
            I_pulse = cfg.I_pulse
            X180_pulse = cfg.X180_pulse
            X90_pulse = cfg.X90_pulse
            Y180_pulse = X180_pulse.with_updates(phase=X180_pulse.phase + 90)
            Y90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase + 90)

            if I_pulse is None:
                I_pulse = X90_pulse.with_updates(gain=0.0)

            # Order must match GATE_LIST = ["I", "X90", "Y90", "X180", "Y180"]
            gate_pulses = [I_pulse, X90_pulse, Y90_pulse, X180_pulse, Y180_pulse]

            # Load each gate pair, reset, then play gate 1 -> gate 2 -> readout.
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("allxy_idx", len(ALLXY_SEQUENCE))
            builder.emit(
                LoadValue(
                    "load_gate1_idx",
                    values=ALLXY_GATE1_IDX,
                    idx_reg="allxy_idx",
                    val_reg="gate_idx1",
                )
            )
            builder.emit(
                LoadValue(
                    "load_gate2_idx",
                    values=ALLXY_GATE2_IDX,
                    idx_reg="allxy_idx",
                    val_reg="gate_idx2",
                )
            )
            if cfg.reset is not None:
                builder.emit(cfg.reset.build("reset"))
            builder.emit(
                ComputedPulse("gate1", val_reg="gate_idx1", pulses=gate_pulses)
            )
            builder.emit(
                ComputedPulse("gate2", val_reg="gate_idx2", pulses=gate_pulses)
            )
            builder.emit(Readout("readout", cfg=cfg.readout))
            prog = builder.build()
            sched.run_acquire_summed(
                prog,
                soc,
                rounds=cfg.rounds,
                reducer=shot_sums(prog, (IQ("readout", offset_corrected=True),)),
            )

        moments = signals_buffer.array
        return AllXY_Result(signals=split_sums(moments)[1], iq_moments=moments)

    def analyze(
        self,
        source: RunRecord[AllXYCfg, AllXY_Result],
        options: AllXYAnalyzeOptions,
        *,
        plots: Plots,
    ) -> AllXYAnalysis:
        """Fit AllXY contrast to the low-order rotation/detuning error model.

        Always fit the IQ center and contrast along with the coherent errors.
        Return relative amplitude error, dimensionless detuning and Bloch-z
        deviations/residuals as defined in AllXYAnalysis, not gate infidelities.
        Publish ``fit``; source cfg may be None.
        """
        del options
        result = source.result
        signals = result.signals

        # 1. Project the gate-pair IQ onto a common contrast axis for error estimation.
        sequence = ALLXY_SEQUENCE
        i, q = _direction(signals)
        real_signals = signals.real * i + signals.imag * q

        # 2. Estimate coherent rotation errors from the 21 pair responses.

        fit = fit_allxy_errors(sequence, real_signals)

        # 3. Compare the measured staircase with the fitted low-order model.

        fig, ax = plots.subplots(num="fit", figsize=config.figsize)
        if result.iq_moments is None:
            ax.plot(real_signals, ".", label="Measured")
        else:
            indices = np.arange(real_signals.size)
            sem = iq_shot_sem(result.iq_moments, direction=(i, q))
            ax.errorbar(
                indices, real_signals, yerr=sem, fmt=".", label="Measured ± shot SEM"
            )
        ax.plot(
            fit.predict_signals,
            marker="x",
            linestyle="none",
            color="red",
            label="Coherent-error model",
        )

        ax.set(xlabel="Ordered gate pair", ylabel="IQ projection (a.u.)")
        ax.set_xticks(
            np.arange(len(sequence)), _GATE_PAIR_LABELS, rotation=90, fontsize=10
        )
        ax.legend()

        ax.set_title(
            f"AllXY · Relative drive error {fit.amplitude_error:+.2%}\n"
            f"Amplitude deviation: {fit.power_err:.3f} in Bloch z\n"
            f"Detuning deviation: {fit.detune_err:.3f} in Bloch z\n"
            f"Residual RMS: {fit.residual_rms:.3f} in Bloch z",
            fontsize=10,
        )

        fig.tight_layout()

        return AllXYAnalysis(
            amplitude_error=fit.amplitude_error,
            detune_param=fit.detune_param,
            power_err=fit.power_err,
            detune_err=fit.detune_err,
            residual_rms=fit.residual_rms,
        )
