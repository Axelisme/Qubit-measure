from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib import rcParams
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.analysis.fitting.allxy import fit_allxy_errors
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ComputedPulse,
    LoadValue,
    Pulse,
    Readout,
    Reset,
    TimelineBuilder,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_line


@dataclass(frozen=True)
class AllXY_Result:
    """Mean complex IQ for the fixed 21 AllXY gate pairs, shaped (21,)."""

    signals: NDArray[np.complex128]


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


def allxy_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    return rotate2real(signals).real


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


class AllXY_Exp(Experiment[AllXY_Result, AllXYCfg]):
    Options: ClassVar[type[AllXYAnalyzeOptions]] = AllXYAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[AllXY_Result, AllXYCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                values=tuple(range(len(ALLXY_SEQUENCE))),
                label="Gate Pair Index",
                unit="",
                dtype=np.int64,
            ),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
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
        ax.set(xlabel="Gate", ylabel="Signal (a.u.)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid()
        viewer = live_line(ax)
        name_map = {
            "I": "$I$",
            "X90": "$X_{90}$",
            "Y90": "$Y_{90}$",
            "X180": "$X_{180}$",
            "Y180": "$Y_{180}$",
        }
        gate_labels = [
            f"{name_map[gate1]}-{name_map[gate2]}" for gate1, gate2 in ALLXY_SEQUENCE
        ]
        ax.set_xticks(np.arange(len(ALLXY_SEQUENCE)))
        ax.set_xticklabels(gate_labels, rotation=30, ha="right", fontsize=8)
        ax.grid(True)
        line = ax.lines[0]
        line.set_marker(".")
        line.set_linestyle(rcParams["lines.linestyle"])
        line.set_markersize(5)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            """Refresh the ALLXY contrast curve.

            data is complex IQ in ALLXY_SEQUENCE order; NaN marks unmeasured pairs.
            Return the changed artists without modifying the input array.
            """
            return viewer.set(
                np.arange(len(ALLXY_SEQUENCE), dtype=np.float64),
                allxy_signal2real(data),
            )

        fig.show()
        signals_buffer = SignalBuffer(
            (len(ALLXY_SEQUENCE),),
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
            sched.run_acquire(prog, soc, rounds=cfg.rounds)

        return AllXY_Result(
            signals=signals_buffer.array,
        )

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
        real_signals = allxy_signal2real(signals)

        # 2. Estimate coherent rotation errors from the 21 pair responses.

        fit = fit_allxy_errors(sequence, real_signals)

        # 3. Compare the measured staircase with the fitted low-order model.

        fig, ax = plots.subplots(num="fit", figsize=config.figsize)
        ax.plot(real_signals, marker="o", linestyle="None", label="Measured Signals")
        ax.plot(
            fit.predict_signals,
            marker="x",
            linestyle="-",
            color="red",
            label="Predicted Signals",
        )
        ax.axhline(y=fit.center, color="green", linestyle="--", alpha=0.2)
        ax.axhline(
            y=fit.center + 0.5 * fit.contrast,
            color="blue",
            linestyle="--",
            alpha=0.2,
        )
        ax.axhline(
            y=fit.center - 0.5 * fit.contrast,
            color="blue",
            linestyle="--",
            alpha=0.2,
        )

        ax.set_xlabel("Gate")
        ax.set_xticks(np.arange(len(sequence)))
        ax.set_xticklabels([f"{g1}-{g2}" for g1, g2 in sequence], rotation=45)

        ax.set_ylabel("Signal (a.u.)")
        ax.legend()
        ax.grid(True)

        ax.set_title(
            f"amp err: {fit.amplitude_error:+.2%}, power dep: {fit.power_err:.1%}, "
            f"detune dep: {fit.detune_err:.1%}, residual: {fit.residual_rms:.1%}"
        )

        fig.tight_layout()

        return AllXYAnalysis(
            amplitude_error=fit.amplitude_error,
            detune_param=fit.detune_param,
            power_err=fit.power_err,
            detune_err=fit.detune_err,
            residual_rms=fit.residual_rms,
        )
