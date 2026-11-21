from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from qick.asm_v2 import QickSweep1D
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
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    IQ,
    BathReset,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    TimelineProgramV2,
    shot_sums,
    split_sums,
)

from zcu_lab.v2.support import live_errorbar, live_line


@dataclass(frozen=True)
class LengthResult:
    """Qubit-tone lengths in us and mean IQ (length, absolute pi/2 phase in degrees)."""

    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]
    # [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²] per duration and phase.
    iq_moments: NDArray[np.float64] | None = None


def bathreset_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    return np.asarray(
        np.abs(signals[..., 2] - signals[..., 0]) ** 2
        + np.abs(signals[..., 3] - signals[..., 1]) ** 2,
        dtype=np.float64,
    )  # (lengths, )


class LengthCfg(ConfigBase):
    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a preparation reset before each shot, or leave blank to skip it.",
        ),
    ] = None
    init_pulse: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="Preparation pulse",
            description="Select the pulse after preparation reset, such as calibrated Xπ, or leave blank to skip it.",
        ),
    ] = None
    tested_reset: Annotated[
        BathReset.Cfg,
        FieldMeta(
            label="Tested reset",
            description="Fill in the cavity and qubit drives and the calibrated tomography π/2 pulse.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Fill in the calibrated resonator readout for the end of each shot.",
        ),
    ]
    length: Annotated[
        SweepCfg,
        FieldMeta(
            label="Qubit-tone duration",
            unit="µs",
            description="Enter the qubit-tone duration range (µs); the cavity drive keeps its configured duration difference.",
        ),
    ]

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per sweep point and sequence branch in each acquisition round (count).",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter the number of acquisition rounds to average (count).",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the delay after each shot to let the qubit relax (µs).",
        ),
    ]


_PHASES = (0.0, 90.0, 180.0, 270.0)
_MOMENT_COLUMNS = (0, 1, 2, 3, 4, 5)
_MEAN_AXES = (
    VariableAxisSpec(values=_PHASES, label="Pi2 Phase", unit="deg", dtype=np.float64),
    VariableAxisSpec(
        field_name="lengths", label="Length", unit="s", scale=1e-6, dtype=np.float64
    ),
)
_XLABEL = "Qubit-tone duration (µs)"
_CONTRAST_LABEL = "Opposite-phase IQ contrast (a.u.²)"
_TITLE = "Bath reset duration · qubit · tomography"


def _restore(data: GroupedLoadData[LengthCfg]) -> LengthResult:
    """Restore historical phase means and optional moments on the same grid."""
    mean = data.variable("data")
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return LengthResult(mean.axes[1], mean.z)
    if not np.array_equal(moments.axes[0], _MOMENT_COLUMNS) or not all(
        np.array_equal(one, other)
        for one, other in zip(mean.axes, moments.axes[1:], strict=True)
    ):
        raise ValueError("iq_moments axes do not match the IQ duration and phase grid")
    return LengthResult(mean.axes[1], mean.z, moments.z)


def _combine_outer_rounds(
    outer_moments: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Keep the historical equal-weight mean of the completed outer slots.

    Pool shot sums only where nonempty outer slots have equal counts. A
    stopped inner acquisition can leave a partial slot; its mean still gets
    one outer vote, so pooled SEM would describe a different estimator.
    """
    counts, means = split_sums(outer_moments)
    present = np.isfinite(means)
    count = present.sum(axis=0)
    avg = np.full(means.shape[1:], np.nan + 1j * np.nan, dtype=np.complex128)
    np.divide(np.nansum(means, axis=0), count, out=avg, where=count > 0)
    largest = np.max(np.where(present, counts, 0.0), axis=0)
    poolable = (count > 0) & np.all(~present | (counts == largest), axis=0)
    return avg, np.where(poolable[..., None], np.nansum(outer_moments, axis=0), np.nan)


class LengthExp(Experiment[LengthResult, LengthCfg]):
    # ADR-0069: retain the original phase-resolved data and Labber log.
    AXES_SPEC: ClassVar[GroupedAxesSpec[LengthResult, LengthCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "data",
                _MEAN_AXES,
                VariableZSpec(
                    "signals", label="Signal", unit="a.u.", dtype=np.complex128
                ),
            ),
            VariableSpec(
                "iq_moments",
                (
                    VariableAxisSpec(label="Moment", unit="", values=_MOMENT_COLUMNS),
                    *_MEAN_AXES,
                ),
                VariableZSpec("iq_moments", label="IQ moments", unit=""),
                required=False,
            ),
        ),
        result_type=LengthResult,
        cfg_type=LengthCfg,
        cfg_schema_version="3.0",
        tag="twotone/reset/bath/length",
        result_builder=_restore,
        labber_variables=("data",),
    )

    def run(
        self,
        config: LengthCfg,
        *,
        context: RunContext,
    ) -> LengthResult:
        """Measure bath-reset tomography contrast versus qubit-tone duration.

        After optional reset and initialization, start cavity and qubit tones
        together, then apply the pi/2 pulse after the cavity tone and read
        out. Use absolute pi/2 phases of 0, 90, 180 and 270 degrees.
        Host-scan qubit-tone length in us while
        preserving the configured cavity-minus-qubit length difference.
        Return lengths and round-averaged complex IQ shaped (Nlength, 4).
        Opposite-phase contrast is not a calibrated reset fidelity.
        """
        # 1. Prepare qubit-tone lengths while preserving the cavity-tone duration offset.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg

        rounds = cfg.rounds  # round loop is driven by Schedule.repeat below

        length_sweep = cfg.length
        tested_reset = cfg.tested_reset
        qub_lengths = sweep2array(
            length_sweep,
            "time",
            {"soccfg": soccfg, "gen_ch": tested_reset.qubit_tone_cfg.ch},
            allow_array=True,
        )
        lengths = qub_lengths  # use qubit tone length as x-axis

        orig_res_length = tested_reset.cavity_tone_cfg.waveform.length
        orig_qub_length = tested_reset.qubit_tone_cfg.waveform.length
        length_diff = orig_res_length - orig_qub_length

        prog_cache: dict[float, TimelineProgramV2] = {}
        reducers = {}

        # 2. Show the tomography contrast and the IQ means that determine it.
        plots = context.plots
        fig, _ = plots.subplots(
            num="measurement",
            nrows=3,
            sharex=True,
            figsize=(8, 8),
            layout="constrained",
        )
        contrast_ax, i_ax, q_ax = fig.axes
        fig.suptitle(_TITLE)
        contrast_ax.set_ylabel(_CONTRAST_LABEL)
        contrast = live_line(contrast_ax)
        i_ax.set_ylabel("Mean IQ real (a.u.)")
        q_ax.set(xlabel=_XLABEL, ylabel="Mean IQ imaginary (a.u.)")
        i_view, q_view = (
            live_errorbar(i_ax, num_lines=4),
            live_errorbar(q_ax, num_lines=4),
        )
        for line, phase in zip(i_ax.lines, _PHASES, strict=True):
            line.set_label(f"π/2 at {phase:g}°")
        q_ax.legend(
            handles=tuple(i_ax.lines),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncols=4,
        )

        @plots.on_plot_thread
        def update(data: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Update tomography contrast, the phase means and their direct shot SEM."""
            signals, moments = _combine_outer_rounds(data)
            i_sem = iq_shot_sem(moments, direction=(1.0, 0.0))
            q_sem = iq_shot_sem(moments, direction=(0.0, 1.0))
            shots = int(np.nan_to_num(moments[..., 0]).max())
            title = contrast_ax.set_title(
                f"Up to {shots} retained shots per duration and phase (count)"
            )
            return (
                *contrast.set(lengths, bathreset_signal2real(signals)),
                *i_view.set(lengths, signals.real.T, i_sem.T),
                *q_view.set(lengths, signals.imag.T, q_sem.T),
                title,
            )

        fig.show()
        buffer = SignalBuffer(
            (rounds, len(lengths), len(_PHASES), len(_MOMENT_COLUMNS)),
            dtype=np.float64,
            on_update=update,
        )

        # 3. Resolve four-phase tomography versus tone duration while retaining the cavity offset.
        with Schedule(buffer, scope=context.scope) as sched:
            for _, rep in sched.repeat("rounds", rounds):
                for length, step in rep.scan("length", lengths.tolist()):
                    program = prog_cache.get(length)
                    if program is None:
                        builder = TimelineBuilder(
                            soccfg,
                            reps=cfg.reps,
                            relax_delay=cfg.relax_delay,
                        )
                        builder.add_loop("phase", 4)
                        if cfg.reset is not None:
                            builder.emit(Reset("reset", cfg.reset))
                        builder.emit(Pulse("init_pulse", cfg.init_pulse))
                        builder.emit(
                            BathReset(
                                "tested_reset",
                                cfg.tested_reset,
                                qub_length=length,
                                res_length=length + length_diff,
                                pi2_phase=QickSweep1D("phase", 0.0, 270.0),
                            )
                        )
                        builder.emit(Readout("readout", cfg.readout))
                        program = builder.build()
                        prog_cache[length] = program

                        reducers[length] = shot_sums(
                            program, (IQ("readout", offset_corrected=True),)
                        )

                    step.run_acquire_summed(
                        program, soc, reducer=reducers[length], rounds=cfg.rounds
                    )
        signals, moments = _combine_outer_rounds(buffer.array)
        return LengthResult(lengths, signals, moments)

    def analyze(
        self,
        source: RunRecord[LengthCfg, LengthResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot the four-phase contrast versus qubit-tone length in us.

        Use opposite-phase IQ differences from source.result and publish fit.
        No duration is fitted or returned; source.cfg may be None.
        """
        # 1. Recover reset tomography contrast from the round-averaged phase responses.
        del options
        result = source.result

        lens, signals = result.lengths, result.signals

        real_signals = bathreset_signal2real(signals)

        # 2. Show the unchanged nonlinear contrast with phase-resolved IQ and SEM.
        fig, _ = plots.subplots(
            num="fit", nrows=3, sharex=True, figsize=(8, 8), layout="constrained"
        )
        contrast_ax, i_ax, q_ax = fig.axes
        fig.suptitle(_TITLE)
        contrast_ax.plot(lens, real_signals, ".-", markersize=5)
        contrast_ax.set_ylabel(_CONTRAST_LABEL)
        for ax, values, direction in (
            (i_ax, signals.real, (1.0, 0.0)),
            (q_ax, signals.imag, (0.0, 1.0)),
        ):
            sem = (
                np.full(values.shape, np.nan)
                if result.iq_moments is None
                else iq_shot_sem(result.iq_moments, direction=direction)
            )
            for phase, means, errors in zip(_PHASES, values.T, sem.T, strict=True):
                ax.errorbar(
                    lens,
                    means,
                    yerr=errors,
                    fmt=".-",
                    markersize=5,
                    label=f"π/2 at {phase:g}°",
                )
        i_ax.set_ylabel("Mean IQ real (a.u.)")
        q_ax.set(xlabel=_XLABEL, ylabel="Mean IQ imaginary (a.u.)")
        q_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncols=4)
