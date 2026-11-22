from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
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
    Pulse,
    PulseReset,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
    split_sums,
)

from zcu_lab.v2.support import live_errorbar
from zcu_lab.v2.twotone.reset._projection import principal_iq_direction


@dataclass(frozen=True)
class LengthResult:
    """Quantized lengths (us) and complex IQ shaped (before_pi, after_pi, length).

    Both pi switch dimensions are ordered off, then on.
    signals[a, b, i] is mean IQ at lengths[i] with those two switches.
    Unacquired values remain NaN.
    """

    lengths: NDArray[np.float64]
    signals: NDArray[np.complex128]
    # Per branch and point: [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]; None in mean-only files.
    iq_moments: NDArray[np.float64] | None = None


@pydantic_dataclass(frozen=True)
class LengthAnalyzeOptions:
    """Residual threshold for selection only; it does not change acquisition."""

    epsilon: Annotated[
        float,
        FieldMeta(
            label="Residual limit (a.u.)",
            description="Maximum p0≈0 residual at the candidate and all longer finite points.",
        ),
        Field(allow_inf_nan=False),
    ] = 0.05


@dataclass(frozen=True)
class LengthAnalysis:
    """Shortest qualifying duration (us) and the operator's residual threshold.

    Every longer measured finite point must also pass epsilon. No qualifying
    point gives None. Contrasts and p0≈0 residual estimates stay in the plot;
    they are not reset fidelity.
    """

    selected_length: float | None
    epsilon: float


class LengthCfg(ConfigBase):
    """Sweep a flux reset pulse's length between two calibrated pi switches."""

    reset: Annotated[
        Reset.Cfg | None,
        FieldMeta(
            label="Reset",
            description="Select a preparation reset before the first π switch, or leave blank to skip it.",
        ),
    ] = None
    pi_pulse: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Pi pulse",
            description="Fill in the calibrated Xπ used by both switches around the tested reset.",
        ),
    ]
    tested_reset: Annotated[
        PulseReset.Cfg,
        FieldMeta(
            label="Tested reset",
            description="Fill in the local flux pulse at fixed gain; this scan replaces its duration.",
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
            label="Flux reset duration",
            unit="µs",
            description="Enter the tested flux-pulse duration range (µs).",
        ),
    ]

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Repetitions per branch/length per round.",
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


_SCAN_AXIS = VariableAxisSpec(
    field_name="lengths",
    label="Flux reset length",
    unit="s",
    scale=1e-6,
    dtype=np.float64,
)
_BRANCH_0 = VariableAxisSpec(
    values=(0, 1), label="After reset pi (0=off, 1=on)", unit="None", dtype=np.int64
)
_BRANCH_1 = VariableAxisSpec(
    values=(0, 1), label="Before reset pi (0=off, 1=on)", unit="None", dtype=np.int64
)
_MEAN_AXES = (_SCAN_AXIS, _BRANCH_0, _BRANCH_1)
_MOMENT_COLUMNS = (0, 1, 2, 3, 4, 5)
_TITLE = "Flux reset duration sweep · qubit"
_XLABEL = "Flux reset duration (µs)"
_BRANCH_LABELS = ("No π pulses", "π after", "π before", "π before and after")
_I_LABEL = "Mean IQ real (a.u.)"
_Q_LABEL = "Mean IQ imaginary (a.u.)"


def _restore(data: GroupedLoadData[LengthCfg]) -> LengthResult:
    """Load mean-only files or the mean and moments sharing its acquisition grid."""
    mean = data.variable("data")
    moments = data.variables.get(DataVariable("iq_moments"))
    if moments is None:
        return LengthResult(mean.axes[0], mean.z)
    if not np.array_equal(moments.axes[0], _MOMENT_COLUMNS) or not all(
        np.array_equal(old, new)
        for old, new in zip(mean.axes, moments.axes[1:], strict=True)
    ):
        raise ValueError(
            "iq_moments axes do not match the stored IQ grid and moment columns"
        )
    return LengthResult(mean.axes[0], mean.z, moments.z)


class LengthExp(Experiment[LengthResult, LengthCfg]):
    Options: ClassVar[type[LengthAnalyzeOptions]] = LengthAnalyzeOptions

    # ADR-0069: preserve historical data axes and export only the mean IQ to Labber.
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
        tag="twotone/reset/flux/length",
        result_builder=_restore,
        labber_variables=("data",),
    )

    def run(self, cfg: LengthCfg, *, context: RunContext) -> LengthResult:
        """Measure four mean-IQ branches per flux hold length at fixed gain.

        In one program apply optional reset, before-reset pi off/on, the tested
        flux pulse, after-reset pi off/on, and the same readout. Both switches
        use pi_pulse unchanged. Keep flux gain and pre/post delays fixed.
        Return rounded lengths (us) and IQ shaped (2, 2, N), switches [0, 1].
        Live uses the no-π branch's principal IQ axis for all four branches.
        """
        # 1. Quantize flux-reset hold lengths and monitor all four before/after-pi branches.
        cfg = deepcopy(cfg)
        pulse_cfg = cfg.tested_reset.pulse_cfg

        soc, soccfg = context.soc, context.soccfg
        lengths = sweep2array(
            cfg.length, "time", {"soccfg": soccfg, "gen_ch": pulse_cfg.ch}
        )
        plots = context.plots
        fig, ax = plots.subplots(num="measurement", layout="constrained")
        ax.set(
            xlabel=_XLABEL,
            ylabel="Mean IQ on no-π principal axis (a.u.)",
            title=f"{_TITLE} · IQ projection",
        )
        viewer = live_errorbar(ax, num_lines=4)
        for line, label in zip(ax.lines, _BRANCH_LABELS, strict=True):
            line.set_label(label)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2)

        @plots.on_plot_thread
        def update(moments: NDArray[np.float64]) -> tuple[Artist, ...]:
            """Show branch means and shot SEM on the same reference axis."""
            signals = split_sums(moments)[1]
            direction = principal_iq_direction(signals[0, 0])
            projected = direction[0] * signals.real + direction[1] * signals.imag
            sem = iq_shot_sem(moments, direction=direction)
            shots = int(np.nan_to_num(moments[..., 0]).max())
            title = ax.set_title(
                f"{_TITLE} · IQ projection\nRetained shots per point and branch: {shots} count"
            )
            return (
                *viewer.set(lengths, projected.reshape(4, -1), sem.reshape(4, -1)),
                title,
            )

        fig.show()
        buffer = SignalBuffer(
            (2, 2, len(lengths), len(_MOMENT_COLUMNS)),
            dtype=np.float64,
            on_update=update,
        )

        # 2. Compare before/after-pi responses at each hold length to isolate the reset contrast.
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            builder.add_loop("before_pi", 2)
            builder.add_loop("after_pi", 2)
            length = builder.add_sweep("length", cfg.length)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            # Keep the selected pi duration; the off case must not gain idle time.
            builder.delay_by_timeline()
            with (
                builder.branch(by="before_pi", align_timeline=False) as cases,
                cases.case(1),
            ):
                builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.delay_by_timeline()
            builder.emit(PulseReset("tested_reset", cfg.tested_reset, length=length))
            builder.delay_by_timeline()
            with (
                builder.branch(by="after_pi", align_timeline=False) as cases,
                cases.case(1),
            ):
                builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
            builder.delay_by_timeline()
            builder.emit(Readout("readout", cfg.readout))
            program = builder.build()
            reducer = shot_sums(program, (IQ("readout", offset_corrected=True),))
            sched.run_acquire_summed(program, soc, reducer=reducer, rounds=cfg.rounds)

        moments = buffer.array
        return LengthResult(lengths, split_sums(moments)[1], moments)

    def analyze(
        self,
        source: RunRecord[LengthCfg, LengthResult],
        options: LengthAnalyzeOptions,
        *,
        plots: Plots,
    ) -> LengthAnalysis:
        """Estimate residual excitation and select the shortest qualifying duration.

        Use d0=S00-S01, d1=S10-S11 and r=Re(d1*conj(d0))/|d0|².
        Under calibrated ideal pi, identical readout and a two-state model,
        p1=(1-r)/2+r*p0 exactly. Four IQs cannot identify p0 separately.
        Plot the unclipped estimate (1-r)/2 assuming reset hardly heats g
        (p0≈0). A thermally excited idle point, such as fluxonium near half
        flux, invalidates this approximation. The fixed offline validation
        model has idle first-excited-state population about 0.358. Only after
        an initial reset makes p0 small, with the other assumptions satisfied,
        can the estimate represent absolute residual excitation.
        Offline validation checks run, save, restore, analyze, PNG output and
        selection/no-solution handling, not physical reset effectiveness.
        This cannot exclude leakage or establish reset fidelity.
        Zero d0 or nonfinite IQ/contrasts/estimates are not candidates. Select the
        shortest quantized length with residual ≤ options.epsilon only if every
        longer measured finite point also passes. Scan order is irrelevant.
        No finite or qualifying candidate gives selected_length=None.
        Publish raw I/Q, r and residual curves. source.cfg may be None.
        """
        # 1. Require a complete before/after-pi branch layout for the hold-time estimate.
        result = source.result
        lengths, signals = result.lengths, result.signals
        if lengths.ndim != 1 or signals.shape != (2, 2, lengths.size):
            raise ValueError("Flux reset IQ must have shape (2, 2, number of lengths)")

        # 2. Find the shortest hold whose residual estimate stays below epsilon at longer lengths.
        # After-pi differences cancel the common readout offset, not p0.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            d0 = signals[0, 0] - signals[0, 1]
            d1 = signals[1, 0] - signals[1, 1]
            ratio = np.real(d1 * np.conj(d0)) / np.abs(d0) ** 2
            residual = (1 - ratio) / 2
        valid = (
            np.isfinite(lengths)
            & np.all(np.isfinite(signals), axis=(0, 1))
            & np.isfinite(d0)
            & np.isfinite(d1)
            & (d0 != 0)
            & np.isfinite(ratio)
            & np.isfinite(residual)
        )
        ratio = np.where(valid, ratio, np.nan)
        residual = np.where(valid, residual, np.nan)
        candidates = np.flatnonzero(valid)
        qualifying = [
            i
            for i in candidates
            if residual[i] <= options.epsilon
            and np.all(residual[valid & (lengths > lengths[i])] <= options.epsilon)
        ]
        selected = float(np.min(lengths[qualifying])) if qualifying else None

        # 3. Show the measured branches alongside the derived diagnostics.
        outcome = LengthAnalysis(selected_length=selected, epsilon=options.epsilon)
        self._plot_result(plots, result, ratio, residual, outcome)
        return outcome

    def _plot_result(
        self,
        plots: Plots,
        result: LengthResult,
        ratio: NDArray[np.float64],
        residual: NDArray[np.float64],
        selection: LengthAnalysis,
    ) -> None:
        """Plot I and Q with shot SEM and the unchanged residual selection."""
        lengths, signals = result.lengths, result.signals
        fig, _ = plots.subplots(
            num="fit",
            nrows=2,
            ncols=2,
            sharex=True,
            figsize=(9, 7),
            layout="constrained",
        )
        i_ax, q_ax, ratio_ax, residual_ax = fig.axes
        fig.suptitle(f"{_TITLE} · IQ and residual estimate")
        # Keep both readout quadratures to diagnose branch motion hidden by a single axis.
        for ax, values, direction in (
            (i_ax, signals.real, (1.0, 0.0)),
            (q_ax, signals.imag, (0.0, 1.0)),
        ):
            errors = (
                np.full(values.shape, np.nan)
                if result.iq_moments is None
                else iq_shot_sem(result.iq_moments, direction=direction)
            )
            for label, trace, err in zip(
                _BRANCH_LABELS,
                values.reshape(4, -1),
                errors.reshape(4, -1),
                strict=True,
            ):
                ax.errorbar(
                    lengths, trace, yerr=err, fmt=".-", markersize=5, label=label
                )
        i_ax.set_ylabel(_I_LABEL)
        q_ax.set_ylabel(_Q_LABEL)
        i_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2)

        ratio_ax.plot(lengths, ratio, ".-")
        ratio_ax.set(xlabel=_XLABEL, ylabel="Contrast ratio r (a.u.)")
        residual_ax.plot(lengths, residual, ".-")
        residual_ax.set(xlabel=_XLABEL, ylabel="Residual with p₀ ≈ 0 (a.u.)")
        residual_ax.axhline(
            selection.epsilon,
            color="gray",
            linestyle=":",
            label=f"Residual limit = {selection.epsilon:.3g} a.u.",
        )
        residual_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20))
        selected = selection.selected_length
        if selected is None:
            residual_ax.set_title("No qualifying duration")
        else:
            residual_ax.set_title(f"Selected duration = {selected:.6g} µs")
            for ax in fig.axes:
                ax.axvline(selected, color="gray", linestyle="--")
