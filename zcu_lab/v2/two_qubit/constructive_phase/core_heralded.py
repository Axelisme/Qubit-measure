"""Ground-ground heralded constructive_phase with saved 81-class counts."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
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
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Classify, Pulse, TimelineBuilder, shot_sums
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    Join,
    MuxReadout,
    Reset,
)

from zcu_lab.v2.support import live_line

from .._rabi_analysis import ReturnData
from .._rabi_scan import RabiScanResult
from .._sums import (
    herald_probabilities,
    joint_probabilities,
    terminal_counts,
    validate_classifiers,
)
from .analysis import ConstructivePhaseAnalysis, analyze_constructive_phase
from .core import ConstructivePhaseCfg


class ConstructivePhaseHeraldedCfg(ConstructivePhaseCfg):
    """Relative F2 minus F1 phase scan in degrees at fixed common frequency. Herald precedes B11 preparation.

    The same full mux cfg supplies herald and terminal reads, separated by
    readout.post_delay. reset=None emits no reset. Other inputs match ordinary
    ConstructivePhaseCfg, including the circle classifiers with radius from GE Post.
    """


@dataclass(frozen=True)
class ConstructivePhaseHeraldedResult(RabiScanResult):
    """F2 relative to F1 phase coordinates, durations in us and (Nc,Nd,3,3,3,3) counts.

    Class order is g/e/other; axes are herald F1/F2 then terminal F1/F2.
    Unmeasured points are NaN. Counts retain all herald outcomes, unnormalized.
    """


def restore_result(
    data: GroupedLoadData[ConstructivePhaseHeraldedCfg],
) -> ConstructivePhaseHeraldedResult:
    """Restore 81-class sums; old raw-IQ files without sums raise ValueError."""
    sums = data.variable("sums")
    return ConstructivePhaseHeraldedResult(
        coordinates=sums.axes[5], durations=sums.axes[4], sums=sums.z.astype(np.float64)
    )


class ConstructivePhaseHeraldedExp(
    Experiment[ConstructivePhaseHeraldedResult, ConstructivePhaseHeraldedCfg]
):
    """Condition constructive_phase analysis on herald classified (g,g)."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[ConstructivePhaseHeraldedResult, ConstructivePhaseHeraldedCfg]
    ] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="sums",
                axes=(
                    VariableAxisSpec.generated_arange("terminal_f2", ""),
                    VariableAxisSpec.generated_arange("terminal_f1", ""),
                    VariableAxisSpec.generated_arange("herald_f2", ""),
                    VariableAxisSpec.generated_arange("herald_f1", ""),
                    VariableAxisSpec(
                        "Drive duration", "s", field_name="durations", scale=1e-6
                    ),
                    VariableAxisSpec(
                        "F2 relative to F1 phase",
                        "deg",
                        field_name="coordinates",
                        scale=1.0,
                    ),
                ),
                z=VariableZSpec(
                    field_name="sums",
                    label="Herald and terminal class counts",
                    unit="",
                    dtype=np.float64,
                ),
            ),
        ),
        result_type=ConstructivePhaseHeraldedResult,
        cfg_type=ConstructivePhaseHeraldedCfg,
        tag="two_qubit/constructive_phase_heralded",
        result_builder=restore_result,
    )

    def run(
        self, config: ConstructivePhaseHeraldedCfg, *, context: RunContext
    ) -> ConstructivePhaseHeraldedResult:
        """Acquire optional reset → herald → B11 → drive → double Xπ → terminal.

        Both reads emit one mux object with separate tags. Return all 81-class
        counts, including completed rounds on cooperative cancellation.
        Acquisition exceptions propagate unchanged. The stop signal carries only
        stop requests and reason text.
        """
        validate_classifiers(config.readout, config.roles)
        coordinates = sweep2array(config.relative_phase)
        durations = sweep2array(config.duration)
        hardware_duration = all(
            isinstance(pulse.waveform, (ConstWaveformCfg, FlatTopWaveformCfg))
            for pulse in (config.drive.f1, config.drive.f2)
        )
        viewer = live_line(
            context.plots,
            "measurement",
            "Drive duration (us)",
            "Terminal P00 | herald (g,g)",
            num_lines=len(coordinates),
        )

        def update_view(sums: NDArray[np.float64]) -> None:
            ordered = sums[0] if hardware_duration else sums.swapaxes(0, 1)
            counts = terminal_counts(ordered, ("g", "g"))
            viewer.update(durations, joint_probabilities(counts)[..., 0, 0])

        class_shape = (3, 3, 3, 3)
        shape = (
            (1, len(coordinates), len(durations), *class_shape)
            if hardware_duration
            else (len(durations), len(coordinates), *class_shape)
        )
        buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
        # Sampled cosine/Gauss/DRAG envelopes reject QickParam lengths. Only their
        # duration stays on the host; coordinate remains a hardware sweep.
        batches = [None] if hardware_duration else durations.tolist()
        with Schedule(config, buffer, scope=context.scope) as sched:
            for duration, step in sched.scan("duration_batch", batches):
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=config.reps,
                    initial_delay=config.initial_delay,
                    relax_delay=config.relax_delay,
                )
                coordinate = builder.add_sweep("relative_phase", config.relative_phase)
                length = (
                    builder.add_sweep("duration", config.duration)
                    if hardware_duration
                    else duration
                )
                drive_f1 = Pulse("drive_f1", config.drive.f1, length=length)
                drive_f2 = Pulse(
                    "drive_f2",
                    config.drive.f2,
                    length=length,
                    phase=float(config.drive.f1.phase) + coordinate,
                )
                prepare_f1 = Pulse("prepare_f1", config.x180_f1)
                prepare_f2 = Pulse("prepare_f2", config.x180_f2)
                contrast_f1 = Pulse("contrast_f1", config.x180_f1)
                contrast_f2 = Pulse("contrast_f2", config.x180_f2)
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset))
                builder.emit(readout, tag="herald")
                builder.emit(Join(prepare_f1, prepare_f2))
                builder.emit(Join(drive_f1, drive_f2))
                builder.emit(Join(contrast_f1, contrast_f2))
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                axes = [
                    Classify("herald", config.roles.f1_tone),
                    Classify("herald", config.roles.f2_tone),
                    Classify("terminal", config.roles.f1_tone),
                    Classify("terminal", config.roles.f2_tone),
                ]
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    processor=shot_sums(prog, axes),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        buffer.trigger_update(flush=True)
        sums = buffer.array[0] if hardware_duration else buffer.array.swapaxes(0, 1)
        return ConstructivePhaseHeraldedResult(coordinates, durations, sums)

    def analyze(
        self,
        source: RunRecord[
            ConstructivePhaseHeraldedCfg, ConstructivePhaseHeraldedResult
        ],
        options: None,
        *,
        plots: Plots,
    ) -> ConstructivePhaseAnalysis:
        """Analyze the herald (g,g) slice and plot all nine herald fractions.

        Pass fraction is not ground-state purity. Empty accepted counts give
        NaN terminal probabilities and unresolved fits. Saved cfg is required.
        """
        del options
        if source.cfg is None:
            raise ValueError(
                "Rabi analysis requires saved cfg; current Form/library is not a substitute"
            )
        cfg, result = source.cfg, source.result
        coordinates = sweep2array(cfg.relative_phase)
        durations = sweep2array(cfg.duration)
        if (
            not np.array_equal(result.coordinates, coordinates)
            or not np.array_equal(result.durations, durations)
            or result.sums.shape != (len(coordinates), len(durations), 3, 3, 3, 3)
        ):
            raise ValueError(
                "Rabi count axes/shape must match saved cfg and class axes"
            )
        counts = terminal_counts(result.sums, ("g", "g"))
        totals = counts.sum(axis=(-2, -1))
        answer = ReturnData(
            joint_probabilities(counts).reshape(len(coordinates), len(durations), 9),
            (result.sums.sum(axis=(-4, -3, -2, -1)) == cfg.reps * cfg.rounds)
            & (totals > 0),
            totals,
            herald_probabilities(result.sums),
        )
        fig, ax = plots.subplots("interference", figsize=(9, 6), layout="constrained")
        image = ax.pcolormesh(
            result.durations,
            result.coordinates,
            answer.populations[..., 0],
            shading="nearest",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(image, ax=ax, label="Observed joint P00 after double Xπ")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="F2 relative to F1 phase (deg)",
            title="Nominal B11 return proxy · not gate fidelity",
        )
        labels = tuple(
            f"({a},{b})" for a in ("g", "e", "other") for b in ("g", "e", "other")
        )
        fig, _ = plots.subplots(
            "return",
            nrows=3,
            ncols=3,
            figsize=(13, 10),
            squeeze=False,
            layout="constrained",
        )
        for outcome, ax in enumerate(fig.axes):
            for coordinate, values in zip(
                result.coordinates, answer.populations, strict=True
            ):
                ax.plot(
                    result.durations, values[:, outcome], ".-", label=f"{coordinate:g}"
                )
            ax.set(
                xlabel="Drive duration (us)",
                ylabel=f"Observed P{labels[outcome]}",
                title=f"Terminal {labels[outcome]} · other retained in denominator",
            )
            ax.legend(
                title="F2 relative to F1 phase (deg)", fontsize="x-small", ncols=2
            )
        herald_matrix = herald_probabilities(result.sums)
        fig, _ = plots.subplots(
            "herald",
            nrows=3,
            ncols=3,
            figsize=(13, 10),
            squeeze=False,
            layout="constrained",
        )
        for outcome, ax in enumerate(fig.axes):
            image = ax.pcolormesh(
                result.durations,
                result.coordinates,
                herald_matrix[..., outcome // 3, outcome % 3],
                shading="nearest",
                vmin=0,
                vmax=1,
            )
            fig.colorbar(image, ax=ax, label="Herald probability")
            ax.set(
                xlabel="Drive duration (us)",
                ylabel="F2 relative to F1 phase (deg)",
                title=f"Herald {labels[outcome]}"
                + (" · pass fraction, not purity" if outcome == 0 else ""),
            )

        ee_not_returned = (
            1.0
            - joint_probabilities(terminal_counts(result.sums, ("e", "e")))[..., 1, 1]
        )
        fig, ax = plots.subplots("ee_control", figsize=(9, 6), layout="constrained")
        image = ax.pcolormesh(
            durations, coordinates, ee_not_returned, shading="nearest", vmin=0, vmax=1
        )
        fig.colorbar(image, ax=ax, label="Terminal not ee | herald (e,e)")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="F2 relative to F1 phase (deg)",
            title="Drive disturbance of |00⟩ · ee control, no shared fit",
        )
        analysis = analyze_constructive_phase(coordinates, durations, answer)
        rates = analysis.rates
        _, ax = plots.subplots("rabi_rate", figsize=(10, 6), layout="constrained")
        ax.errorbar(
            coordinates,
            [rate.rate if rate.rate is not None else np.nan for rate in rates],
            yerr=[
                rate.uncertainty if rate.uncertainty is not None else np.nan
                for rate in rates
            ],
            fmt="o",
            color="C0",
        )
        ax.text(
            0.02,
            0.98,
            "\n".join(
                f"{coordinate:g}: {rate.reason}"
                for coordinate, rate in zip(coordinates, rates, strict=True)
                if rate.reason
            ),
            va="top",
            transform=ax.transAxes,
            fontsize="small",
        )
        ax.set(
            xlabel="F2 relative to F1 phase (deg)",
            ylabel="Rabi rate (MHz)",
            title="Undamped cosine model · rates below 1/duration span excluded",
        )
        _, ax = plots.subplots("rabi_traces", figsize=(10, 6), layout="constrained")
        for coordinate, rate in zip(coordinates, rates, strict=True):
            if rate.fitted is not None:
                ax.plot(durations, rate.fitted, label=f"{coordinate:g}")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="Fitted joint P00",
            title="Duration cosine fits · inspect model against measured return curves",
        )
        ax.legend(title="F2 relative to F1 phase (deg)", fontsize="x-small", ncols=2)

        if analysis.interference is not None:
            phases, squared, dense, curve, maximum_phase = analysis.interference
            _, ax = plots.subplots(
                "constructive_phase", figsize=(9, 6), layout="constrained"
            )
            ax.plot(phases, squared, "o", label="Identifiable duration-slice rates²")
            ax.plot(dense, curve, label="A + B cosφ + C sinφ")
            ax.axvline(maximum_phase, ls="--", label="Maximum rate phase")
            ax.set(
                xlabel="F2 relative to F1 phase (deg)",
                ylabel="Rabi rate² (MHz²)",
                title="Constructive phase · fixed signed gain ratio",
            )
            ax.legend()
        return analysis
