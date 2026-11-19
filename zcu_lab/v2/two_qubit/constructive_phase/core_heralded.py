"""Ground-ground heralded constructive_phase with saved 81-class counts."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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
)
from .core import (
    ConstructivePhaseAnalysis,
    ConstructivePhaseCfg,
    analyze_constructive_phase,
)


@dataclass(frozen=True)
class ConstructivePhaseHeraldedResult(RabiScanResult):
    """F2 relative to F1 phase coordinates, durations in us and (Nc,Nd,3,3,3,3) counts.

    Class order is g/e/other; axes are herald F1/F2 then terminal F1/F2.
    Unmeasured points are NaN. Counts retain all herald outcomes, unnormalized.
    """


class ConstructivePhaseHeraldedCfg(ConstructivePhaseCfg):
    """Relative F2 minus F1 phase scan in degrees at fixed common frequency. Herald precedes B11 preparation.

    The same full mux cfg supplies herald and terminal reads, separated by
    readout.post_delay. reset=None emits no reset. Other inputs match ordinary
    ConstructivePhaseCfg, including the circle classifiers with radius from GE Post.
    """


def restore_result(
    data: GroupedLoadData[ConstructivePhaseHeraldedCfg],
) -> ConstructivePhaseHeraldedResult:
    """Restore 81-class sums; old raw-IQ files without sums raise ValueError.

    data supplies the grouped sums variable and its persisted scan axes. Return
    this experiment's Result with float64 class counts and its original axis
    ordering. Missing variables and invalid data propagate load errors. Do not
    mutate the loaded data, acquire or draw; no raw-IQ reconstruction is attempted.
    """
    sums = data.variable("sums")
    return ConstructivePhaseHeraldedResult(
        coordinates=sums.axes[5], durations=sums.axes[4], sums=sums.z.astype(np.float64)
    )


def _plot_return(
    plots: Plots,
    coordinates: NDArray[np.float64],
    durations: NDArray[np.float64],
    populations: NDArray[np.float64],
    relative_phase_deg: float | None,
) -> None:
    """Plot three terminal fractions at the nearest measured phase in degrees.

    populations contains all nine normalized outcomes per phase and duration.
    None selects the first phase. Annotate the largest remaining outcome and
    its peak fraction. Do not acquire or change the supplied arrays.
    """
    phase_index = 0
    if relative_phase_deg is not None:
        phase_distance = (coordinates - relative_phase_deg + 180) % 360 - 180
        phase_index = int(np.argmin(np.abs(phase_distance)))
    values = populations[phase_index]
    other_indices = [1, 2, 3, 5, 6, 7, 8]
    other_peaks = np.nanmax(values[:, other_indices], axis=0)
    largest_other = other_indices[int(np.argmax(other_peaks))]
    states = ("in g", "in e", "outside g/e")
    _, ax = plots.subplots(num="return", figsize=(11, 6), layout="constrained")
    ax.plot(durations, values[:, 0], ".-", color="C0", label="Both qubits in g")
    ax.plot(durations, values[:, 4], ".-", color="C3", label="Both qubits in e")
    ax.plot(
        durations,
        values[:, other_indices].sum(axis=1),
        ".-",
        color="0.45",
        label="Anything else",
    )
    ax.set(
        xlabel="Drive duration (us)",
        ylabel="Fraction of herald-accepted shots",
        ylim=(0, 1),
        title="Where the two qubits end up after the drive · both in g at herald\n"
        f"(relative phase {coordinates[phase_index]:g}°"
        + (
            "; closest measured to recommendation)"
            if relative_phase_deg is not None
            else "; first measured phase, no recommendation)"
        ),
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=3, frameon=False)
    ax.text(
        0.99,
        0.02,
        f"Largest part of 'anything else': qubit 1 {states[largest_other // 3]}, "
        f"qubit 2 {states[largest_other % 3]} (up to {other_peaks.max():.1%})",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize="x-small",
        color="0.35",
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
                    VariableAxisSpec.generated_arange(label="terminal_f2", unit=""),
                    VariableAxisSpec.generated_arange(label="terminal_f1", unit=""),
                    VariableAxisSpec.generated_arange(label="herald_f2", unit=""),
                    VariableAxisSpec.generated_arange(label="herald_f1", unit=""),
                    VariableAxisSpec(
                        label="Drive duration",
                        unit="s",
                        field_name="durations",
                        scale=1e-6,
                    ),
                    VariableAxisSpec(
                        label="F2 relative to F1 phase",
                        unit="deg",
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
        """Dispatch duration acquisition by the two drive waveform types.

        Constant and flat-top drives use a hardware duration sweep. Sampled
        envelopes rebuild at each host duration. Both return coordinate-major
        float64 counts with the same Result axes and cancellation semantics.

        """
        # 1. Use hardware duration sweeps only when both drives are constant or flat-top.
        if all(
            isinstance(pulse.waveform, (ConstWaveformCfg, FlatTopWaveformCfg))
            for pulse in (config.drive.f1, config.drive.f2)
        ):
            return self._run_hardware_duration(config, context=context)
        # 2. Rebuild sampled drive envelopes on the host while keeping the other axis in hardware.
        return self._run_host_duration(config, context=context)

    def _run_hardware_duration(
        self, config: ConstructivePhaseHeraldedCfg, *, context: RunContext
    ) -> ConstructivePhaseHeraldedResult:
        """Acquire all coordinates and durations in hardware, accumulating rounds."""
        # 1. Pair relative-phase and duration axes for the two-drive return map.
        coordinates = sweep2array(config.relative_phase)
        durations = sweep2array(config.duration)
        # 2. Track gg-heralded terminal P00 versus duration for each relative-phase slice.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Drive duration (us)", ylabel="Terminal P00 | herald (g,g)")
        viewer = live_line(ax, num_lines=len(coordinates))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            ordered = sums
            counts = terminal_counts(ordered, ("g", "g"))
            return viewer.set(durations, joint_probabilities(counts)[..., 0, 0])

        class_shape = (3, 3, 3, 3)
        shape = (len(coordinates), len(durations), *class_shape)
        # 3. Sweep phase/duration in hardware with herald before preparation and terminal readout after contrast.
        buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=config.reps,
                initial_delay=config.initial_delay,
                relax_delay=config.relax_delay,
            )
            coordinate = builder.add_sweep("relative_phase", config.relative_phase)
            length = builder.add_sweep("duration", config.duration)
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
            builder.fork("default", into=["prepare_f1_time", "prepare_f2_time"])
            builder.emit(prepare_f1, timeline="prepare_f1_time")
            builder.emit(prepare_f2, timeline="prepare_f2_time")
            builder.join("prepare_f1_time", "prepare_f2_time", into="default")
            builder.fork("default", into=["drive_f1_time", "drive_f2_time"])
            builder.emit(drive_f1, timeline="drive_f1_time")
            builder.emit(drive_f2, timeline="drive_f2_time")
            builder.join("drive_f1_time", "drive_f2_time", into="default")
            builder.fork("default", into=["contrast_f1_time", "contrast_f2_time"])
            builder.emit(contrast_f1, timeline="contrast_f1_time")
            builder.emit(contrast_f2, timeline="contrast_f2_time")
            builder.join("contrast_f1_time", "contrast_f2_time", into="default")
            builder.emit(readout, tag="terminal")
            prog = builder.build()
            axes = [
                Classify("herald", config.roles.f1_tone),
                Classify("herald", config.roles.f2_tone),
                Classify("terminal", config.roles.f1_tone),
                Classify("terminal", config.roles.f2_tone),
            ]
            sched.run_acquire_summed(
                prog,
                context.soc,
                reducer=shot_sums(prog, axes),
                rounds=config.rounds,
                options=RunOptions(),
            )
        buffer.trigger_update(flush=True)
        sums = buffer.array
        # 4. Retain all 81 class counts, including completed rounds and NaN gaps after cancellation.
        return ConstructivePhaseHeraldedResult(coordinates, durations, sums)

    def _run_host_duration(
        self, config: ConstructivePhaseHeraldedCfg, *, context: RunContext
    ) -> ConstructivePhaseHeraldedResult:
        """Rebuild sampled envelopes per host duration; keep coordinates in hardware."""
        # 1. Keep the relative-phase grid fixed while sampled-envelope durations advance on the host.
        coordinates = sweep2array(config.relative_phase)
        durations = sweep2array(config.duration)
        # 2. Track gg-heralded terminal P00 versus duration for each relative-phase slice.
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Drive duration (us)", ylabel="Terminal P00 | herald (g,g)")
        viewer = live_line(ax, num_lines=len(coordinates))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            ordered = sums.swapaxes(0, 1)
            counts = terminal_counts(ordered, ("g", "g"))
            return viewer.set(durations, joint_probabilities(counts)[..., 0, 0])

        class_shape = (3, 3, 3, 3)
        shape = (len(durations), len(coordinates), *class_shape)
        # 3. Rebuild each duration's herald/prepare/drive/contrast sequence with a hardware relative-phase sweep.
        buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
        # Sampled cosine/Gauss/DRAG envelopes reject QickParam lengths. Only their
        # duration stays on the host; coordinate remains a hardware sweep.
        with Schedule(buffer, scope=context.scope) as sched:
            for duration, step in sched.scan("duration", durations.tolist()):
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=config.reps,
                    initial_delay=config.initial_delay,
                    relax_delay=config.relax_delay,
                )
                coordinate = builder.add_sweep("relative_phase", config.relative_phase)
                length = duration
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
                builder.fork("default", into=["prepare_f1_time", "prepare_f2_time"])
                builder.emit(prepare_f1, timeline="prepare_f1_time")
                builder.emit(prepare_f2, timeline="prepare_f2_time")
                builder.join("prepare_f1_time", "prepare_f2_time", into="default")
                builder.fork("default", into=["drive_f1_time", "drive_f2_time"])
                builder.emit(drive_f1, timeline="drive_f1_time")
                builder.emit(drive_f2, timeline="drive_f2_time")
                builder.join("drive_f1_time", "drive_f2_time", into="default")
                builder.fork("default", into=["contrast_f1_time", "contrast_f2_time"])
                builder.emit(contrast_f1, timeline="contrast_f1_time")
                builder.emit(contrast_f2, timeline="contrast_f2_time")
                builder.join("contrast_f1_time", "contrast_f2_time", into="default")
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
                    reducer=shot_sums(prog, axes),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        buffer.trigger_update(flush=True)
        sums = buffer.array.swapaxes(0, 1)
        # 4. Restore coordinate-major axes without discarding herald classes or partial acquisition counts.
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
        # 1. Select gg-heralded return probabilities; determine completion from all herald classes.
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
        # 2. Show phase-dependent return and leakage before extracting identifiable duration-slice rates.
        fig, ax = plots.subplots(
            num="interference", figsize=(9, 6), layout="constrained"
        )
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
        # 3. Inspect all herald fractions and ee disturbance separately from the gg return fit.
        herald_matrix = herald_probabilities(result.sums)
        fig, _ = plots.subplots(
            num="herald",
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
        fig, ax = plots.subplots(num="ee_control", figsize=(9, 6), layout="constrained")
        image = ax.pcolormesh(
            durations, coordinates, ee_not_returned, shading="nearest", vmin=0, vmax=1
        )
        fig.colorbar(image, ax=ax, label="Terminal not ee | herald (e,e)")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="F2 relative to F1 phase (deg)",
            title="Drive disturbance of |00⟩ · ee control, no shared fit",
        )
        # 4. Fit complete duration traces, then regress identifiable rates squared against relative phase.
        analysis = analyze_constructive_phase(coordinates, durations, answer)
        modulation = analysis.modulation
        _, ax = plots.subplots(num="rabi_rate", figsize=(10, 6), layout="constrained")
        ax.errorbar(
            coordinates,
            [rate.rate if rate.rate is not None else np.nan for rate in analysis.rates],
            yerr=[
                rate.uncertainty if rate.uncertainty is not None else np.nan
                for rate in analysis.rates
            ],
            fmt="o",
            color="C0",
        )
        ax.text(
            0.02,
            0.98,
            "\n".join(
                f"{coordinate:g}: {rate.reason}"
                for coordinate, rate in zip(coordinates, analysis.rates, strict=True)
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
        _, ax = plots.subplots(num="rabi_traces", figsize=(10, 6), layout="constrained")
        for coordinate, rate in zip(coordinates, analysis.rates, strict=True):
            if rate.fitted is not None:
                ax.plot(durations, rate.fitted, label=f"{coordinate:g}")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="Fitted joint P00",
            title="Duration cosine fits · inspect model against measured return curves",
        )
        ax.legend(title="F2 relative to F1 phase (deg)", fontsize="x-small", ncols=2)

        # 5. Summarize terminal outcomes at the measured phase nearest the proposed maximum.
        _plot_return(
            plots,
            coordinates,
            durations,
            answer.populations,
            modulation.prediction.maximum_phase_deg
            if modulation.prediction is not None
            else None,
        )

        # 6. Propose the interference maximum only when the rate-squared modulation is identifiable.
        if modulation.prediction is not None:
            _, ax = plots.subplots(
                num="constructive_phase", figsize=(9, 6), layout="constrained"
            )
            ax.plot(
                analysis.phases,
                analysis.squared_rates,
                "o",
                label="Identifiable duration-slice rates²",
            )
            ax.plot(
                modulation.prediction.phase_degrees,
                modulation.prediction.squared_rates,
                label="A + B cosφ + C sinφ",
            )
            ax.axvline(
                modulation.prediction.maximum_phase_deg,
                ls="--",
                label="Maximum rate phase",
            )
            ax.set(
                xlabel="F2 relative to F1 phase (deg)",
                ylabel="Rabi rate² (MHz²)",
                title="Constructive phase · fixed signed gain ratio",
            )
            ax.legend()
            return ConstructivePhaseAnalysis(
                modulation.prediction.maximum_phase_deg,
                modulation.reason,
                modulation.fit_quality,
            )
        # 7. Keep an unresolved constructive-phase proposal explicit with its fit diagnostics.
        return ConstructivePhaseAnalysis(
            None, modulation.reason, modulation.fit_quality
        )
