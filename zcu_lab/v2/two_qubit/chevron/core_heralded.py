"""Ground-ground heralded chevron with saved 81-class counts."""

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

from .._rabi_analysis import RabiAnalysis, ReturnData, fit_duration_rates
from .._rabi_scan import RabiScanResult
from .._sums import (
    herald_probabilities,
    joint_probabilities,
    terminal_counts,
    validate_classifiers,
)
from .core import ChevronCfg


class ChevronHeraldedCfg(ChevronCfg):
    """Common MHz frequency scan with fixed signed gains and relative phase. Herald precedes B11 preparation.

    The same full mux cfg supplies herald and terminal reads, separated by
    readout.post_delay. reset=None emits no reset. Other inputs match ordinary
    ChevronCfg, including the circle classifiers with radius from GE Post.
    """


@dataclass(frozen=True)
class ChevronHeraldedResult(RabiScanResult):
    """Common drive frequency coordinates, durations in us and (Nc,Nd,3,3,3,3) counts.

    Class order is g/e/other; axes are herald F1/F2 then terminal F1/F2.
    Unmeasured points are NaN. Counts retain all herald outcomes, unnormalized.
    """


def restore_result(
    data: GroupedLoadData[ChevronHeraldedCfg],
) -> ChevronHeraldedResult:
    """Restore 81-class sums; old raw-IQ files without sums raise ValueError."""
    sums = data.variable("sums")
    return ChevronHeraldedResult(
        coordinates=sums.axes[5], durations=sums.axes[4], sums=sums.z.astype(np.float64)
    )


class ChevronHeraldedExp(Experiment[ChevronHeraldedResult, ChevronHeraldedCfg]):
    """Condition chevron analysis on herald classified (g,g)."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[ChevronHeraldedResult, ChevronHeraldedCfg]] = (
        GroupedAxesSpec(
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
                            label="Common drive frequency",
                            unit="Hz",
                            field_name="coordinates",
                            scale=1e6,
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
            result_type=ChevronHeraldedResult,
            cfg_type=ChevronHeraldedCfg,
            tag="two_qubit/chevron_heralded",
            result_builder=restore_result,
        )
    )

    def run(
        self, config: ChevronHeraldedCfg, *, context: RunContext
    ) -> ChevronHeraldedResult:
        """Dispatch duration acquisition by the two drive waveform types.

        Constant and flat-top drives use a hardware duration sweep. Sampled
        envelopes rebuild at each host duration. Both return coordinate-major
        float64 counts with the same Result axes and cancellation semantics.
        """
        if all(
            isinstance(pulse.waveform, (ConstWaveformCfg, FlatTopWaveformCfg))
            for pulse in (config.drive.f1, config.drive.f2)
        ):
            return self._run_hardware_duration(config, context=context)
        return self._run_host_duration(config, context=context)

    def _run_hardware_duration(
        self, config: ChevronHeraldedCfg, *, context: RunContext
    ) -> ChevronHeraldedResult:
        """Acquire all coordinates and durations in hardware, accumulating rounds."""
        validate_classifiers(config.readout, config.roles)
        coordinates = sweep2array(config.frequency)
        durations = sweep2array(config.duration)
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
        buffer = SignalBuffer(shape, dtype=np.float64, on_update=update_view)
        with Schedule(buffer, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=config.reps,
                initial_delay=config.initial_delay,
                relax_delay=config.relax_delay,
            )
            coordinate = builder.add_sweep("frequency", config.frequency)
            length = builder.add_sweep("duration", config.duration)
            drive_f1 = Pulse(
                "drive_f1", config.drive.f1, length=length, freq=coordinate
            )
            drive_f2 = Pulse(
                "drive_f2", config.drive.f2, length=length, freq=coordinate
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
                processor=shot_sums(prog, axes),
                rounds=config.rounds,
                options=RunOptions(),
            )
        buffer.trigger_update(flush=True)
        sums = buffer.array
        return ChevronHeraldedResult(coordinates, durations, sums)

    def _run_host_duration(
        self, config: ChevronHeraldedCfg, *, context: RunContext
    ) -> ChevronHeraldedResult:
        """Rebuild sampled envelopes per host duration; keep coordinates in hardware."""
        validate_classifiers(config.readout, config.roles)
        coordinates = sweep2array(config.frequency)
        durations = sweep2array(config.duration)
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
                coordinate = builder.add_sweep("frequency", config.frequency)
                length = duration
                drive_f1 = Pulse(
                    "drive_f1", config.drive.f1, length=length, freq=coordinate
                )
                drive_f2 = Pulse(
                    "drive_f2", config.drive.f2, length=length, freq=coordinate
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
                    processor=shot_sums(prog, axes),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        buffer.trigger_update(flush=True)
        sums = buffer.array.swapaxes(0, 1)
        return ChevronHeraldedResult(coordinates, durations, sums)

    def analyze(
        self,
        source: RunRecord[ChevronHeraldedCfg, ChevronHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> RabiAnalysis:
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
        coordinates = sweep2array(cfg.frequency)
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
        acquired = result.sums.sum(axis=(-4, -3, -2, -1))
        answer = ReturnData(
            joint_probabilities(counts).reshape(len(coordinates), len(durations), 9),
            (acquired == cfg.reps * cfg.rounds) & (totals > 0),
            totals,
            herald_probabilities(result.sums),
        )
        fig, ax = plots.subplots(num="chevron", figsize=(9, 6), layout="constrained")
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
            ylabel="Common drive frequency (MHz)",
            title="Nominal B11 return proxy · not gate fidelity",
        )
        labels = tuple(
            f"({a},{b})" for a in ("g", "e", "other") for b in ("g", "e", "other")
        )
        fig, _ = plots.subplots(
            num="return",
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
            ax.legend(title="Common drive frequency (MHz)", fontsize="x-small", ncols=2)
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
                ylabel="Common drive frequency (MHz)",
                title=f"Herald {labels[outcome]}"
                + (" · pass fraction, not purity" if outcome == 0 else ""),
            )

        ee_counts = terminal_counts(result.sums, ("e", "e"))
        ee_not_returned = 1.0 - joint_probabilities(ee_counts)[..., 1, 1]
        fig, ax = plots.subplots(num="ee_control", figsize=(9, 6), layout="constrained")
        image = ax.pcolormesh(
            durations, coordinates, ee_not_returned, shading="nearest", vmin=0, vmax=1
        )
        fig.colorbar(image, ax=ax, label="Terminal not ee | herald (e,e)")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="Common drive frequency (MHz)",
            title="Drive disturbance of |00⟩ · ee control, no shared fit",
        )
        rates = fit_duration_rates(durations, answer)
        _, ax = plots.subplots(num="rabi_rate", figsize=(10, 6), layout="constrained")
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
            xlabel="Common drive frequency (MHz)",
            ylabel="Rabi rate (MHz)",
            title="Undamped cosine model · rates below 1/duration span excluded",
        )
        _, ax = plots.subplots(num="rabi_traces", figsize=(10, 6), layout="constrained")
        for coordinate, rate in zip(coordinates, rates, strict=True):
            if rate.fitted is not None:
                ax.plot(durations, rate.fitted, label=f"{coordinate:g}")
        ax.set(
            xlabel="Drive duration (us)",
            ylabel="Fitted joint P00",
            title="Duration cosine fits · inspect model against measured return curves",
        )
        ax.legend(title="Common drive frequency (MHz)", fontsize="x-small", ncols=2)

        return RabiAnalysis(answer, rates)
