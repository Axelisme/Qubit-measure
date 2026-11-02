"""Common CZ amplitude calibration conditioned on a ground-ground herald."""

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
from zcu_tools.program.v2 import Classify, Pulse, Reset, TimelineBuilder, shot_sums
from zcu_tools.program.v2.modules import MuxReadout

from zcu_lab.v2.support import live_line

from .._sums import (
    herald_probabilities,
    joint_probabilities,
    terminal_counts,
    validate_classifiers,
)
from .analysis import (
    OUTCOME_LABELS,
    CZAmplitudeAnalysis,
    analyze_return,
    summarize_return,
)
from .core import CZAmplitudeCfg


class CZAmplitudeHeraldedCfg(CZAmplitudeCfg):
    """Optional reset → herald → B11 → CZ → Xπ⊗Xπ → terminal mux.

    Inherited readout supplies both reads and their g/e/other circle radii.
    Its post_delay is the wait after herald. reset=None emits no reset.
    All other fields and units match CZAmplitudeCfg.
    """


@dataclass(frozen=True)
class CZAmplitudeHeraldedResult:
    """scales is the gain multiplier; sums is float64 (Ns,3,3,3,3) raw counts.

    Class axes are herald F1/F2 then terminal F1/F2, each ordered g/e/other.
    Rounds add without averaging; all herald classes remain stored.
    Unmeasured scales remain NaN; completed rounds survive cancellation.
    """

    scales: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class CZAmplitudeHeraldedAnalysis:
    """gg first-return fit summary; herald distributions remain plot-local."""

    terminal: CZAmplitudeAnalysis


def restore_result(
    data: GroupedLoadData[CZAmplitudeHeraldedCfg],
) -> CZAmplitudeHeraldedResult:
    """Restore 81-class sums; old per-shot IQ files without sums raise ValueError."""
    sums = data.variable("sums")
    return CZAmplitudeHeraldedResult(sums.axes[4], sums.z.astype(np.float64))


class CZAmplitudeHeraldedExp(
    Experiment[CZAmplitudeHeraldedResult, CZAmplitudeHeraldedCfg]
):
    """Scan common gain scales with ground-ground herald post-selection."""

    AXES_SPEC: ClassVar[
        GroupedAxesSpec[CZAmplitudeHeraldedResult, CZAmplitudeHeraldedCfg]
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
                        label="Common gain scale", unit="", field_name="scales"
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
        result_type=CZAmplitudeHeraldedResult,
        cfg_type=CZAmplitudeHeraldedCfg,
        tag="two_qubit/cz_amplitude_heralded",
        result_builder=restore_result,
    )

    def run(
        self, config: CZAmplitudeHeraldedCfg, *, context: RunContext
    ) -> CZAmplitudeHeraldedResult:
        """Acquire unchanged raw sums for all 81 herald/terminal outcomes.

        No host round loop or per-shot IQ is stored. Completed rounds survive
        cooperative cancellation. Acquisition exceptions propagate unchanged.
        The stop signal carries only stop requests and reason text.
        """
        validate_classifiers(config.readout, config.roles)
        scales = sweep2array(config.scales)
        fig, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Common gain scale", ylabel="P(g,g) | herald (g,g)")
        viewer = live_line(ax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(False)
        fig.show()

        @context.plots.on_plot_thread
        def update_view(sums: NDArray[np.float64]) -> tuple[Line2D, ...]:
            counts = terminal_counts(sums, ("g", "g"))
            return viewer.set(scales, joint_probabilities(counts)[..., 0, 0])

        buffer = SignalBuffer(
            (len(scales), 3, 3, 3, 3), dtype=np.float64, on_update=update_view
        )
        builder = TimelineBuilder(
            context.soccfg,
            reps=config.reps,
            initial_delay=config.initial_delay,
            relax_delay=config.relax_delay,
        )
        scale = builder.add_sweep("scale", config.scales)
        drive_f1 = Pulse(
            "cz_f1", config.drive.f1, gain=float(config.drive.f1.gain) * scale
        )
        drive_f2 = Pulse(
            "cz_f2", config.drive.f2, gain=float(config.drive.f2.gain) * scale
        )
        readout = MuxReadout("readout", config.readout)
        if config.reset is not None:
            builder.emit(Reset("reset", config.reset))
        builder.emit(readout, tag="herald")
        builder.fork("default", into=["prepare_f1_time", "prepare_f2_time"])
        builder.emit(Pulse("prepare_f1", config.x180_f1), timeline="prepare_f1_time")
        builder.emit(Pulse("prepare_f2", config.x180_f2), timeline="prepare_f2_time")
        builder.join("prepare_f1_time", "prepare_f2_time", into="default")
        builder.fork("default", into=["drive_f1_time", "drive_f2_time"])
        builder.emit(drive_f1, timeline="drive_f1_time")
        builder.emit(drive_f2, timeline="drive_f2_time")
        builder.join("drive_f1_time", "drive_f2_time", into="default")
        builder.fork("default", into=["contrast_f1_time", "contrast_f2_time"])
        builder.emit(Pulse("contrast_f1", config.x180_f1), timeline="contrast_f1_time")
        builder.emit(Pulse("contrast_f2", config.x180_f2), timeline="contrast_f2_time")
        builder.join("contrast_f1_time", "contrast_f2_time", into="default")
        builder.emit(readout, tag="terminal")
        prog = builder.build()
        axes = [
            Classify("herald", config.roles.f1_tone),
            Classify("herald", config.roles.f2_tone),
            Classify("terminal", config.roles.f1_tone),
            Classify("terminal", config.roles.f2_tone),
        ]
        with Schedule(buffer, scope=context.scope) as sched:
            sched.run_acquire_summed(
                prog,
                context.soc,
                processor=shot_sums(prog, axes),
                rounds=config.rounds,
                options=RunOptions(),
            )
        buffer.trigger_update(flush=True)
        return CZAmplitudeHeraldedResult(scales, buffer.array)

    def analyze(
        self,
        source: RunRecord[CZAmplitudeHeraldedCfg, CZAmplitudeHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> CZAmplitudeHeraldedAnalysis:
        """Fit accepted terminal P00 and plot all nine herald probabilities.

        Fit and coverage policy match the ordinary version, using all herald
        classes to determine acquisition completeness. No accepted shots gives
        NaN terminal probabilities and no proposal. Missing cfg raises ValueError.
        """
        del options
        cfg, result = source.cfg, source.result
        if cfg is None:
            raise ValueError("CZ return analysis requires saved cfg")
        if not np.array_equal(result.scales, sweep2array(cfg.scales)):
            raise ValueError("CZ return scales must match saved cfg")
        if result.sums.shape != (len(result.scales), 3, 3, 3, 3):
            raise ValueError("Heralded CZ return requires sums (Ns,3,3,3,3)")
        terminal = analyze_return(
            result.scales,
            terminal_counts(result.sums, ("g", "g")),
            acquired_shots=result.sums.sum(axis=(-4, -3, -2, -1)),
            expected_shots=cfg.reps * cfg.rounds,
        )
        _, ax = plots.subplots(
            num="joint_return", figsize=(10, 5), layout="constrained"
        )
        for index, label in enumerate(OUTCOME_LABELS):
            ax.plot(
                result.scales,
                terminal.data.probabilities.reshape(-1, 9)[:, index],
                ".-",
                label=label,
            )
        ax.set(
            xlabel="Common gain scale",
            ylabel="Same-shot joint probability",
            title="B11 → CZ → Xπ⊗Xπ · classified return, not gate fidelity",
        )
        if np.any(np.isfinite(terminal.fit.curve)):
            ax.plot(
                result.scales, terminal.fit.curve, "k--", label="Zero-phase P00 fit"
            )
        diagnostic = (
            terminal.fit.reason
            or f"First positive return scale = {terminal.fit.scale:.6g} ± {terminal.fit.error:.2g}"
        )
        ax.text(
            0.02,
            0.02,
            diagnostic,
            transform=ax.transAxes,
            fontsize="small",
            bbox={"facecolor": "white", "alpha": 0.85},
        )
        ax.legend()

        ee_counts = terminal_counts(result.sums, ("e", "e"))
        ee_not_returned = 1.0 - joint_probabilities(ee_counts)[:, 1, 1]
        _, ax = plots.subplots(num="ee_control", figsize=(10, 5), layout="constrained")
        ax.plot(result.scales, ee_not_returned, ".-")
        ax.set(
            xlabel="Common gain scale",
            ylabel="Terminal not ee | herald (e,e)",
            title="Drive disturbance of |00⟩ · ee control, no shared fit",
        )
        herald_matrix = herald_probabilities(result.sums)
        _, ax = plots.subplots(num="herald", figsize=(10, 5), layout="constrained")
        for index, label in enumerate(OUTCOME_LABELS):
            ax.plot(
                result.scales, herald_matrix.reshape(-1, 9)[:, index], ".-", label=label
            )
        ax.set(
            xlabel="Common gain scale",
            ylabel="Herald probability",
            title="Herald (g,g) is pass fraction, not ground-state purity",
        )
        ax.legend()
        return CZAmplitudeHeraldedAnalysis(summarize_return(terminal.fit))
