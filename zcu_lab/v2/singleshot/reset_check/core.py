from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass as pydantic_dataclass
from zcu_tools.analysis.singleshot import correct_populations
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    GainSweep,
    Pulse,
    PulseReadout,
    Readout,
    Reset,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules.readout import require_circle_classifiers

from zcu_lab.v2.support import live_line

BRANCH_LABELS = ("Before reset", "Reset only", "Reset + Rabi")
BRANCH_LINESTYLES = ("-", "--", ":")


class ResetCheckCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    rabi_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Rabi Pulse")]
    tested_reset: Annotated[Reset.Cfg, FieldMeta(label="Tested Reset")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep, FieldMeta(label="Gain", unit="a.u.")]
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0


@dataclass(frozen=True)
class ResetCheckResult:
    """G/E/Other shot counts in (gain, reset_state, population_state) order.

    Branches are independent preparations, not paired shot trajectories.
    """

    gains: NDArray[np.float64]
    counts: NDArray[np.float64]


@dataclass(frozen=True)
class ResetCheckAnalysis:
    """Scan-independent reset-only population diagnostics across sampled gains."""

    reset_mean_excited_population: float
    reset_max_excited_population: float
    reset_max_other_population: float
    worst_sample_gain: float
    analyzed_reset_points: int


def _population_line_kwargs() -> list[dict[str, str]]:
    return [
        {"label": f"{label}: {state}", "color": color, "linestyle": style}
        for label, style in zip(BRANCH_LABELS, BRANCH_LINESTYLES, strict=True)
        for state, color in (("Ground", "blue"), ("Excited", "red"), ("Other", "green"))
    ]


@pydantic_dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class ResetCheckAnalyzeOptions:
    confusion_matrix: NDArray[np.float64] | None = None


class ResetCheckExp(Experiment[ResetCheckResult, ResetCheckCfg]):
    Options: ClassVar[type[ResetCheckAnalyzeOptions]] = ResetCheckAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[ResetCheckResult, ResetCheckCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis(
                values=(0, 1, 2), label="Population", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis(
                values=(0, 1, 2), label="Reset State", unit="None", dtype=np.int64
            ),
            AxesSpec.Axis("gains", label="Gain", unit="a.u.", dtype=np.float64),
        ),
        z=AxesSpec.Z("counts", label="Class counts", unit="count", dtype=np.float64),
        data_variable=DataVariable("counts"),
        result_type=ResetCheckResult,
        cfg_type=ResetCheckCfg,
        cfg_schema_version="4.0",
        tag="singleshot/reset_check",
    )

    def run(
        self,
        cfg: ResetCheckCfg,
        *,
        context: RunContext,
    ) -> ResetCheckResult:
        """Compare classified populations before and after the tested reset.

        Each independent branch starts with optional reset and a gain-swept
        Rabi pulse, then applies no tested reset, tested reset alone, or tested
        reset followed by the same Rabi pulse, before readout. Return quantized
        gains and summed G/E/Other shot counts in (gain, branch, population) order.
        These populations are not paired trajectories or reset-channel fidelity.
        Invalid classification centers or radius raise ValueError.
        """
        # 1. Validate classification and prepare hardware-quantized gain points.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(cfg)
        require_circle_classifiers(cfg.readout)
        axis = Classify("readout")
        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.rabi_pulse.ch},
        )

        # 2. Show G/E/Other for all three independent preparations.
        def configure_axes(ax: Axes) -> None:
            ax.set_ylim(-0.02, 1.02)
            for line, kwargs in zip(ax.lines, _population_line_kwargs(), strict=True):
                line.set(**kwargs)
            ax.legend()

        figure, ax = context.plots.subplots(num="measurement")
        ax.set(xlabel="Pulse gain (a.u.)", ylabel="Classified population")
        viewer = live_line(ax, num_lines=9)
        configure_axes(ax)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.float64]):
            return viewer.set(
                gains,
                (data / data.sum(axis=-1, keepdims=True)).reshape(gains.size, 9).T,
            )

        figure.show()
        buffer = SignalBuffer(
            (gains.size, 3, 3),
            dtype=np.float64,
            on_update=update,
        )
        with Schedule(buffer, scope=context.scope) as sched:
            # 3. Sweep gain outside the three-way hardware branch loop.
            builder = TimelineBuilder(
                soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
            )
            gain = builder.add_sweep("gain", cfg.gain)
            rabi_pulse = Pulse("rabi_pulse", cfg.rabi_pulse, gain=gain)
            tested_reset = Reset("tested_reset", cfg.tested_reset)
            builder.add_loop("reset_sel", 3)
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(rabi_pulse)
            builder.delay_by_timeline()
            with builder.branch(by="reset_sel", align_timeline=False) as cases:
                with cases.case(1):
                    builder.emit(tested_reset)
                with cases.case(2):
                    builder.emit(tested_reset)
                    builder.emit(rabi_pulse)
            builder.delay_by_timeline()
            builder.emit(Readout("readout", cfg.readout))

            # 4. Retain cumulative class counts, including completed partial rounds.
            program = builder.build()
            sched.run_acquire_summed(
                program,
                soc,
                processor=shot_sums(program, [axis]),
                rounds=cfg.rounds,
            )
            sched.trigger_update(flush=True)
        return ResetCheckResult(
            gains=gains,
            counts=buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[ResetCheckCfg, ResetCheckResult],
        options: ResetCheckAnalyzeOptions,
        *,
        plots: Plots,
    ) -> ResetCheckAnalysis:
        """Summarize reset-only excited and Other populations across sampled gains.

        Apply options.confusion_matrix when supplied, otherwise use classified
        fractions normalized per row from source.result.counts. Return reset-only
        mean/worst sampled populations, and plot all three corrected G/E/Other branches.
        Invalid axes or no finite reset-only samples raise ValueError.
        """
        # 1. Check population axes before optional readout-confusion correction.
        result = source.result
        confusion_matrix = options.confusion_matrix
        if result.counts.shape != (result.gains.size, 3, 3):
            raise ValueError(
                "Reset check requires (gain, branch, G/E/Other) populations"
            )
        populations = result.counts / result.counts.sum(axis=-1, keepdims=True)
        populations = correct_populations(populations, confusion_matrix)

        # 2. Summarize finite reset-only samples, not reset-channel fidelity.
        reset_e = populations[:, 1, 1]
        valid = np.isfinite(result.gains) & np.isfinite(populations[:, 1]).all(axis=-1)
        if not valid.any():
            raise ValueError("No finite reset-only populations to analyze")
        worst = np.flatnonzero(valid)[np.argmax(reset_e[valid])]
        analysis = ResetCheckAnalysis(
            float(np.mean(reset_e[valid])),
            float(reset_e[worst]),
            float(np.max(populations[valid, 1, 2])),
            float(result.gains[worst]),
            int(valid.sum()),
        )

        # 3. Compare the independent preparations on the same gain axis.
        _, ax = plots.subplots(num="populations", figsize=(10, 6))
        order = np.argsort(result.gains)
        for values, kwargs in zip(
            populations[order].reshape(result.gains.size, 9).T,
            _population_line_kwargs(),
            strict=True,
        ):
            ax.plot(
                result.gains[order],
                values,
                marker=".",
                label=kwargs["label"],
                color=kwargs["color"],
                linestyle=kwargs["linestyle"],
            )
        ax.set(
            xlabel="Pulse gain (a.u.)",
            ylabel="Population"
            if confusion_matrix is not None
            else "Classified population",
            ylim=(-0.02, 1.02),
            title=f"Reset excited: mean {analysis.reset_mean_excited_population:.2%}, worst sampled {analysis.reset_max_excited_population:.2%}\nOther is not calibrated leakage; populations are not reset-channel fidelity",
        )
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True)
        return analysis
