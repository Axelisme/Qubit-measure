"""Paired reference/interleaved RB with explicit rounds and seed uncertainty."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Self

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.experiment import IDENTITY, AxesSpec, Axis, PersistableExperiment, ZSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2

from ..rb.core import RBCfg
from ..rb.program import build_rb_program
from ..rb.sequence import TargetGate, make_seed_tables
from .analysis import IRBAnalysis, analyze_irb, project_arm_means


class IRBCfg(RBCfg):
    """Paired RB settings; depth counts random Cliffords, rounds are host passes."""

    target_gate: TargetGate = "X90"
    """Physical Clifford inserted after each random Clifford, in its current frame."""
    n_seeds: int = Field(ge=2)
    """Number of independent random sequences shared by both arms."""

    @model_validator(mode="after")
    def validate_depths(self) -> Self:
        """Reject fractional, negative, repeated or empty depths before device I/O."""
        depths = sweep2array(self.sweep.depth, allow_array=True)
        if (
            len(depths) == 0
            or not np.isfinite(depths).all()
            or np.any(depths < 0)
            or np.any(depths != np.floor(depths))
            or len(np.unique(depths)) != len(depths)
        ):
            raise ValueError("IRB requires unique nonnegative integer depths")
        return self


@dataclass(frozen=True)
class IRBResult:
    """Fixed raw layout, including NaNs for unfinished acquisitions."""

    rounds: NDArray[np.int64]
    """Zero-based host pass IDs."""
    sub_seeds: NDArray[np.int64]
    """Independent RNG seeds, paired across arms and reused across rounds."""
    arms: NDArray[np.int64]
    """Fixed order [0, 1]: reference, interleaved."""
    depths: NDArray[np.int64]
    """Number of random Cliffords; interleaved also inserts this many targets."""
    signals: NDArray[np.complex128]
    """Repetition-averaged IQ, shape (round, seed, arm, depth)."""


def _label_arms(ax: Axes) -> None:
    for line, label in zip(ax.lines, ("Reference", "Interleaved"), strict=True):
        line.set_label(label)
    ax.legend()


class IRB_Exp(PersistableExperiment[IRBResult, IRBCfg]):
    """Acquire both arms with software switching and hardware depth/repetitions."""

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("depths", "Depth", "a.u.", IDENTITY, np.int64),
            Axis("arms", "Arm (0=reference, 1=interleaved)", "", IDENTITY, np.int64),
            Axis("sub_seeds", "Seed", "", IDENTITY, np.int64),
            Axis("rounds", "Round", "", IDENTITY, np.int64),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=IRBResult,
        cfg_type=IRBCfg,
        tag="twotone/ge/irb",
    )

    def run(self, cfg: IRBCfg, *, context: RunContext) -> IRBResult:
        """Run paired rounds; preserve partial IQ and propagate failure via context.

        Each seed owns two cached programs. Alternate arm order with seed and
        round parity. Each program acquires exactly one round; stopping leaves
        untouched slots NaN. No calibration or device settings are written back.
        """
        run_cfg = deepcopy(cfg)
        depths = sweep2array(run_cfg.sweep.depth, allow_array=True).astype(np.int64)
        seeds = np.array(
            [
                int(child.generate_state(1)[0])
                for child in np.random.SeedSequence(run_cfg.seed).spawn(run_cfg.n_seeds)
            ],
            dtype=np.int64,
        )
        setup_devices(
            run_cfg, context.devices, progress=True, cancel_signal=context.cancel_signal
        )
        viewer = context.plots.liveplot_1d(
            "measurement",
            "Random Clifford depth",
            "Signal (a.u.)",
            num_lines=2,
            configure_axes=_label_arms,
        )
        buffer = SignalBuffer(
            (run_cfg.rounds, len(seeds), 2, len(depths)),
            on_update=lambda data: viewer.update(
                depths.astype(float), project_arm_means(data)
            ),
        )
        cache: dict[tuple[int, int], ModularProgramV2] = {}
        program_cfg = run_cfg.with_updates(rounds=1)
        with Schedule(run_cfg, buffer, stop=context.cancel_signal) as sched:
            for round_index, round_step in sched.repeat("round", run_cfg.rounds):
                for seed_index, (seed, seed_step) in enumerate(
                    round_step.scan("seed", seeds.tolist())
                ):
                    order = (0, 1) if (round_index + seed_index) % 2 == 0 else (1, 0)
                    for arm in order:
                        if sched.is_stop():
                            break
                        arm_step = seed_step.child(arm)
                        builder = arm_step.prog_builder(
                            context.soc, context.soccfg, cfg=program_cfg
                        )
                        key = (seed_index, arm)
                        if key not in cache:
                            tables = make_seed_tables(
                                seed, depths, run_cfg.target_gate if arm else None
                            )
                            cache[key] = build_rb_program(
                                builder, run_cfg.modules, tables
                            )
                        _ = builder.run_program(cache[key])
            sched.trigger_update(flush=True)
        return IRBResult(
            rounds=np.arange(run_cfg.rounds, dtype=np.int64),
            sub_seeds=seeds,
            arms=np.array([0, 1], dtype=np.int64),
            depths=depths,
            signals=buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[IRBCfg, IRBResult],
        options: None,
        *,
        plots: Plots,
    ) -> IRBAnalysis:
        """Fit paired complete IQ, publishing both decays and seed-bootstrap CI."""
        del options
        result = source.result
        if not np.array_equal(result.arms, [0, 1]):
            raise ValueError("IRB arm axis must be [0=reference, 1=interleaved]")
        analysis = analyze_irb(result.depths, result.signals)
        target = source.cfg.target_gate if source.cfg is not None else "target"
        fig, ax = plots.subplots("fit", figsize=(9, 6))
        for arm, label in enumerate(("Reference", f"Interleaved {target}")):
            (line,) = ax.plot(
                result.depths, analysis.mean_signals[arm], "o", label=label
            )
            ax.plot(result.depths, analysis.fit_signals[arm], color=line.get_color())
        ax.set(
            xlabel="Random Clifford depth",
            ylabel="Projected IQ (a.u.)",
            title=f"{target} fidelity estimate: {analysis.gate_fidelity:.6f}\n"
            f"95% seed bootstrap: [{analysis.fidelity_ci_low:.6f}, "
            f"{analysis.fidelity_ci_high:.6f}] (statistical only)",
        )
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return analysis
