"""Paired reference/interleaved RB with explicit rounds and seed uncertainty."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ComputedPulse,
    LoadValue,
    Readout,
    Reset,
    TimelineBuilder,
    TimelineProgramV2,
)

from zcu_lab.v2.support import live_line

from ..rb.core import RBCfg
from ..rb.sequence import TargetGate, make_seed_tables
from .analysis import IRBAnalysis, analyze_irb, project_arm_means


class IRBCfg(RBCfg):
    """Paired RB settings; depth counts random Cliffords, rounds are host passes."""

    target_gate: Annotated[TargetGate, FieldMeta(label="Target gate")] = "X90"
    """Physical Clifford inserted after each random Clifford, in its current frame."""
    n_seeds: Annotated[int, FieldMeta(label="Number of sequences")] = Field(ge=2)
    """Number of independent random sequences shared by both arms."""


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


class IRB_Exp(Experiment[IRBResult, IRBCfg]):
    """Acquire both arms with software switching and hardware depth/repetitions."""

    AXES_SPEC: ClassVar[AxesSpec[IRBResult, IRBCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("depths", label="Depth", unit="a.u.", dtype=np.int64),
            AxesSpec.Axis(
                "arms",
                label="Arm (0=reference, 1=interleaved)",
                unit="",
                dtype=np.int64,
            ),
            AxesSpec.Axis("sub_seeds", label="Seed", unit="", dtype=np.int64),
            AxesSpec.Axis("rounds", label="Round", unit="", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=IRBResult,
        cfg_type=IRBCfg,
        cfg_schema_version="4.0",
        tag="twotone/ge/irb",
    )

    def run(self, cfg: IRBCfg, *, context: RunContext) -> IRBResult:
        """Compare reference and target-interleaved Clifford decay on paired seeds.

        Reset, play each random Clifford (with a target after it in arm 1),
        apply that arm's full inverse in two recovery slots, then read out.
        Return repetition-averaged IQ in (round, seed, arm, depth) order for
        decay-ratio target-error analysis; unfinished slots remain NaN.
        Each seed/arm program is cached, with hardware depth/reps and host rounds.
        Acquisition errors are recorded on context.scope.stop.
        """
        # 1. Pair both arms on the same reproducible seeds and depth axis.
        run_cfg = deepcopy(cfg)
        depths = np.asarray(run_cfg.depth, dtype=np.int64)
        seeds = np.array(
            [
                int(child.generate_state(1)[0])
                for child in np.random.SeedSequence(run_cfg.seed).spawn(run_cfg.n_seeds)
            ],
            dtype=np.int64,
        )
        viewer = live_line(
            context.plots,
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
        cache: dict[tuple[int, int], TimelineProgramV2] = {}
        program_cfg = run_cfg.with_updates(rounds=1)
        with Schedule(buffer, scope=context.scope) as sched:
            for round_index, round_step in sched.repeat("round", run_cfg.rounds):
                for seed_index, (seed, seed_step) in enumerate(
                    round_step.scan("seed", seeds.tolist())
                ):
                    # Alternate which arm is measured first to balance slow drift.
                    order = (0, 1) if (round_index + seed_index) % 2 == 0 else (1, 0)
                    for arm in order:
                        if sched.is_stop():
                            break
                        arm_step = seed_step.child(arm)
                        builder = TimelineBuilder(
                            context.soccfg,
                            reps=program_cfg.reps,
                            relax_delay=program_cfg.relax_delay,
                        )
                        key = (seed_index, arm)
                        if key not in cache:
                            # 2. Prepare reference or interleaved physical gates.
                            tables = make_seed_tables(
                                seed, depths, run_cfg.target_gate if arm else None
                            )
                            (
                                rand_gate_seq,
                                prefix_len_by_depth,
                                recovery_gate0_by_depth,
                                recovery_gate1_by_depth,
                            ) = tables
                            max_rand_len = max(prefix_len_by_depth, default=0)

                            Id_pulse = run_cfg.I_pulse
                            X90_pulse = run_cfg.X90_pulse
                            X180_pulse = run_cfg.X180_pulse
                            MX90_pulse = X90_pulse.with_updates(
                                phase=X90_pulse.phase + 180.0
                            )
                            Y90_pulse = X90_pulse.with_updates(
                                phase=X90_pulse.phase + 90.0
                            )
                            Y180_pulse = X180_pulse.with_updates(
                                phase=X180_pulse.phase + 90.0
                            )
                            MY90_pulse = X90_pulse.with_updates(
                                phase=X90_pulse.phase - 90.0
                            )

                            if Id_pulse is None:
                                Id_pulse = X90_pulse.with_updates(gain=0.0)

                            gate_pulses = [
                                Id_pulse,
                                X90_pulse,
                                X180_pulse,
                                MX90_pulse,
                                Y90_pulse,
                                Y180_pulse,
                                MY90_pulse,
                            ]

                            # 3. Play this arm's prefix and its own full inverse.
                            builder.add_loop("depth", len(prefix_len_by_depth))
                            builder.emit(
                                LoadValue(
                                    "load_rand_len",
                                    values=prefix_len_by_depth,
                                    idx_reg="depth",
                                    val_reg="rand_len",
                                )
                            )
                            builder.emit(
                                LoadValue(
                                    "load_recovery_gate_0",
                                    values=recovery_gate0_by_depth,
                                    idx_reg="depth",
                                    val_reg="recovery_gate_0",
                                )
                            )
                            builder.emit(
                                LoadValue(
                                    "load_recovery_gate_1",
                                    values=recovery_gate1_by_depth,
                                    idx_reg="depth",
                                    val_reg="recovery_gate_1",
                                )
                            )
                            if run_cfg.reset is not None:
                                builder.emit(Reset("reset", run_cfg.reset))
                            builder.delay_by_timeline()
                            with builder.repeat(
                                "rand_gate_idx",
                                "rand_len",
                                range_hint=(0, max_rand_len),
                            ):
                                builder.emit(
                                    LoadValue(
                                        "load_rand_gate",
                                        values=rand_gate_seq,
                                        idx_reg="rand_gate_idx",
                                        val_reg="gate_idx",
                                    )
                                )
                                builder.emit(
                                    ComputedPulse(
                                        "basic_gate",
                                        val_reg="gate_idx",
                                        pulses=gate_pulses,
                                    )
                                )
                                builder.delay_by_timeline()
                            # Both recovery slots retain the complete candidate set for fixed timing.
                            builder.emit(
                                ComputedPulse(
                                    "recovery_gate_0",
                                    val_reg="recovery_gate_0",
                                    pulses=gate_pulses,
                                )
                            )
                            builder.emit(
                                ComputedPulse(
                                    "recovery_gate_1",
                                    val_reg="recovery_gate_1",
                                    pulses=gate_pulses,
                                )
                            )
                            builder.emit(Readout("readout", run_cfg.readout))
                            program = builder.build()

                            cache[key] = program
                        # 4. Store one round's mean IQ; reuse programs next round.
                        arm_step.run_acquire(cache[key], context.soc, rounds=1)
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
