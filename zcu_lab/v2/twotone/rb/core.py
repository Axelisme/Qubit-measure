"""Single-qubit RB with optional paired target interleaving."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, TypeAlias

import numpy as np
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fitting import (
    IRBAnalysis,
    analyze_irb,
    analyze_reference_rb,
    project_arm_means,
)
from zcu_tools.analysis.fitting import RBAnalysis as ReferenceRBAnalysis
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ComputedPulse,
    LoadValue,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    TimelineBuilder,
)

from zcu_lab.v2.support import live_errorbar

from .._rb_sequence import TargetGate, make_seed_tables


@dataclass(frozen=True)
class RB_Result:
    """Raw RB axes and IQ, including NaNs for unfinished acquisitions."""

    sub_seeds: NDArray[np.int64]
    """Independent RNG seeds shared with the interleaved arm, if present."""
    arms: NDArray[np.int64]
    """[0] for reference only; [0, 1] for reference followed by interleaved."""
    depths: NDArray[np.int64]
    """Number of random Cliffords; interleaved also inserts this many targets."""
    signals: NDArray[np.complex128]
    """IQ averaged over reps and completed rounds, shape (seed, arm, depth)."""


RBAnalysis: TypeAlias = ReferenceRBAnalysis | IRBAnalysis


class RBCfg(ConfigBase):
    """Single-qubit RB settings: depth counts random Cliffords, rounds sets acquisition rounds."""

    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    I_pulse: Annotated[PulseCfg | None, FieldMeta(label="Identity Pulse")] = None
    X90_pulse: Annotated[PulseCfg, FieldMeta(label="X90 Pulse")]
    X180_pulse: Annotated[PulseCfg, FieldMeta(label="X180 Pulse")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]
    depth: Annotated[
        list[Annotated[int, Field(ge=0, strict=True)]],
        FieldMeta(label="Clifford depths", unit="count"),
    ] = Field(min_length=1)
    """Ordered nonempty nonnegative integer Clifford depths."""
    seed: Annotated[int, FieldMeta(label="Random seed")]
    n_seeds: Annotated[int, FieldMeta(label="Number of sequences", unit="count")] = (
        Field(ge=2)
    )
    """Number of independent random sequences, shared between arms when interleaving."""

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
            description="Number of acquisition rounds to average per seed and arm.",
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

    target_gate: Annotated[
        TargetGate | None,
        FieldMeta(
            label="Target gate",
            description="None runs only reference RB. Select a gate to also run the interleaved arm.",
        ),
    ] = None
    """Physical Clifford inserted after each random Clifford in the interleaved arm."""


def _live_plot(
    plots: Plots, depths: NDArray[np.int64], n_arms: int
) -> Callable[[NDArray[np.complex128]], None]:
    """Show one live line per arm and return its owner-dispatched IQ update."""
    fig, ax = plots.subplots(num="measurement")
    ax.set(xlabel="Random Clifford depth (count)", ylabel="Signal (a.u.)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid()
    viewer = live_errorbar(ax, num_lines=n_arms)
    labels = ("Reference", "Interleaved")[:n_arms]
    for line, label in zip(ax.lines, labels, strict=True):
        line.set_label(label)
    ax.legend()

    xs = depths.astype(np.float64)

    @plots.on_plot_thread
    def update(data: NDArray[np.complex128]):
        projection = project_arm_means(data)
        return viewer.set(xs, projection.mean_signals, projection.seed_std)

    fig.show()
    return update


class RB_Exp(Experiment[RB_Result, RBCfg]):
    """Acquire reference RB and optionally an interleaved arm on the same seeds."""

    AXES_SPEC: ClassVar[AxesSpec[RB_Result, RBCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("depths", label="Depth", unit="a.u.", dtype=np.int64),
            AxesSpec.Axis(
                "arms",
                label="Arm (0=reference, 1=interleaved)",
                unit="",
                dtype=np.int64,
            ),
            AxesSpec.Axis("sub_seeds", label="Seed", unit="", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=RB_Result,
        cfg_type=RBCfg,
        cfg_schema_version="5.0",
        tag="twotone/ge/rb",
    )

    def run(self, cfg: RBCfg, *, context: RunContext) -> RB_Result:
        """Measure reference Clifford decay and, if configured, target decay.

        Reset, play each random Clifford (with a target after it in arm 1),
        apply that arm's full inverse in two recovery slots, then read out.
        Acquire each seed/arm program once, sweeping depth and reps on board.
        Return IQ averaged over reps and completed rounds in (seed, arm, depth)
        order. Cells without a completed round remain NaN.
        """
        # 1. Prepare the arms, reproducible seeds and depth axis.
        run_cfg = deepcopy(cfg)
        depths = np.asarray(run_cfg.depth, dtype=np.int64)
        arms = np.arange(1 if run_cfg.target_gate is None else 2, dtype=np.int64)
        seeds = np.array(
            [
                int(child.generate_state(1)[0])
                for child in np.random.SeedSequence(run_cfg.seed).spawn(run_cfg.n_seeds)
            ],
            dtype=np.int64,
        )
        update = _live_plot(context.plots, depths, len(arms))
        buffer = SignalBuffer(
            (len(seeds), len(arms), len(depths)),
            on_update=update,
        )
        # 2. Acquire reference first, then the optional interleaved arm for each seed.
        with Schedule(buffer, scope=context.scope) as sched:
            for seed, seed_step in sched.scan("seed", seeds.tolist()):
                for arm, arm_step in seed_step.scan("arm", arms.tolist()):
                    builder = TimelineBuilder(
                        context.soccfg,
                        reps=run_cfg.reps,
                        relax_delay=run_cfg.relax_delay,
                    )
                    # 3. Realize reference or target-interleaved Cliffords with calibrated rotations.
                    tables = make_seed_tables(
                        seed, depths, run_cfg.target_gate if arm == 1 else None
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
                    MX90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase + 180.0)
                    Y90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase + 90.0)
                    Y180_pulse = X180_pulse.with_updates(phase=X180_pulse.phase + 90.0)
                    MY90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase - 90.0)

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

                    # 4. Play this arm's prefix and its own full inverse.
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
                    # 5. ADR-0083: Average finished rounds in acquisition.
                    arm_step.run_acquire(
                        builder.build(), context.soc, rounds=run_cfg.rounds
                    )
            sched.trigger_update(flush=True)
        return RB_Result(
            sub_seeds=seeds,
            arms=arms,
            depths=depths,
            signals=buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[RBCfg, RB_Result],
        options: None,
        *,
        plots: Plots,
    ) -> RBAnalysis:
        """Fit complete reference or paired IQ and plot seed-bootstrap estimates.

        Use the saved arm axis to choose the model. Return reference decay, error
        per Clifford and its 95% CI for one arm, or both decays and target-gate
        fidelity with its 95% CI for two. The intervals quantify seed variation,
        not model systematics. Invalid data or unidentifiable fits raise.
        """
        # 1. Select the estimate based on the saved arms, not on current settings.
        del options
        result = source.result
        if len(result.arms) == 1:
            fit = analyze_reference_rb(result.depths, result.signals)
            analysis = fit.summary
            labels = ("Reference",)
            title = (
                f"Reference decay: p = {analysis.p_reference:.6f}, "
                f"error per Clifford = {analysis.reference_epc:.6f}\n"
                f"95% seed bootstrap EPC: [{analysis.epc_ci_low:.6f}, "
                f"{analysis.epc_ci_high:.6f}] (statistical only)"
            )
        else:
            fit = analyze_irb(result.depths, result.signals)
            analysis = fit.summary
            target = source.cfg.target_gate if source.cfg is not None else None
            name = "target" if target is None else target
            labels = ("Reference", f"Interleaved {name}")
            title = (
                f"{name} fidelity estimate: {analysis.gate_fidelity:.6f}\n"
                f"95% seed bootstrap: [{analysis.fidelity_ci_low:.6f}, "
                f"{analysis.fidelity_ci_high:.6f}] (statistical only)"
            )

        # 2. Plot the observed and fitted decay in the same IQ projection.
        fig, ax = plots.subplots(num="fit", figsize=(9, 6))
        for arm, label in enumerate(labels):
            (line,) = ax.plot(result.depths, fit.mean_signals[arm], "o", label=label)
            ax.errorbar(
                result.depths,
                fit.mean_signals[arm],
                yerr=fit.seed_std[arm],
                fmt="none",
                ecolor=line.get_color(),
            )
            ax.plot(result.depths, fit.fit_signals[arm], color=line.get_color())
        ax.set(
            xlabel="Random Clifford depth (count)",
            ylabel="Projected IQ (a.u.)",
            title=title,
        )
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return analysis
