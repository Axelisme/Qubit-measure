from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from zcu_tools.analysis.fitting import fit_decay
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting import live_line
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    ComputedPulse,
    LoadValue,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.process import rotate2real

from .sequence import make_seed_tables


@dataclass(frozen=True)
class RB_Result:
    sub_seeds: NDArray[np.int64]
    depths: NDArray[np.int64]
    signals2D: NDArray[np.complex128]


def rb_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    mask = np.any(np.isfinite(signals), axis=0)  # (depths, )
    mean_signals = np.full(
        (signals.shape[1],), np.nan, dtype=np.complex128
    )  # (depths,)
    mean_signals[mask] = np.nanmean(signals[..., mask], axis=0)
    return rotate2real(mean_signals).real


class RBCfg(ConfigBase):
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
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset", group="Modules")] = None
    I_pulse: Annotated[
        PulseCfg | None, FieldMeta(label="Identity Pulse", group="Modules")
    ] = None
    X90_pulse: Annotated[PulseCfg, FieldMeta(label="X90 Pulse", group="Modules")]
    X180_pulse: Annotated[PulseCfg, FieldMeta(label="X180 Pulse", group="Modules")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout", group="Modules")]
    depths: Annotated[
        list[Annotated[int, Field(ge=0, strict=True)]],
        FieldMeta(label="Clifford depths"),
    ] = Field(min_length=1)
    """Ordered nonempty, unique nonnegative integer Clifford depths."""
    seed: Annotated[int, FieldMeta(label="Random seed")]
    n_seeds: Annotated[int, FieldMeta(label="Number of sequences")]

    @model_validator(mode="after")
    def validate_depths(self) -> Self:
        """Reject repeated depths before building any sequence or doing device I/O."""
        if len(set(self.depths)) != len(self.depths):
            raise ValueError("RB requires unique nonnegative integer depths")
        return self


@dataclass(frozen=True)
class RBAnalysis:
    epc: float
    fidelity: float


class RB_Exp(Experiment[RB_Result, RBCfg]):
    # depths/sub_seeds are integer sweeps on disk -> scale=IDENTITY (1.0).
    # axes inner-first [depths, sub_seeds] so native z == signals2D
    # (n_seeds, n_depths) with zero transpose.
    AXES_SPEC: ClassVar[AxesSpec[RB_Result, RBCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("depths", "Depth", "a.u.", dtype=np.int64),
            AxesSpec.Axis("sub_seeds", "Entropy", "a.u.", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals2D", "Signal", "a.u."),
        result_type=RB_Result,
        cfg_type=RBCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/rb",
    )

    def run(self, cfg: RBCfg, *, context: RunContext) -> RB_Result:
        """Measure randomized Clifford decay for error-per-Clifford analysis.

        For each seed, reset, play a random Clifford prefix, apply its full
        inverse in two recovery slots, then read out. Return mean complex IQ
        per (seed, depth); unfinished acquisitions remain NaN. Depth and reps
        run on hardware, with rounds averaged within each seed's program.
        Acquisition errors are recorded on context.scope.stop.
        """
        # 1. Prepare reproducible seeds and a depth axis without changing cfg.
        run_cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        depths = np.asarray(run_cfg.depths, dtype=np.int64)

        ss = np.random.SeedSequence(run_cfg.seed)
        entropys = np.array(
            [int(child.generate_state(1)[0]) for child in ss.spawn(run_cfg.n_seeds)],
            dtype=np.int64,
        )

        viewer = live_line(context.plots, "measurement", "Depth", "Signal")

        signals_buffer = SignalBuffer(
            (len(entropys), len(depths)),
            on_update=lambda data: viewer.update(
                depths.astype(np.float64),
                rb_signal2real(data),
            ),
        )
        with Schedule(run_cfg, signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("seed", entropys.tolist()):
                tables = make_seed_tables(int(step.value), depths)
                builder = step.prog_builder(
                    soc, soccfg, reps=step.cfg.reps, relax_delay=step.cfg.relax_delay
                )

                # 2. Map the seed's physical gate IDs to calibrated pulses.
                modules = step.cfg
                (
                    rand_gate_seq,
                    prefix_len_by_depth,
                    recovery_gate0_by_depth,
                    recovery_gate1_by_depth,
                ) = tables
                max_rand_len = max(prefix_len_by_depth, default=0)

                Id_pulse = modules.I_pulse
                X90_pulse = modules.X90_pulse
                X180_pulse = modules.X180_pulse
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

                # 3. Reset, play the depth prefix, recover the full Clifford, read out.
                program = (
                    builder.add(
                        LoadValue(
                            "load_rand_len",
                            values=prefix_len_by_depth,
                            idx_reg="depth_idx",
                            val_reg="rand_len",
                        ),
                        LoadValue(
                            "load_recovery_gate_0",
                            values=recovery_gate0_by_depth,
                            idx_reg="depth_idx",
                            val_reg="recovery_gate_0",
                        ),
                        LoadValue(
                            "load_recovery_gate_1",
                            values=recovery_gate1_by_depth,
                            idx_reg="depth_idx",
                            val_reg="recovery_gate_1",
                        ),
                        Reset("reset", cfg=modules.reset),
                        Repeat(
                            "rand_gate_idx",
                            "rand_len",
                            range_hint=(0, max_rand_len),
                        ).add_content(
                            [
                                LoadValue(
                                    "load_rand_gate",
                                    values=rand_gate_seq,
                                    idx_reg="rand_gate_idx",
                                    val_reg="gate_idx",
                                ),
                                ComputedPulse(
                                    "basic_gate",
                                    val_reg="gate_idx",
                                    pulses=gate_pulses,
                                ),
                            ]
                        ),
                        # Full candidates in both slots keep recovery timing
                        # independent of depth, even for identity padding.
                        ComputedPulse(
                            "recovery_gate_0",
                            val_reg="recovery_gate_0",
                            pulses=gate_pulses,
                        ),
                        ComputedPulse(
                            "recovery_gate_1",
                            val_reg="recovery_gate_1",
                            pulses=gate_pulses,
                        ),
                        Readout("readout", cfg=modules.readout),
                    )
                    .declare_sweep("depth_idx", len(prefix_len_by_depth))
                    .build()
                )

                # 4. Average reps/rounds into this seed's depth trace.
                _ = builder.run_program(program, rounds=cfg.rounds)

        return RB_Result(
            sub_seeds=entropys,
            depths=depths,
            signals2D=signals_buffer.array,
        )

    def analyze(
        self,
        source: RunRecord[RBCfg, RB_Result],
        options: None,
        *,
        plots: Plots,
    ) -> RBAnalysis:
        result = source.result
        del options

        depths = result.depths
        signals2D = result.signals2D

        real_signals_avg = rb_signal2real(signals2D)
        depths_f = depths.astype(np.float64)

        decay_time, decay_err, fit_signals, _ = fit_decay(depths_f, real_signals_avg)

        p = np.exp(-1.0 / decay_time)
        p_err = p / (decay_time**2) * decay_err
        epc = (1.0 - p) / 2.0  # (1 - p)(d - 1) / d, d = 2
        epc_err = p_err / 2.0
        fidelity = 1.0 - epc
        fidelity_err = epc_err

        fig, ax = plots.subplots("fit", figsize=config.figsize)

        for si in range(signals2D.shape[0]):
            per_seed = rotate2real(signals2D[si]).real
            ax.plot(
                depths_f,
                per_seed,
                marker=".",
                linestyle="None",
                color="gray",
                alpha=0.3,
                markersize=3,
            )

        ax.plot(
            depths_f,
            real_signals_avg,
            marker="o",
            linestyle="None",
            label="Average",
            zorder=2,
        )
        ax.plot(depths_f, fit_signals, "-", color="red", label="Fit", zorder=3)

        ax.set_xlabel("Number of Cliffords")
        ax.set_ylabel("Signal (a.u.)")
        ax.set_title(
            f"RB: EPC = {epc:.2e}, Fidelity = {fidelity:.6f} ± {fidelity_err:.6f}"
        )
        ax.legend()
        ax.grid(True)

        fig.tight_layout()

        return RBAnalysis(epc=float(epc), fidelity=float(fidelity))
