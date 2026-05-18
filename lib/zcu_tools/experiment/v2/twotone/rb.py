from __future__ import annotations

from copy import deepcopy
from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Literal, Optional, TypeAlias, Union

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ComputedPulse,
    LoadValue,
    ModularProgramV2,
    ProgramV2Cfg,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
    SweepCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.process import rotate2real

# (sub_seeds, depths, signals2D[n_seeds x n_depths])
RB_Result: TypeAlias = tuple[
    NDArray[np.int64], NDArray[np.int64], NDArray[np.complex128]
]

# ==============================================================================
# Single-qubit Clifford group (24 elements)
#
# Decompositions follow Table I of arXiv:1501.02041v3.  Each Clifford is
# U = Rx(θx)·Ry(θy)·Rz(θz) with Rz acting first on the qubit state.  The
# tuple is ordered left-to-right as (Rz, Ry, Rx).
#
# Only X90 and X180 are physical pulses from the cfg.  Y gates use the
# same pulses with +90° phase offset; -X gates use +180° offset.  Z gates
# (Z90, Z180, -Z90) are purely virtual — they accumulate in frame_phase
# and cost zero physical pulses.
# ==============================================================================


class BasicGate(IntEnum):
    Id = 0
    X90 = 1
    X180 = 2
    MX90 = 3
    Y90 = 4
    Y180 = 5
    MY90 = 6


GateName: TypeAlias = Literal[
    "Id", "X90", "X180", "-X90", "Y90", "Y180", "-Y90", "Z90", "Z180", "-Z90"
]


NUM_CLIFFORDS = 24

# fmt: off
CliffordDecomp: TypeAlias = tuple[GateName, ...]
CLIFFORD_GROUP: list[CliffordDecomp] = [
    ("Id",),                        # C0  — Id
    ("Z90",),                       # C1  — Rz(π/2)
    ("Z180",),                      # C2  — Rz(π)
    ("-Z90",),                      # C3  — Rz(-π/2)
    ("Y180",),                      # C4  — Ry(π)
    ("Z90", "Y180"),                # C5  — Ry(π)·Rz(π/2)
    ("X180",),                      # C6  — Rx(π)
    ("Z90", "X180"),                # C7  — Rx(π)·Rz(π/2)
    ("Y90", "X180"),                # C8  — Rx(π)·Ry(π/2)
    ("-Y90",),                      # C9  — Ry(-π/2)
    ("Z90", "X90"),                 # C10 — Rx(π/2)·Rz(π/2)
    ("Z90", "Y180", "X90"),         # C11 — Rx(π/2)·Ry(π)·Rz(π/2)
    ("-Y90", "X180"),               # C12 — Rx(π)·Ry(-π/2)
    ("Z90", "-X90"),                # C13 — Rx(-π/2)·Rz(π/2)
    ("Y90",),                       # C14 — Ry(π/2)
    ("Z90", "Y180", "-X90"),        # C15 — Rx(-π/2)·Ry(π)·Rz(π/2)
    ("-Y90", "-X90"),               # C16 — Rx(-π/2)·Ry(-π/2)
    ("Y90", "-X90"),                # C17 — Rx(-π/2)·Ry(π/2)
    ("Y180", "-X90"),               # C18 — Rx(-π/2)·Ry(π)
    ("-X90",),                      # C19 — Rx(-π/2)
    ("-Y90", "X90"),                # C20 — Rx(π/2)·Ry(-π/2)
    ("X90",),                       # C21 — Rx(π/2)
    ("Y180", "X90"),                # C22 — Rx(π/2)·Ry(π)
    ("Y90", "X90"),                 # C23 — Rx(π/2)·Ry(π/2)
]


# ---------- 6-state Bloch-sphere tracking -----------------------------------
# States:  +X=0  -X=1  +Y=2  -Y=3  +Z=4  -Z=5
# Each primitive gate is a permutation of these 6 cardinal states, derived
# from the SO(3) rotation matrices of Rx, Ry, Rz at ±π/2 and π.
PX, MX, PY, MY, PZ, MZ = 0, 1, 2, 3, 4, 5
GATE_EFFECT_MAP = {
    "Id":    (PX, MX, PY, MY, PZ, MZ),
    "X90":  (PX, MX, PZ, MZ, MY, PY),
    "-X90": (PX, MX, MZ, PZ, PY, MY),
    "X180": (PX, MX, MY, PY, MZ, PZ),
    "Y90":  (MZ, PZ, PY, MY, PX, MX),
    "-Y90": (PZ, MZ, PY, MY, MX, PX),
    "Y180": (MX, PX, PY, MY, MZ, PZ),
    "Z90":  (PY, MY, MX, PX, PZ, MZ),
    "Z180": (MX, PX, MY, PY, PZ, MZ),
    "-Z90": (MY, PY, PX, MX, PZ, MZ),
}
# maps it back to +Z (= |0⟩).
# use ("-Y90", "Y90", "X90", "-X90", "I", "X180")
RECOVERY_INDEX = (9, 14, 21, 19, 0, 6)
# fmt: on


def reduce_gate_seq(seq: list[CliffordDecomp]) -> list[BasicGate]:
    phase_axis: int = 0

    def convert_gate(gate: GateName) -> Optional[BasicGate]:
        nonlocal phase_axis

        # fmt: off
        axis_map = {"Z90": 3, "Z180": 2, "-Z90": 1}
        gate_map = {
            "Id":   (BasicGate.Id,   BasicGate.Id,   BasicGate.Id,   BasicGate.Id),
            "X90":  (BasicGate.X90,  BasicGate.Y90,  BasicGate.MX90, BasicGate.MY90),
            "X180": (BasicGate.X180, BasicGate.Y180, BasicGate.X180, BasicGate.Y180),
            "-X90": (BasicGate.MX90, BasicGate.MY90, BasicGate.X90,  BasicGate.Y90),
            "Y90":  (BasicGate.Y90,  BasicGate.MX90, BasicGate.MY90, BasicGate.X90),
            "Y180": (BasicGate.Y180, BasicGate.X180, BasicGate.Y180, BasicGate.X180),
            "-Y90": (BasicGate.MY90, BasicGate.X90,  BasicGate.Y90,  BasicGate.MX90),
        }
        # fmt: on

        if gate in gate_map:
            return gate_map[gate][phase_axis]

        # virtual Z gate
        phase_axis = (phase_axis + axis_map[gate]) % 4

        return None

    reduced_seq: list[BasicGate] = []
    for cf_group in seq:
        for gate in cf_group:
            basic_gate = convert_gate(gate)
            if basic_gate is not None:
                reduced_seq.append(basic_gate)

    return reduced_seq


def rb_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    mask = np.any(np.isfinite(signals), axis=0)  # (depths, )
    mean_signals = np.full(
        (signals.shape[1],), np.nan, dtype=np.complex128
    )  # (depths,)
    mean_signals[mask] = np.nanmean(signals[..., mask], axis=0)
    return rotate2real(mean_signals).real


def build_seed_program_tables(
    total_clifford_seq: list[int],
    acc_states: list[int],
    depths: NDArray[np.int64],
) -> tuple[list[int], list[int], list[int]]:
    max_depth = int(np.max(depths))
    rand_cliffords = [CLIFFORD_GROUP[ci] for ci in total_clifford_seq[:max_depth]]
    rand_gate_seq = [int(gate) for gate in reduce_gate_seq(rand_cliffords)]

    prefix_len_by_depth: list[int] = []
    recovery_gate_by_depth: list[int] = []
    for depth_i64 in depths:
        depth = int(depth_i64)
        prefix_cliffords = [CLIFFORD_GROUP[ci] for ci in total_clifford_seq[:depth]]
        prefix_len_by_depth.append(len(reduce_gate_seq(prefix_cliffords)))

        recovery_idx = RECOVERY_INDEX[acc_states[depth]]
        recovery_seq = reduce_gate_seq([CLIFFORD_GROUP[recovery_idx]])
        if len(recovery_seq) != 1:
            raise ValueError(
                "RB recovery Clifford must map to exactly one physical BasicGate"
            )
        recovery_gate_by_depth.append(int(recovery_seq[0]))

    return rand_gate_seq, prefix_len_by_depth, recovery_gate_by_depth


class RBModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    I_pulse: Optional[PulseCfg] = None
    X90_pulse: PulseCfg
    X180_pulse: PulseCfg
    readout: ReadoutCfg


class RBSweepCfg(ConfigBase):
    depth: Union[SweepCfg, list[int]]


class RBCfg(ProgramV2Cfg, ExpCfgModel):
    modules: RBModuleCfg
    sweep: RBSweepCfg
    seed: int
    n_seeds: int


class RB_Exp(AbsExperiment[RB_Result, RBCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: RBCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> RB_Result:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)

        depths = sweep2array(cfg.sweep.depth, allow_array=True).astype(np.int64)

        max_depth = int(np.max(depths))

        ss = np.random.SeedSequence(cfg.seed)
        entropys = np.array(
            [child.entropy for child in ss.spawn(cfg.n_seeds)], dtype=np.int64
        )

        prog_cache: dict[int, ModularProgramV2] = {}

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, RBCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            seed: int = ctx.env["seed"]

            if seed not in prog_cache:
                rand_gate_seq: list[int] = ctx.env["rand_gate_seq"]
                prefix_len_by_depth: list[int] = ctx.env["prefix_len_by_depth"]
                recovery_gate_by_depth: list[int] = ctx.env["recovery_gate_by_depth"]
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

                prog_cache[seed] = ModularProgramV2(
                    soccfg,
                    cfg,
                    modules=[
                        LoadValue(
                            "load_rand_len",
                            values=prefix_len_by_depth,
                            idx_reg="depth_idx",
                            val_reg="rand_len",
                        ),
                        LoadValue(
                            "load_recovery_gate",
                            values=recovery_gate_by_depth,
                            idx_reg="depth_idx",
                            val_reg="recovery_gate",
                        ),
                        Reset("reset", modules.reset),
                        Repeat(
                            "rand_gate_idx", "rand_len", range_hint=(0, max_rand_len)
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
                        ComputedPulse(
                            "recovery_gate",
                            val_reg="recovery_gate",
                            pulses=gate_pulses,
                        ),
                        Readout("readout", modules.readout),
                    ],
                    sweep=[("depth_idx", len(depths))],
                )

            return prog_cache[seed].acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                **(acquire_kwargs or {}),
            )

        with LivePlot1D("Depth", "Signal") as viewer:

            def update_seq_seed(si: int, ctx: TaskState, entropy: int) -> None:
                child = np.random.SeedSequence(entropy)
                rng = np.random.Generator(np.random.PCG64(child))
                clifford_idxs = rng.integers(0, NUM_CLIFFORDS, size=max_depth)

                # simulate the state evolution after each Clifford gate
                state = PZ
                cum_states: list[int] = [state]
                for ci in clifford_idxs:
                    for gate in CLIFFORD_GROUP[ci]:
                        state = GATE_EFFECT_MAP[gate][state]
                    cum_states.append(state)

                ctx.env["seed"] = entropy
                rand_gate_seq, prefix_len_by_depth, recovery_gate_by_depth = (
                    build_seed_program_tables(
                        clifford_idxs.tolist(), cum_states, depths
                    )
                )
                ctx.env["rand_gate_seq"] = rand_gate_seq
                ctx.env["prefix_len_by_depth"] = prefix_len_by_depth
                ctx.env["recovery_gate_by_depth"] = recovery_gate_by_depth

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(depths),),
                    pbar_n=cfg.rounds,
                ).scan("seed", entropys.tolist(), before_each=update_seq_seed),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    depths.astype(np.float64),
                    rb_signal2real(np.asarray(ctx.root_data, dtype=np.complex128)),
                ),
            )
            signals2D = np.asarray(signals, dtype=np.complex128)  # (n_seeds, n_depths)

        self.last_cfg = deepcopy(cfg)
        self.last_result = (entropys, depths, signals2D)

        return entropys, depths, signals2D

    def analyze(
        self,
        result: Optional[RB_Result] = None,
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        entropys, depths, signals2D = result

        real_signals_avg = rb_signal2real(signals2D)
        depths_f = depths.astype(np.float64)

        decay_time, decay_err, fit_signals, (pOpt, pCov) = fit_decay(
            depths_f, real_signals_avg
        )

        p = np.exp(-1.0 / decay_time)
        p_err = p / (decay_time**2) * decay_err
        epc = (1.0 - p) / 2.0  # (1 - p)(d - 1) / d, d = 2
        epc_err = p_err / 2.0
        fidelity = 1.0 - epc
        fidelity_err = epc_err

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

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

        return epc, fidelity, fig

    def save(
        self,
        filepath: str,
        result: Optional[RB_Result] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/rb",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        entropys, depths, signals2D = result
        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={
                "name": "Entropy",
                "unit": "a.u.",
                "values": entropys.astype(np.float64),
            },
            y_info={
                "name": "Depth",
                "unit": "a.u.",
                "values": depths.astype(np.float64),
            },
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> RB_Result:
        signals, entropys, depths, comment = load_data(
            filepath, return_comment=True, **kwargs
        )
        assert depths is not None
        assert len(entropys.shape) == 1 and len(depths.shape) == 1
        assert signals.shape == (len(depths), len(entropys))

        signals = signals.T

        sub_seeds = entropys.astype(np.int64)
        depths = depths.astype(np.int64)
        signals2D = signals.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)

            if cfg is not None:
                self.last_cfg = RBCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (sub_seeds, depths, signals2D)

        return sub_seeds, depths, signals2D
