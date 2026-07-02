from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ComputedPulse,
    LoadValue,
    ProgramV2Cfg,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
    SweepCfg,
)
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class RB_Result:
    sub_seeds: NDArray[np.int64]
    depths: NDArray[np.int64]
    signals2D: NDArray[np.complex128]
    cfg_snapshot: RBCfg | None = None


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

    phase_axis: int = 0

    def convert_gate(gate: GateName) -> BasicGate | None:
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

    rand_gate_seq: list[int] = []
    prefix_len_all: list[int] = [0] * (max_depth + 1)
    recovery_gate_all: list[int] = [0] * (max_depth + 1)

    for d in range(max_depth + 1):
        prefix_len_all[d] = len(rand_gate_seq)

        recovery_idx = RECOVERY_INDEX[acc_states[d]]
        saved_phase = phase_axis
        recovery_gate = None
        for r_gate in CLIFFORD_GROUP[recovery_idx]:
            basic_gate = convert_gate(r_gate)
            if basic_gate is not None:
                if recovery_gate is not None:
                    raise ValueError(
                        "RB recovery Clifford must map to exactly one physical BasicGate"
                    )
                recovery_gate = basic_gate
        assert recovery_gate is not None
        recovery_gate_all[d] = int(recovery_gate)
        phase_axis = saved_phase

        if d < max_depth:
            ci = total_clifford_seq[d]
            for gate in CLIFFORD_GROUP[ci]:
                basic_gate = convert_gate(gate)
                if basic_gate is not None:
                    rand_gate_seq.append(int(basic_gate))

    prefix_len_by_depth: list[int] = []
    recovery_gate_by_depth: list[int] = []
    for depth_i64 in depths:
        d = int(depth_i64)
        prefix_len_by_depth.append(prefix_len_all[d])
        recovery_gate_by_depth.append(recovery_gate_all[d])

    return rand_gate_seq, prefix_len_by_depth, recovery_gate_by_depth


class RBModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    I_pulse: PulseCfg | None = None
    X90_pulse: PulseCfg
    X180_pulse: PulseCfg
    readout: ReadoutCfg


class RBSweepCfg(ConfigBase):
    depth: SweepCfg | list[int]


class RBCfg(ProgramV2Cfg, ExpCfgModel):
    modules: RBModuleCfg
    sweep: RBSweepCfg
    seed: int
    n_seeds: int


class RB_Exp(PersistableExperiment[RB_Result, RBCfg]):
    # depths/sub_seeds are integer sweeps on disk -> scale=IDENTITY (1.0).
    # axes inner-first [depths, sub_seeds] so native z == signals2D
    # (n_seeds, n_depths) with zero transpose.
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("depths", "Depth", "a.u.", IDENTITY, np.int64),
            Axis("sub_seeds", "Entropy", "a.u.", IDENTITY, np.int64),
        ),
        z=ZSpec("signals2D", "Signal", "a.u."),
        result_type=RB_Result,
        cfg_type=RBCfg,
        tag="twotone/ge/rb",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: RBCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> RB_Result:
        orig_cfg = deepcopy(cfg)
        run_cfg = deepcopy(cfg)
        setup_devices(run_cfg, progress=True)

        depths = sweep2array(run_cfg.sweep.depth, allow_array=True).astype(np.int64)

        max_depth = int(np.max(depths))

        ss = np.random.SeedSequence(run_cfg.seed)
        entropys = np.array(
            [child.entropy for child in ss.spawn(run_cfg.n_seeds)], dtype=np.int64
        )

        prog_cache: dict[int, Any] = {}

        with LivePlot1D("Depth", "Signal") as viewer:

            def build_seq_seed(
                entropy: int,
            ) -> tuple[int, list[int], list[int], list[int]]:
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

                rand_gate_seq, prefix_len_by_depth, recovery_gate_by_depth = (
                    build_seed_program_tables(
                        clifford_idxs.tolist(), cum_states, depths
                    )
                )
                return (
                    entropy,
                    rand_gate_seq,
                    prefix_len_by_depth,
                    recovery_gate_by_depth,
                )

            signals_buffer = SignalBuffer(
                (len(entropys), len(depths)),
                on_update=lambda data: viewer.update(
                    depths.astype(np.float64),
                    rb_signal2real(data),
                ),
            )
            with Schedule(run_cfg, signals_buffer) as sched:
                for _, step in sched.scan("seed", entropys.tolist()):
                    (
                        seed,
                        rand_gate_seq,
                        prefix_len_by_depth,
                        recovery_gate_by_depth,
                    ) = build_seq_seed(int(step.value))
                    builder = step.prog_builder(soc, soccfg)
                    if seed not in prog_cache:
                        modules = step.cfg.modules
                        max_rand_len = max(prefix_len_by_depth, default=0)

                        Id_pulse = modules.I_pulse
                        X90_pulse = modules.X90_pulse
                        X180_pulse = modules.X180_pulse
                        MX90_pulse = X90_pulse.with_updates(
                            phase=X90_pulse.phase + 180.0
                        )
                        Y90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase + 90.0)
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

                        prog_cache[seed] = (
                            builder.add(
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
                                ComputedPulse(
                                    "recovery_gate",
                                    val_reg="recovery_gate",
                                    pulses=gate_pulses,
                                ),
                                Readout("readout", cfg=modules.readout),
                            )
                            .declare_sweep("depth_idx", len(depths))
                            .build()
                        )

                    _ = builder.run_program(
                        prog_cache[seed],
                        **(acquire_kwargs or {}),
                    )
                signals2D = signals_buffer.array  # (n_seeds, n_depths)

        self.last_result = RB_Result(
            sub_seeds=entropys,
            depths=depths,
            signals2D=signals2D,
            cfg_snapshot=orig_cfg,
        )

        return self.last_result

    @retrieve_result
    def analyze(
        self,
        result: RB_Result | None = None,
    ) -> tuple[float, float, Figure]:
        assert result is not None, (
            "No measurement data available. Run experiment first."
        )

        depths = result.depths
        signals2D = result.signals2D

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
