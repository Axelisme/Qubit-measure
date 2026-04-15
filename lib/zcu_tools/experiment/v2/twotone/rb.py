from __future__ import annotations

from copy import deepcopy
from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    Literal,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    Union,
    cast,
)

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    ScanWith,
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
    mask = np.any(np.isfinite(signals), axis=0) # (depths, )
    mean_signals = np.full((signals.shape[1],), np.nan, dtype=np.complex128) # (depths,)
    mean_signals[mask] = np.nanmean(signals[..., mask], axis=0)
    return rotate2real(mean_signals).real


class RB_ModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    I_pulse: NotRequired[PulseCfg]
    X90_pulse: PulseCfg
    X180_pulse: PulseCfg
    readout: ReadoutCfg


class RB_SweepCfg(TypedDict, closed=True):
    depth: Union[SweepCfg, NDArray]


class RB_Cfg(ModularProgramCfg, TaskCfg):
    modules: RB_ModuleCfg
    sweep: RB_SweepCfg
    seed: int
    n_seeds: int


class RB_Exp(AbsExperiment[RB_Result, RB_Cfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: dict[str, Any],
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> RB_Result:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "depth")
        _cfg = check_type(deepcopy(cfg), RB_Cfg)

        rounds = _cfg["rounds"]
        _cfg["rounds"] = 1 # implement by task scan

        depths = sweep2array(_cfg["sweep"]["depth"], allow_array=True).astype(np.int64)

        max_depth = int(np.max(depths))

        ss = np.random.SeedSequence(_cfg["seed"])
        entropys = np.array(
            [child.entropy for child in ss.spawn(_cfg["n_seeds"])], dtype=np.int64
        )

        prog_cahce: dict[tuple[int, int], ModularProgramV2] = {}
        def measure_fn(
            ctx: TaskState,
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = cast(RB_Cfg, ctx.cfg)
            modules = cfg["modules"]

            seed:int = ctx.env["seed"]
            depth:int = ctx.env["depth"]

            if (seed, depth) not in prog_cahce:
                gate_seq = reduce_gate_seq(ctx.env["clifford_seq"])

                Id_pulse = modules.get("I_pulse")
                X90_pulse = modules["X90_pulse"]
                X180_pulse = modules["X180_pulse"]
                MX90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase + 180.0)
                Y90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase + 90.0)
                Y180_pulse = X180_pulse.with_updates(phase=X180_pulse.phase + 90.0)
                MY90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase - 90.0)

                prog_cahce[(seed, depth)] = ModularProgramV2(
                    soccfg,
                    cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        ScanWith("gate_idx", gate_seq, val_reg="gate_idx").add_content(
                            Branch(
                                "basic_gate",
                                Pulse("gate_Id", Id_pulse),
                                Pulse("gate_X90", X90_pulse),
                                Pulse("gate_X180", X180_pulse),
                                Pulse("gate_MX90", MX90_pulse),
                                Pulse("gate_Y90", Y90_pulse),
                                Pulse("gate_Y180", Y180_pulse),
                                Pulse("gate_MY90", MY90_pulse),
                                compare_by="gate_idx",
                            )
                        ),
                        Readout("readout", modules["readout"]),
                    ],
                )

            return prog_cahce[(seed, depth)].acquire(
                soc,
                progress=False,
                callback=update_hook,
                **(acquire_kwargs or {}),
            )

        def average_signals(signals: list[list[list[NDArray[np.complex128]]]]) -> NDArray[np.complex128]:
            _signals = np.asarray(signals)  # shape: (n_seeds, rounds, n_depths)
            mean_signals = np.full((_signals.shape[0], _signals.shape[2]), np.nan, np.complex128) # (n_seeds, n_depths)
            for i in range(_signals.shape[0]):
                mask = np.any(np.isfinite(_signals[i]), axis=0) # (n_depths, )
                mean_signals[i, mask] = np.nanmean(_signals[i, :, mask], axis=1)
            return mean_signals

        with LivePlot1D("Depth", "Signal") as viewer:

            def update_seq_depth(di: int, ctx: TaskState, depth: int) -> None:
                total_clifford_seq: list[int] = ctx.env["total_clifford_seq"]
                acc_states: list[int] = ctx.env["acc_states"]

                clifford_seq = total_clifford_seq[:depth]
                clifford_seq.append(RECOVERY_INDEX[acc_states[depth]])

                ctx.env["depth"] = depth
                ctx.env["clifford_seq"] = [CLIFFORD_GROUP[ci] for ci in clifford_seq]

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
                ctx.env["total_clifford_seq"] = clifford_idxs.tolist()
                ctx.env["acc_states"] = cum_states

            signals = run_task(
                task=Task(measure_fn=measure_fn)
                .scan("depth", depths.tolist(), before_each=update_seq_depth)
                .scan("rounds", list(range(rounds)), before_each=lambda *_: None)
                .scan("seed", entropys.tolist(), before_each=update_seq_seed),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    depths.astype(np.float64),
                    rb_signal2real(average_signals(ctx.root_data))
                ),
            )
            signals2D = average_signals(signals)  # shape: (n_seeds, n_depths)

        self.last_cfg = cast(RB_Cfg, deepcopy(cfg))
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
        signals, entropys, depths = load_data(filepath, **kwargs)
        assert depths is not None
        assert len(entropys.shape) == 1 and len(depths.shape) == 1
        assert signals.shape == (len(depths), len(entropys))

        signals = signals.T

        sub_seeds = entropys.astype(np.int64)
        depths = depths.astype(np.int64)
        signals2D = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (sub_seeds, depths, signals2D)

        return sub_seeds, depths, signals2D
