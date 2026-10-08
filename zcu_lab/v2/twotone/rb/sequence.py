"""Clifford algebra and frame-preserving tables shared by RB and IRB."""

from enum import IntEnum
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

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

TargetGate: TypeAlias = Literal["X90", "X180", "Y90", "Y180"]


def make_seed_tables(
    seed: int,
    depths: NDArray[np.int64],
    target: TargetGate | None = None,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Build reproducible RB/IRB tables for nonnegative integer depths.

    The same seed supplies identical random Cliffords in both arms. When target
    is given, append that physical Clifford after every random Clifford without
    combining them. Depth still counts random Cliffords (and target insertions).
    Recovery inverts the full sequence, including target gates. Raises ValueError
    for an empty/negative depth grid or an unsupported target.
    """
    if depths.ndim != 1 or depths.size == 0 or np.any(depths < 0):
        raise ValueError("Depths must be a nonempty nonnegative integer vector")
    if target is not None and target not in ("X90", "X180", "Y90", "Y180"):
        raise ValueError(f"Unsupported interleaved target: {target}")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    random_sequence = rng.integers(0, NUM_CLIFFORDS, size=int(np.max(depths)))
    sequence = random_sequence.tolist()
    positions = depths
    if target is not None:
        target_index = CLIFFORD_GROUP.index((target,))
        sequence = [gate for ci in sequence for gate in (ci, target_index)]
        positions = 2 * depths
    accumulated = 0
    recovery = [INVERSE_INDEX[accumulated]]
    for ci in sequence:
        accumulated = CAYLEY[ci][accumulated]
        recovery.append(INVERSE_INDEX[accumulated])
    return build_seed_program_tables(sequence, recovery, positions)


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


# ---------- 6-state Bloch-sphere permutation representation ------------------
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
# fmt: on


# ---------- Cayley table + inverse lookup ------------------------------------
# The action of the 24-element Clifford quotient group on the 6 Bloch cardinal
# states is faithful (the quotient is the chiral octahedral group ≅ S4), so
# permutation equality is group-element equality and the multiplication table
# can be synthesized from GATE_EFFECT_MAP alone.
#
# Order convention (must stay in sync with the accumulation in RB_Exp.run):
# CAYLEY[i][j] = k  such that  perm_k == perm_i ∘ perm_j,  i.e. apply C_j
# FIRST, then C_i.  Accumulating a sequence therefore reads
# acc = CAYLEY[next][acc], and the full-inverse recovery (applied last) is
# INVERSE_INDEX[acc] since CAYLEY[INVERSE_INDEX[acc]][acc] == 0 (identity).


def _clifford_perm(ci: int) -> tuple[int, ...]:
    perm: list[int] = []
    for s in range(6):
        st = s
        for gate in CLIFFORD_GROUP[ci]:
            st = GATE_EFFECT_MAP[gate][st]
        perm.append(st)
    return tuple(perm)


def _build_cayley_and_inverse() -> tuple[list[list[int]], list[int]]:
    perms = [_clifford_perm(i) for i in range(NUM_CLIFFORDS)]
    perm_index: dict[tuple[int, ...], int] = {}
    for i, p in enumerate(perms):
        if p in perm_index:
            raise ValueError(
                "CLIFFORD_GROUP 6-state permutations are not faithful: "
                f"C{perm_index[p]} and C{i} coincide"
            )
        perm_index[p] = i

    def compose(pi: tuple[int, ...], pj: tuple[int, ...]) -> tuple[int, ...]:
        # apply pj first, then pi
        return tuple(pi[pj[s]] for s in range(6))

    cayley = [[0] * NUM_CLIFFORDS for _ in range(NUM_CLIFFORDS)]
    for i in range(NUM_CLIFFORDS):
        for j in range(NUM_CLIFFORDS):
            k = perm_index.get(compose(perms[i], perms[j]))
            if k is None:
                raise ValueError(f"Clifford product C{i}·C{j} is not in the group")
            cayley[i][j] = k

    inverse = [0] * NUM_CLIFFORDS
    for i in range(NUM_CLIFFORDS):
        invs = [j for j in range(NUM_CLIFFORDS) if cayley[i][j] == 0]
        if len(invs) != 1 or cayley[invs[0]][i] != 0:
            raise ValueError(f"Clifford C{i} has no unique two-sided inverse")
        inverse[i] = invs[0]
    return cayley, inverse


CAYLEY, INVERSE_INDEX = _build_cayley_and_inverse()

# The recovery Clifford decompositions contain at most 2 physical pulses
# (Z gates are virtual), so the program uses exactly 2 recovery slots and
# pads with BasicGate.Id.
NUM_RECOVERY_SLOTS = 2


def build_seed_program_tables(
    total_clifford_seq: list[int],
    recovery_idx_by_pos: list[int],
    depths: NDArray[np.int64],
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Reduce a Clifford sequence to physical-gate dmem tables.

    `recovery_idx_by_pos[d]` is the Clifford index of the full inverse of the
    first `d` sequence Cliffords (one entry per position 0..len(seq)).
    Returns `(rand_gate_seq, prefix_len_by_depth, recovery_gate0_by_depth,
    recovery_gate1_by_depth)` — the recovery decomposition holds at most
    NUM_RECOVERY_SLOTS physical pulses; missing slots are padded with
    BasicGate.Id (slot 0 fires first).
    """
    max_depth = int(np.max(depths))

    if len(recovery_idx_by_pos) != len(total_clifford_seq) + 1:
        raise ValueError(
            "recovery_idx_by_pos must have one entry per position "
            f"0..len(seq): expected {len(total_clifford_seq) + 1}, "
            f"got {len(recovery_idx_by_pos)}"
        )
    if max_depth > len(total_clifford_seq):
        raise ValueError(
            f"max depth {max_depth} exceeds Clifford sequence length "
            f"{len(total_clifford_seq)}"
        )

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
    recovery_gate0_all: list[int] = [int(BasicGate.Id)] * (max_depth + 1)
    recovery_gate1_all: list[int] = [int(BasicGate.Id)] * (max_depth + 1)

    for d in range(max_depth + 1):
        prefix_len_all[d] = len(rand_gate_seq)

        # Recovery inherits the virtual-Z frame accumulated by the prefix;
        # its own frame advance is discarded (nothing follows before readout).
        saved_phase = phase_axis
        recovery_gates: list[int] = []
        for r_gate in CLIFFORD_GROUP[recovery_idx_by_pos[d]]:
            basic_gate = convert_gate(r_gate)
            if basic_gate is not None:
                recovery_gates.append(int(basic_gate))
        phase_axis = saved_phase

        if len(recovery_gates) > NUM_RECOVERY_SLOTS:
            raise ValueError(
                "recovery Clifford decomposition exceeds "
                f"{NUM_RECOVERY_SLOTS} physical gates: {recovery_gates}"
            )
        if len(recovery_gates) >= 1:
            recovery_gate0_all[d] = recovery_gates[0]
        if len(recovery_gates) >= 2:
            recovery_gate1_all[d] = recovery_gates[1]

        if d < max_depth:
            ci = total_clifford_seq[d]
            for gate in CLIFFORD_GROUP[ci]:
                basic_gate = convert_gate(gate)
                if basic_gate is not None:
                    rand_gate_seq.append(int(basic_gate))

    prefix_len_by_depth: list[int] = []
    recovery_gate0_by_depth: list[int] = []
    recovery_gate1_by_depth: list[int] = []
    for depth_i64 in depths:
        d = int(depth_i64)
        prefix_len_by_depth.append(prefix_len_all[d])
        recovery_gate0_by_depth.append(recovery_gate0_all[d])
        recovery_gate1_by_depth.append(recovery_gate1_all[d])

    return (
        rand_gate_seq,
        prefix_len_by_depth,
        recovery_gate0_by_depth,
        recovery_gate1_by_depth,
    )
