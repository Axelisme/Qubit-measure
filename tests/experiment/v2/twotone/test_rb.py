from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from zcu_tools.experiment.v2.twotone import rb

# ---------------------------------------------------------------------------
# Independent SU(2) reference (built from scratch, not from rb internals)
# ---------------------------------------------------------------------------


def _rot(axis: str, theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    if axis == "x":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if axis == "y":
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if axis == "z":
        return np.array(
            [[np.exp(-1j * theta / 2.0), 0.0], [0.0, np.exp(1j * theta / 2.0)]],
            dtype=np.complex128,
        )
    raise ValueError(f"unknown axis: {axis}")


_GATE_UNITARY: dict[str, np.ndarray] = {
    "Id": np.eye(2, dtype=np.complex128),
    "X90": _rot("x", np.pi / 2),
    "X180": _rot("x", np.pi),
    "-X90": _rot("x", -np.pi / 2),
    "Y90": _rot("y", np.pi / 2),
    "Y180": _rot("y", np.pi),
    "-Y90": _rot("y", -np.pi / 2),
    "Z90": _rot("z", np.pi / 2),
    "Z180": _rot("z", np.pi),
    "-Z90": _rot("z", -np.pi / 2),
}

_BASIC_GATE_UNITARY: dict[int, np.ndarray] = {
    int(rb.BasicGate.Id): np.eye(2, dtype=np.complex128),
    int(rb.BasicGate.X90): _rot("x", np.pi / 2),
    int(rb.BasicGate.X180): _rot("x", np.pi),
    int(rb.BasicGate.MX90): _rot("x", -np.pi / 2),
    int(rb.BasicGate.Y90): _rot("y", np.pi / 2),
    int(rb.BasicGate.Y180): _rot("y", np.pi),
    int(rb.BasicGate.MY90): _rot("y", -np.pi / 2),
}


def _clifford_unitary(ci: int) -> np.ndarray:
    # decomposition tuples are ordered first-applied first -> rightmost factor
    u = np.eye(2, dtype=np.complex128)
    for gate in rb.CLIFFORD_GROUP[ci]:
        u = _GATE_UNITARY[gate] @ u
    return u


def _equal_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-9) -> bool:
    return bool(np.isclose(np.abs(np.trace(a.conj().T @ b)), 2.0, atol=atol))


def _clifford_perm(ci: int) -> tuple[int, ...]:
    perm = []
    for s in range(6):
        st = s
        for gate in rb.CLIFFORD_GROUP[ci]:
            st = rb.GATE_EFFECT_MAP[gate][st]
        perm.append(st)
    return tuple(perm)


def _recovery_idx_by_pos(clifford_seq: Sequence[int]) -> list[int]:
    acc = 0
    out = [rb.INVERSE_INDEX[acc]]
    for ci in clifford_seq:
        acc = rb.CAYLEY[int(ci)][acc]
        out.append(rb.INVERSE_INDEX[acc])
    return out


# ---------------------------------------------------------------------------
# Cayley table / inverse structure
# ---------------------------------------------------------------------------


def test_clifford_perms_are_faithful() -> None:
    perms = [_clifford_perm(i) for i in range(rb.NUM_CLIFFORDS)]
    assert len(set(perms)) == rb.NUM_CLIFFORDS


def test_cayley_table_structure() -> None:
    n = rb.NUM_CLIFFORDS
    full = set(range(n))
    for i in range(n):
        assert set(rb.CAYLEY[i]) == full  # each row is a permutation
        assert {rb.CAYLEY[j][i] for j in range(n)} == full  # each column too
        assert rb.CAYLEY[0][i] == i  # C0 is left identity
        assert rb.CAYLEY[i][0] == i  # C0 is right identity


def test_inverse_index_is_two_sided() -> None:
    for i in range(rb.NUM_CLIFFORDS):
        inv = rb.INVERSE_INDEX[i]
        assert rb.CAYLEY[i][inv] == 0
        assert rb.CAYLEY[inv][i] == 0


def test_cayley_matches_su2_products() -> None:
    unitaries = [_clifford_unitary(i) for i in range(rb.NUM_CLIFFORDS)]
    for i in range(rb.NUM_CLIFFORDS):
        for j in range(rb.NUM_CLIFFORDS):
            # CAYLEY[i][j]: apply C_j first, then C_i
            product = unitaries[i] @ unitaries[j]
            assert _equal_up_to_phase(product, unitaries[rb.CAYLEY[i][j]])


def test_clifford_unitaries_pairwise_distinct() -> None:
    unitaries = [_clifford_unitary(i) for i in range(rb.NUM_CLIFFORDS)]
    for i in range(rb.NUM_CLIFFORDS):
        for j in range(i + 1, rb.NUM_CLIFFORDS):
            assert not _equal_up_to_phase(unitaries[i], unitaries[j])


# ---------------------------------------------------------------------------
# Recovery semantics: sequence + full inverse == identity
# ---------------------------------------------------------------------------


def test_sequence_with_recovery_is_identity() -> None:
    rng = np.random.Generator(np.random.PCG64(12345))
    for _ in range(50):
        depth = int(rng.integers(0, 30))
        seq = rng.integers(0, rb.NUM_CLIFFORDS, size=depth).tolist()
        recovery_idx = _recovery_idx_by_pos(seq)[depth]

        u = np.eye(2, dtype=np.complex128)
        for ci in seq:
            u = _clifford_unitary(int(ci)) @ u
        u = _clifford_unitary(recovery_idx) @ u
        assert _equal_up_to_phase(u, np.eye(2, dtype=np.complex128))


def test_physical_gate_product_is_z_rotation() -> None:
    # The emitted physical pulses omit virtual Zs, so the physical product
    # must be a pure Z rotation (diagonal) — Z-basis readout equivalent to
    # identity — at every requested depth.
    rng = np.random.Generator(np.random.PCG64(6789))
    for _ in range(20):
        max_depth = int(rng.integers(1, 25))
        seq = rng.integers(0, rb.NUM_CLIFFORDS, size=max_depth).tolist()
        depths = np.unique(rng.integers(0, max_depth + 1, size=4).astype(np.int64))

        rand_gate_seq, prefix_lens, rec0s, rec1s = rb.build_seed_program_tables(
            seq, _recovery_idx_by_pos(seq), depths
        )

        for prefix_len, rec0, rec1 in zip(prefix_lens, rec0s, rec1s):
            gates = rand_gate_seq[:prefix_len] + [rec0, rec1]
            u = np.eye(2, dtype=np.complex128)
            for g in gates:
                u = _BASIC_GATE_UNITARY[g] @ u
            assert np.allclose(u[0, 1], 0.0, atol=1e-9)
            assert np.allclose(u[1, 0], 0.0, atol=1e-9)
            assert np.allclose(np.abs(np.diag(u)), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# build_seed_program_tables reference reconstruction
# ---------------------------------------------------------------------------


def _reference_reduce_gate_seq(seq: Sequence[tuple[str, ...]]) -> list[int]:
    phase_axis = 0
    axis_map = {"Z90": 3, "Z180": 2, "-Z90": 1}
    gate_map = {
        "Id": (rb.BasicGate.Id, rb.BasicGate.Id, rb.BasicGate.Id, rb.BasicGate.Id),
        "X90": (
            rb.BasicGate.X90,
            rb.BasicGate.Y90,
            rb.BasicGate.MX90,
            rb.BasicGate.MY90,
        ),
        "X180": (
            rb.BasicGate.X180,
            rb.BasicGate.Y180,
            rb.BasicGate.X180,
            rb.BasicGate.Y180,
        ),
        "-X90": (
            rb.BasicGate.MX90,
            rb.BasicGate.MY90,
            rb.BasicGate.X90,
            rb.BasicGate.Y90,
        ),
        "Y90": (
            rb.BasicGate.Y90,
            rb.BasicGate.MX90,
            rb.BasicGate.MY90,
            rb.BasicGate.X90,
        ),
        "Y180": (
            rb.BasicGate.Y180,
            rb.BasicGate.X180,
            rb.BasicGate.Y180,
            rb.BasicGate.X180,
        ),
        "-Y90": (
            rb.BasicGate.MY90,
            rb.BasicGate.X90,
            rb.BasicGate.Y90,
            rb.BasicGate.MX90,
        ),
    }

    reduced_seq = []
    for group in seq:
        for gate in group:
            if gate in gate_map:
                reduced_seq.append(int(gate_map[gate][phase_axis]))
            else:
                phase_axis = (phase_axis + axis_map[gate]) % 4
    return reduced_seq


def test_build_seed_program_tables_matches_reference_construction() -> None:
    total_clifford_seq = [21, 6, 14, 23, 10]
    recovery_idx_by_pos = _recovery_idx_by_pos(total_clifford_seq)
    depths = np.array([1, 3, 5], dtype=np.int64)

    rand_gate_seq, prefix_len_by_depth, rec0_by_depth, rec1_by_depth = (
        rb.build_seed_program_tables(
            total_clifford_seq,
            recovery_idx_by_pos,
            depths,
        )
    )

    expected_rand = _reference_reduce_gate_seq(
        [rb.CLIFFORD_GROUP[ci] for ci in total_clifford_seq[: int(np.max(depths))]]
    )
    assert rand_gate_seq == expected_rand

    expected_prefix = [
        len(
            _reference_reduce_gate_seq(
                [rb.CLIFFORD_GROUP[ci] for ci in total_clifford_seq[:depth]]
            )
        )
        for depth in depths.tolist()
    ]
    assert prefix_len_by_depth == expected_prefix

    for depth, prefix_len, rec0, rec1 in zip(
        depths.tolist(), prefix_len_by_depth, rec0_by_depth, rec1_by_depth
    ):
        prefix_cliffords = [rb.CLIFFORD_GROUP[ci] for ci in total_clifford_seq[:depth]]
        # Recovery gates inherit the virtual-Z frame from the prefix
        full_seq = prefix_cliffords + [rb.CLIFFORD_GROUP[recovery_idx_by_pos[depth]]]
        full_reduced = _reference_reduce_gate_seq(full_seq)
        expected_rec = full_reduced[prefix_len:]
        assert len(expected_rec) <= rb.NUM_RECOVERY_SLOTS
        expected_rec += [int(rb.BasicGate.Id)] * (
            rb.NUM_RECOVERY_SLOTS - len(expected_rec)
        )
        assert [rec0, rec1] == expected_rec


def test_build_seed_program_tables_supports_zero_depth() -> None:
    total_clifford_seq = [6, 6, 6]
    recovery_idx_by_pos = _recovery_idx_by_pos(total_clifford_seq)
    depths = np.array([0, 2], dtype=np.int64)

    _, prefix_len_by_depth, rec0_by_depth, rec1_by_depth = rb.build_seed_program_tables(
        total_clifford_seq,
        recovery_idx_by_pos,
        depths,
    )

    assert prefix_len_by_depth[0] == 0
    # depth 0 accumulates identity -> recovery is identity -> both slots Id
    assert rec0_by_depth[0] == int(rb.BasicGate.Id)
    assert rec1_by_depth[0] == int(rb.BasicGate.Id)
    assert len(rec0_by_depth) == len(depths)
    assert len(rec1_by_depth) == len(depths)


def test_build_seed_program_tables_rejects_wrong_recovery_length() -> None:
    total_clifford_seq = [0, 0]
    depths = np.array([2], dtype=np.int64)

    with pytest.raises(ValueError, match="one entry per position"):
        rb.build_seed_program_tables(total_clifford_seq, [0], depths)
