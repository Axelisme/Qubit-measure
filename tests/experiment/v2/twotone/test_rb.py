from __future__ import annotations

import numpy as np
import pytest
from typing_extensions import Sequence
from zcu_tools.experiment.v2.twotone import rb


def _simulate_acc_states(total_clifford_seq: list[int]) -> list[int]:
    state = rb.PZ
    acc_states: list[int] = [state]
    for ci in total_clifford_seq:
        for gate in rb.CLIFFORD_GROUP[ci]:
            state = rb.GATE_EFFECT_MAP[gate][state]
        acc_states.append(state)
    return acc_states


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
    acc_states = _simulate_acc_states(total_clifford_seq)
    depths = np.array([1, 3, 5], dtype=np.int64)

    rand_gate_seq, prefix_len_by_depth, recovery_gate_by_depth = (
        rb.build_seed_program_tables(
            total_clifford_seq,
            acc_states,
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

    expected_recovery: list[int] = []
    for depth in depths.tolist():
        prefix_cliffords = [rb.CLIFFORD_GROUP[ci] for ci in total_clifford_seq[:depth]]
        recovery_idx = rb.RECOVERY_INDEX[acc_states[depth]]
        # Recovery gate inherits phase from prefix
        full_seq = prefix_cliffords + [rb.CLIFFORD_GROUP[recovery_idx]]
        full_reduced = _reference_reduce_gate_seq(full_seq)
        expected_recovery.append(full_reduced[-1])
    assert recovery_gate_by_depth == expected_recovery


def test_build_seed_program_tables_supports_zero_depth() -> None:
    total_clifford_seq = [6, 6, 6]
    acc_states = _simulate_acc_states(total_clifford_seq)
    depths = np.array([0, 2], dtype=np.int64)

    _, prefix_len_by_depth, recovery_gate_by_depth = rb.build_seed_program_tables(
        total_clifford_seq,
        acc_states,
        depths,
    )

    assert prefix_len_by_depth[0] == 0
    assert len(recovery_gate_by_depth) == len(depths)


def test_build_seed_program_tables_rejects_multi_gate_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rb, "RECOVERY_INDEX", (9, 14, 21, 19, 11, 6))

    total_clifford_seq = [0, 0]
    acc_states = _simulate_acc_states(total_clifford_seq)
    depths = np.array([0], dtype=np.int64)

    with pytest.raises(
        ValueError,
        match="RB recovery Clifford must map to exactly one physical BasicGate",
    ):
        rb.build_seed_program_tables(total_clifford_seq, acc_states, depths)
