from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.experiment.v2.twotone import rb


def _simulate_acc_states(total_clifford_seq: list[int]) -> list[int]:
    state = rb.PZ
    acc_states: list[int] = [state]
    for ci in total_clifford_seq:
        for gate in rb.CLIFFORD_GROUP[ci]:
            state = rb.GATE_EFFECT_MAP[gate][state]
        acc_states.append(state)
    return acc_states


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

    expected_rand = rb.reduce_gate_seq(
        [rb.CLIFFORD_GROUP[ci] for ci in total_clifford_seq[: int(np.max(depths))]]
    )
    assert rand_gate_seq == [int(g) for g in expected_rand]

    expected_prefix = [
        len(
            rb.reduce_gate_seq(
                [rb.CLIFFORD_GROUP[ci] for ci in total_clifford_seq[:depth]]
            )
        )
        for depth in depths.tolist()
    ]
    assert prefix_len_by_depth == expected_prefix

    expected_recovery: list[int] = []
    for depth in depths.tolist():
        recovery_idx = rb.RECOVERY_INDEX[acc_states[depth]]
        recovery_seq = rb.reduce_gate_seq([rb.CLIFFORD_GROUP[recovery_idx]])
        assert len(recovery_seq) == 1
        expected_recovery.append(int(recovery_seq[0]))
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
