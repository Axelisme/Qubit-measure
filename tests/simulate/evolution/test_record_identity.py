"""Callback semantics are numerical dependencies, not just callback data labels."""

from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import (
    Basis,
    DensityState,
    Evolution,
    EvolutionLibrary,
    NumericModel,
)


def test_removing_dynamic_hamiltonian_cannot_hit_its_exact_record(tmp_path):
    basis = Basis("flags", ("g", "e"), np.eye(2, dtype=complex))
    h = np.diag([0.0, 0.013]).astype(complex)
    model = NumericModel(
        "flags",
        basis,
        np.zeros((2, 2), dtype=complex),
        hamiltonian_at=lambda t: h,
        dynamics_identity="fixed-data-v1",
    )
    state = DensityState(basis, np.full((2, 2), 0.5, dtype=complex))
    library = EvolutionLibrary(tmp_path)
    dynamic = Evolution(model, library=library).evolve(state, end_ns=8)
    static_solver = Evolution(replace(model, hamiltonian_at=None), library=library)
    actual = static_solver.evolve(state, end_ns=8)
    assert static_solver.last_evaluation.origin == "direct"
    np.testing.assert_array_equal(actual.density, state.density)
    assert np.max(np.abs(actual.density - dynamic.density)) > 0.1


def test_direct_record_keys_cannot_select_paths_outside_numeric_identity(tmp_path):
    import pytest

    library = EvolutionLibrary(tmp_path)
    density = np.eye(2, dtype=complex)/2
    with pytest.raises(ValueError, match="identity"):
        library.save_record("../outside", density)
    with pytest.raises(ValueError, match="identity"):
        library.read_record("not-a-digest")
