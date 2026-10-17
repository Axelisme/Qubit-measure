"""Explicit batched density conversion at the evolution/readout seam."""

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Basis


def test_batch_populations_transform_coherences_and_reorder_target_states():
    from zcu_tools.simulate.evolution import batch_populations

    source = Basis("energy", ("g", "e"), np.eye(2, dtype=complex))
    target = Basis(
        "minus-plus",
        ("minus", "plus"),
        np.array([[1, 1], [-1, 1]], dtype=complex) / np.sqrt(2),
    )
    densities = np.array([[[0.5, 0.5], [0.5, 0.5]], [[1, 0], [0, 0]]], complex)
    original = densities.copy()
    values = batch_populations(densities, source, target)
    np.testing.assert_allclose(values, [[0, 1], [0.5, 0.5]], atol=1e-14)
    swap = Basis("e-g", ("e", "g"), np.array([[0, 1], [1, 0]], complex))
    np.testing.assert_allclose(
        batch_populations(densities, source, swap), [[0.5, 0.5], [0, 1]]
    )
    np.testing.assert_array_equal(densities, original)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [("shape", "shape"), ("finite", "finite"), ("basis", "dimensions")],
)
def test_batch_populations_reject_incompatible_inputs(invalid, message):
    from zcu_tools.simulate.evolution import batch_populations

    basis = Basis("energy", ("g", "e"), np.eye(2, dtype=complex))
    target = basis
    densities = np.eye(2, dtype=np.complex128)[None] / 2
    if invalid == "shape":
        densities = densities[0]
    elif invalid == "finite":
        densities[0, 0, 0] = np.nan
    else:
        target = Basis("one", ("g",), np.ones((1, 1), complex))
    with pytest.raises(ValueError, match=message):
        batch_populations(densities, basis, target)
