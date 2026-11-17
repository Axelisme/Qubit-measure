"""Solver-free reference persistence through the public evolution API."""

from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from zcu_tools.simulate import evolution

from tests.simulate.evolution.contract.accuracy_cases import accuracy_case


def test_reference_cache_round_trip_and_missing_key_fail_without_solver(
    tmp_path: Path,
) -> None:
    assert callable(getattr(evolution, "load_reference", None)), (
        "missing public cache-only reader"
    )
    case = accuracy_case("idle_coherence")
    with closing(case.source):
        with pytest.raises(FileNotFoundError, match="reference cache miss"):
            evolution.load_reference(tmp_path, case.source, case.batch)
        density = np.broadcast_to(case.source.initial_density, (2, 3, 3)).copy()
        result = evolution.ReferenceResult(density, 1.25e-9)
        key = evolution.write_reference(tmp_path, case.source, case.batch, result)
        assert key == evolution.reference_key(case.source, case.batch)
        restored = evolution.load_reference(tmp_path, case.source, case.batch)
        assert_array_equal(restored.density, density)
        assert restored.density.dtype == np.complex128
        assert restored.convergence_error == 1.25e-9
        with pytest.raises(FileNotFoundError, match="reference cache miss"):
            evolution.load_reference(
                tmp_path,
                case.source,
                case.batch,
                coarse=evolution.SolverTolerances(1e-8, 1e-10),
            )
