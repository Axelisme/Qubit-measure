"""Publication, explicit version snapshots and bounded host residency."""

from dataclasses import replace

import numpy as np
import pytest

from zcu_tools.simulate.evolution import Basis, DriveSource, NumericModel, PlateauFamily


def _family():
    basis = Basis("storage", ("g", "e"), np.eye(2, dtype=complex))
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return PlateauFamily(NumericModel("storage", basis, 0.003*y, (DriveSource("a", 0.012*x),)), 8.0, (1+0.2j,))


def test_partial_resume_publish_interpolate_and_freeze_versions(tmp_path):
    from zcu_tools.simulate.evolution import BuildPlan, EvolutionLibrary, IntelOpenCL

    family = _family()
    plan = BuildPlan(family, -0.1, 0.1, 6)
    library = EvolutionLibrary(tmp_path / "data", budget_bytes=1024)
    backend = IntelOpenCL(tmp_path / "compiler")
    frozen = library.snapshot()
    partial = library.build(plan, backend, node_limit=2)
    assert partial.completed_nodes == 2
    assert partial.published is None
    assert not library.snapshot().tables
    full = library.build(plan, backend)
    assert full.completed_nodes == 7
    assert full.computed_nodes == 5
    assert full.published is not None
    assert library.evaluate(family, 0.037, frozen) is None
    current = library.snapshot()
    actual = library.evaluate(family, 0.037, current)
    from scipy.linalg import expm
    expected = expm(-2j*np.pi*8*(family.model.hamiltonian_ghz+0.037*family.drive_ghz))
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)
    assert library.evaluate(family, 0.101, current) is None
    assert library.evaluate(replace(family, duration_ns=10), 0.037, current) is None
    changed = replace(family, model=replace(family.model, collapse_operators=(np.diag([0., .1]).astype(complex),)))
    assert changed.identity == family.identity
    assert library.resident_bytes <= library.budget_bytes
    library.clear_memory()
    assert library.resident_bytes == 0
    reloaded = EvolutionLibrary(tmp_path / "data", budget_bytes=1)
    np.testing.assert_array_equal(reloaded.evaluate(family, 0.037, reloaded.snapshot()), actual)
    assert reloaded.resident_bytes == 0


def test_validation_failure_never_publishes_or_auto_raises_degree(tmp_path):
    from zcu_tools.simulate.evolution import BuildPlan, EvolutionLibrary, IntelOpenCL, ValidationLimits

    library = EvolutionLibrary(tmp_path / "data")
    plan = BuildPlan(_family(), -1.0, 1.0, 4, ValidationLimits(density_abs=1e-15))
    with pytest.raises(ValueError, match="validation"):
        library.build(plan, IntelOpenCL(tmp_path / "compiler"))
    assert not library.snapshot().tables


def test_corrupt_published_data_is_not_a_miss(tmp_path):
    from zcu_tools.simulate.evolution import BuildPlan, EvolutionLibrary, IntelOpenCL

    library = EvolutionLibrary(tmp_path / "data")
    library.build(BuildPlan(_family(), -0.1, 0.1, 6), IntelOpenCL(tmp_path / "compiler"))
    snapshot = library.snapshot()
    snapshot.tables[0].path.write_bytes(b"broken file")
    with pytest.raises(ValueError, match="(corrupt|integrity)"):
        library.evaluate(_family(), 0.037, snapshot)
