"""Deterministic exact records, input identities and actual table state propagation."""

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Basis,
    ControlPulse,
    ControlSequence,
    DensityState,
    DriveSource,
    Envelope,
    Evolution,
    NumericModel,
    SolverSettings,
)


def _inputs():
    basis = Basis("reuse", ("g", "e"), np.eye(2, dtype=complex))
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    model = NumericModel(
        "reuse", basis, np.diag([0.0, 0.017]).astype(complex), (DriveSource("a", x),)
    )
    state = DensityState(basis, np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=complex))
    return model, state


def test_evolution_persists_full_lindblad_and_never_caches_sampled_state(tmp_path):
    from zcu_tools.simulate.evolution import EvolutionLibrary

    model, initial = _inputs()
    model = replace(
        model, collapse_operators=(np.array([[0, 0.1], [0, 0]], dtype=complex),)
    )
    controls = ControlSequence((ControlPulse("a", 0, 8, Envelope.flat(0.003)),))
    library = EvolutionLibrary(tmp_path)
    solver = Evolution(model, library=library)
    expected = Evolution(model).evolve(initial, controls, end_ns=8)
    first = solver.evolve(initial, controls, end_ns=8)
    assert solver.last_evaluation.origin == "direct"
    np.testing.assert_array_equal(first.density, expected.density)
    second = Evolution(model, library=EvolutionLibrary(tmp_path))
    np.testing.assert_array_equal(
        second.evolve(initial, controls, end_ns=8).density, first.density
    )
    assert second.last_evaluation.origin == "direct-record"
    assert second.last_evaluation.content_identities
    changed = replace(
        model, collapse_operators=(np.array([[0, 0.2], [0, 0]], dtype=complex),)
    )
    third = Evolution(changed, library=library)
    different = third.evolve(initial, controls, end_ns=8)
    assert third.last_evaluation.origin == "direct"
    assert np.max(np.abs(different.density - first.density)) > 1e-3
    library.clear_memory()
    np.testing.assert_array_equal(
        solver.evolve(initial, controls, end_ns=8).density, first.density
    )
    assert not library.snapshot().tables


def test_persisted_table_is_applied_to_phase_bearing_density(tmp_path):
    from zcu_tools.simulate.evolution import (
        BuildPlan,
        EvolutionLibrary,
        IntelOpenCL,
        PlateauFamily,
    )

    model, initial = _inputs()
    library = EvolutionLibrary(tmp_path / "data")
    family = PlateauFamily(model, 8, (0.003 + 0.001j,))
    library.build(BuildPlan(family, -0.1, 0.1, 6), IntelOpenCL(tmp_path / "compiler"))
    solver = Evolution(model, library=library, tables=library.snapshot())
    controls = ControlSequence(
        (ControlPulse("a", 0, 8, Envelope.flat(0.000111 + 0.000037j)),)
    )
    actual = solver.evolve(initial, controls, end_ns=8)
    expected = Evolution(model).evolve(initial, controls, end_ns=8)
    assert solver.last_evaluation.origin == "table"
    np.testing.assert_allclose(actual.density, expected.density, atol=2e-9, rtol=0)
    # Explicit continuation has no hidden quantum or carrier memory.
    continued = solver.evolve(actual, controls, end_ns=8)
    direct = Evolution(model).evolve(expected, controls, end_ns=8)
    np.testing.assert_allclose(continued.density, direct.density, atol=2e-9, rtol=0)


def test_cross_process_exact_record_readback(tmp_path):
    from zcu_tools.simulate.evolution import EvolutionLibrary

    model, initial = _inputs()
    solver = Evolution(model, library=EvolutionLibrary(tmp_path))
    state = solver.evolve(initial, end_ns=8)
    output = tmp_path / "child.npy"
    # The behavior under test is persistence across process lifetime, not a Python logic wrapper.
    code = """
from pathlib import Path
import numpy as np
from zcu_tools.simulate.evolution import *
b=Basis("reuse",("g","e"),np.eye(2,dtype=complex))
m=NumericModel("reuse",b,np.diag([0.,.017]).astype(complex),(DriveSource("a",np.array([[0,1],[1,0]],dtype=complex)),))
s=DensityState(b,np.array([[.5,.5j],[-.5j,.5]],dtype=complex))
e=Evolution(m,library=EvolutionLibrary(Path(__import__("sys").argv[1])))
r=e.evolve(s,end_ns=8)
assert e.last_evaluation.origin=="direct-record"
np.save(__import__("sys").argv[2],r.density)
"""
    lane = Path(__file__).resolve().parents[3]
    command = [
        "uv",
        "run",
        "--directory",
        str(lane),
        "--no-sync",
        "--",
        "python",
        "-c",
        code,
        str(tmp_path),
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=lane,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    np.testing.assert_array_equal(np.load(output), state.density)


def test_corrupt_direct_and_write_errors_propagate(tmp_path):
    from zcu_tools.simulate.evolution import EvolutionLibrary

    model, initial = _inputs()
    solver = Evolution(model, library=EvolutionLibrary(tmp_path / "data"))
    solver.evolve(initial, end_ns=8)
    path = next((tmp_path / "data" / "direct").glob("*.npz"))
    path.write_bytes(b"broken")
    fresh = Evolution(model, library=EvolutionLibrary(tmp_path / "data"))
    with pytest.raises(ValueError, match="(corrupt|integrity)"):
        fresh.evolve(initial, end_ns=8)
    blocked = tmp_path / "not-directory"
    blocked.write_text("blocked")
    with pytest.raises(NotADirectoryError):
        Evolution(model, library=EvolutionLibrary(blocked)).evolve(initial, end_ns=8)


def test_callbacks_require_reproducible_identity_and_still_validate_on_hit(tmp_path):
    from zcu_tools.simulate.evolution import EvolutionLibrary

    model, initial = _inputs()
    dynamic = replace(model, hamiltonian_at=lambda t: np.zeros((2, 2), dtype=complex))
    with pytest.raises(ValueError, match="identity"):
        Evolution(dynamic, library=EvolutionLibrary(tmp_path)).evolve(initial, end_ns=8)
    valid = replace(dynamic, dynamics_identity="zero-dynamic-v1")
    solver = Evolution(valid, library=EvolutionLibrary(tmp_path))
    solver.evolve(initial, end_ns=8)
    bad = ControlSequence((ControlPulse("unknown", 0, 8),))
    with pytest.raises(ValueError, match="unbound"):
        solver.evolve(initial, bad, end_ns=8)
    changed_precision = Evolution(
        valid, SolverSettings(rtol=1e-9), library=EvolutionLibrary(tmp_path)
    )
    changed_precision.evolve(initial, end_ns=8)
    assert changed_precision.last_evaluation.origin == "direct"
