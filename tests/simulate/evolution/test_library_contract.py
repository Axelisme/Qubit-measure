"""Public numerical/storage coverage added after the first implementation.

These cases are first-green coverage, not historical test-first reds.
"""

import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Basis,
    BuildPlan,
    ControlPulse,
    ControlSequence,
    DensityState,
    DriveSource,
    Envelope,
    Evolution,
    EvolutionLibrary,
    IntelOpenCL,
    NumericModel,
    PlateauFamily,
)


def _family():
    b = Basis("contract", ("g", "e"), np.eye(2, dtype=complex))
    h = np.diag([0.0, 0.013]).astype(complex)
    x = np.array([[0, 0.003], [0.003, 0]], dtype=complex)
    return PlateauFamily(
        NumericModel("contract", b, h, (DriveSource("a", x),)), 8.0, (1 + 0j,)
    )


def test_lru_evicts_a_loaded_table_and_reloads_exactly(tmp_path):
    family = _family()
    library = EvolutionLibrary(tmp_path / "data", budget_bytes=600)
    backend = IntelOpenCL(tmp_path / "compiler")
    first = library.build(BuildPlan(family, -0.1, 0.1, 6), backend).published
    other = replace(family, duration_ns=10.0)
    library.build(BuildPlan(other, -0.1, 0.1, 6), backend)
    frozen = library.snapshot()
    expected = library.evaluate(family, 0.037, frozen)
    assert expected is not None
    assert 0 < library.resident_bytes <= 600
    library.evaluate(other, 0.037, frozen)
    assert library.resident_bytes <= 600
    assert first is not None
    payload = first.path.read_bytes()
    first.path.write_bytes(b"broken-after-eviction")
    try:
        with pytest.raises(ValueError, match="integrity"):
            library.evaluate(family, 0.037, frozen)
    finally:
        first.path.write_bytes(payload)
    np.testing.assert_array_equal(library.evaluate(family, 0.037, frozen), expected)


def test_table_cross_process_load_and_evaluation(tmp_path):
    family = _family()
    library = EvolutionLibrary(tmp_path / "data")
    library.build(BuildPlan(family, -0.1, 0.1, 6), IntelOpenCL(tmp_path / "compiler"))
    expected = library.evaluate(family, 0.037, library.snapshot())
    output = tmp_path / "child.npy"
    code = """
import sys
from pathlib import Path
import numpy as np
from zcu_tools.simulate.evolution import *
b=Basis("contract",("g","e"),np.eye(2,dtype=complex))
m=NumericModel("contract",b,np.diag([0.,.013]).astype(complex),(DriveSource("a",np.array([[0,.003],[.003,0]],dtype=complex)),))
l=EvolutionLibrary(Path(sys.argv[1]))
u=l.evaluate(PlateauFamily(m,8.,(1+0j,)),.037,l.snapshot())
assert u is not None
np.save(sys.argv[2],u)
"""
    lane = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            str(lane),
            "--no-sync",
            "--",
            "python",
            "-c",
            code,
            str(library.path),
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    np.testing.assert_array_equal(np.load(output), expected)


def test_direct_controls_initial_basis_and_dynamic_dependencies_do_not_collide(
    tmp_path,
):
    family = _family()
    model = family.model
    library = EvolutionLibrary(tmp_path)
    initial = DensityState(model.basis, np.full((2, 2), 0.5, dtype=complex))
    controls = ControlSequence(
        (
            ControlPulse(
                "a", 0, 8, Envelope.gaussian(sigma_ns=2, center_ns=4), 0.013, 0.3
            ),
        )
    )
    direct = Evolution(model, library=library)
    expected = direct.evolve(initial, controls, end_ns=8)
    changed_controls = ControlSequence((replace(controls.pulses[0], phase_rad=0.7),))
    actual = direct.evolve(initial, changed_controls, end_ns=8)
    assert direct.last_evaluation.origin == "direct"
    assert np.max(np.abs(actual.density - expected.density)) > 1e-3
    excited = DensityState(model.basis, np.diag([0.0, 1.0]).astype(complex))
    direct.evolve(excited, controls, end_ns=8)
    assert direct.last_evaluation.origin == "direct"
    vectors = np.array([[0, 1], [1, 0]], dtype=complex)
    changed_model = replace(model, basis=replace(model.basis, vectors=vectors))
    changed_initial = DensityState(changed_model.basis, initial.density)
    changed = Evolution(changed_model, library=library)
    changed.evolve(changed_initial, controls, end_ns=8)
    assert changed.last_evaluation.origin == "direct"
    assert replace(family, model=changed_model).identity != family.identity
    # Same descriptor family, different dependency values supplied explicitly.
    dynamic = replace(
        model,
        hamiltonian_at=lambda t: 0.01 * model.hamiltonian_ghz,
        dynamics_identity="flux-1",
    )
    first = Evolution(dynamic, library=library)
    first.evolve(initial, controls, end_ns=8)
    second = Evolution(replace(dynamic, dynamics_identity="flux-2"), library=library)
    second.evolve(initial, controls, end_ns=8)
    assert second.last_evaluation.origin == "direct"


def test_callback_solver_errors_are_not_legal_misses(tmp_path):
    model = replace(
        _family().model,
        hamiltonian_at=lambda t: np.full((2, 2), np.nan, dtype=complex),
        dynamics_identity="bad-h",
    )
    state = DensityState(model.basis, np.diag([1.0, 0.0]).astype(complex))
    with pytest.raises(ValueError, match="finite"):
        Evolution(model, library=EvolutionLibrary(tmp_path)).evolve(state, end_ns=8)
    assert not tuple((tmp_path / "direct").glob("*.npz"))
