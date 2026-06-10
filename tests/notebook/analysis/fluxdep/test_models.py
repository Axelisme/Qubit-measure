import numpy as np
import pytest
from zcu_tools.notebook.analysis.fluxdep.models import (
    TransitionDict,
    compile_transitions,
    energy2linearform,
)
from zcu_tools.notebook.analysis.fluxdep.njit import energy2linearform_nb


def _check_matches(transitions, energies):
    B_py, C_py = energy2linearform(energies, transitions)
    pairs, coeffs, offsets = compile_transitions(transitions, energies.shape[1])
    B_nb, C_nb = energy2linearform_nb(energies, pairs, coeffs, offsets)
    assert B_nb.shape == B_py.shape
    assert C_nb.shape == C_py.shape
    np.testing.assert_allclose(B_nb, B_py, rtol=0, atol=1e-15)
    np.testing.assert_allclose(C_nb, C_py, rtol=0, atol=1e-15)


def test_nb_matches_python_plain_transitions():
    rng = np.random.default_rng(0)
    energies = rng.standard_normal((20, 6))
    _check_matches({"transitions": [(0, 1), (0, 2), (1, 2)]}, energies)


def test_nb_matches_python_with_sidebands():
    rng = np.random.default_rng(1)
    energies = rng.standard_normal((15, 5))
    transitions = {
        "transitions": [(0, 1)],
        "blue side": [(0, 1), (0, 2)],
        "red side": [(0, 1)],
        "r_f": 7.123,
    }
    _check_matches(transitions, energies)


def test_nb_matches_python_with_mirror():
    rng = np.random.default_rng(2)
    energies = rng.standard_normal((10, 5))
    transitions = {
        "transitions": [(0, 1)],
        "mirror": [(0, 1)],
        "mirror blue": [(0, 2)],
        "mirror red": [(0, 2)],
        "r_f": 7.1,
        "sample_f": 12.5,
    }
    _check_matches(transitions, energies)


def test_nb_matches_python_higher_order():
    rng = np.random.default_rng(3)
    energies = rng.standard_normal((8, 6))
    transitions = {
        "transitions": [(0, 1)],
        "transitions2": [(0, 2)],
        "mirror2": [(0, 2)],
        "transitions3": [(0, 3)],
        "r_f": 7.0,
        "sample_f": 10.0,
    }
    _check_matches(transitions, energies)


def test_energy2linearform_mirror_accepts_sample_f_without_r_f():
    # Regression: a `mirror` transition needs sample_f (E = sample_f - E_ji), not
    # r_f. The validation previously copy-pasted the side-band check and wrongly
    # demanded r_f, so a mirror with only sample_f raised. It must now succeed and
    # agree with the compiled / njit paths.
    rng = np.random.default_rng(4)
    energies = rng.standard_normal((6, 4))
    transitions = {
        "transitions": [(0, 1)],
        "mirror": [(0, 1)],
        "sample_f": 9.0,
    }
    _check_matches(transitions, energies)


def test_mirror_without_sample_f_or_r_f_raises_consistently():
    energies = np.zeros((4, 4), dtype=np.float64)
    transitions: TransitionDict = {"mirror": [(0, 1)]}
    with pytest.raises(ValueError, match="sample_f"):
        energy2linearform(energies, transitions)
    with pytest.raises(ValueError, match="sample_f"):
        compile_transitions(transitions, energies.shape[1])


def test_nb_empty_transitions_k_zero():
    energies = np.zeros((5, 4), dtype=np.float64)
    pairs, coeffs, offsets = compile_transitions({}, 4)
    assert pairs.shape == (0, 2)
    B, C = energy2linearform_nb(energies, pairs, coeffs, offsets)
    assert B.shape == (5, 0)
    assert C.shape == (5, 0)
