from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.notebook.analysis.t1_curve.Qcap import (
    calc_Qcap_vs_omega,
    charge_spectral_density,
)
from zcu_tools.notebook.analysis.t1_curve.Qind import (
    calc_Qind_vs_omega,
    inductive_spectral_density,
)


def test_charge_spectral_density_golden_values() -> None:
    omega = 2 * np.pi * 5.0
    temp = 0.06
    ec = 0.95

    assert charge_spectral_density(omega, temp, ec) == pytest.approx(15.48377415649128)
    assert charge_spectral_density(-omega, temp, ec) == pytest.approx(
        0.28377415649128146
    )

    actual = charge_spectral_density(np.array([omega, -omega]), temp, ec)
    np.testing.assert_allclose(actual, [15.48377415649128, 0.28377415649128146])


def test_inductive_spectral_density_golden_values() -> None:
    omega = 2 * np.pi * 5.0
    temp = 0.06
    el = 0.58

    assert inductive_spectral_density(omega, temp, el) == pytest.approx(
        1.1816564487848609
    )
    assert inductive_spectral_density(-omega, temp, el) == pytest.approx(
        0.021656448784860956
    )

    actual = inductive_spectral_density(np.array([omega, -omega]), temp, el)
    np.testing.assert_allclose(actual, [1.1816564487848609, 0.021656448784860956])


@pytest.mark.parametrize("mechanism", ["capacitive", "inductive"])
def test_q_vs_omega_propagates_error_arrays(mechanism: str) -> None:
    calc_q = calc_Qcap_vs_omega if mechanism == "capacitive" else calc_Qind_vs_omega
    params = (5.2, 0.95, 0.58)
    omegas = 2 * np.pi * np.array([1.5, 3.0, 5.0], dtype=np.float64)
    t1s = np.array([12000.0, 24000.0, 36000.0], dtype=np.float64)
    t1errs = np.array([0.0, 500.0, 1800.0], dtype=np.float64)
    elements = np.zeros((3, 2, 2), dtype=np.complex128)
    elements[:, 0, 1] = [0.2 + 0.3j, 0.5 - 0.1j, -0.4 + 0.6j]
    inputs = (omegas, t1s, elements, t1errs)
    originals = tuple(array.copy() for array in inputs)

    no_error_values = calc_q(params, omegas, t1s, elements, None, Temp=0.06)
    result = calc_q(params, omegas, t1s, elements, t1errs, Temp=0.06)

    assert isinstance(result, tuple)
    values, errors = result
    np.testing.assert_allclose(values, no_error_values)
    np.testing.assert_allclose(errors, values * t1errs / t1s)
    assert errors[0] == 0.0
    for output in (no_error_values, values, errors):
        assert output.shape == (3,)
        assert output.dtype == np.dtype(np.float64)
        for array in inputs:
            assert not np.shares_memory(output, array)
    for array, original in zip(inputs, originals, strict=True):
        np.testing.assert_array_equal(array, original)
