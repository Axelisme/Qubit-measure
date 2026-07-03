from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.notebook.analysis.t1_curve.Qcap import charge_spectral_density
from zcu_tools.notebook.analysis.t1_curve.Qind import inductive_spectral_density


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
