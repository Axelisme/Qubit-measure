from __future__ import annotations

import pytest
from zcu_tools.simulate.equation import (
    C2EC,
    EC2C,
    EL2L,
    L2EL,
    Cfreq2L,
    LC2freq,
    Lfreq2C,
    invC2EC,
    n_coeff,
    phi_coeff,
)


def test_capacitance_energy_conversion_golden_values() -> None:
    assert C2EC(100.0) == pytest.approx(0.19370229324659122)
    assert invC2EC(0.01) == pytest.approx(0.19370229324659122)
    assert EC2C(C2EC(100.0)) == pytest.approx(100.0)


def test_inductance_energy_conversion_golden_values() -> None:
    assert L2EL(100.0) == pytest.approx(1.6346151280678123)
    assert EL2L(L2EL(100.0)) == pytest.approx(100.0)


def test_lc_frequency_conversion_golden_values() -> None:
    freq = LC2freq(100.0, 100.0)

    assert freq == pytest.approx(1.5915494309189535)
    assert Lfreq2C(100.0, freq) == pytest.approx(100.0)
    assert Cfreq2L(100.0, freq) == pytest.approx(100.0)


def test_zero_point_coefficients_golden_values() -> None:
    assert n_coeff(0.95, 0.58) == pytest.approx(0.37165382126822083)
    assert phi_coeff(0.95, 0.58) == pytest.approx(1.3453379768673288)
