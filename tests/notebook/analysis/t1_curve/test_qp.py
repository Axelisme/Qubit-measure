from __future__ import annotations

import numpy as np
import pytest
from scqubits.core.fluxonium import Fluxonium
from zcu_tools.notebook.analysis.t1_curve.Qqp import calc_qp_oper, calc_Qqp_vs_omega
from zcu_tools.simulate.fluxonium import calculate_eff_t1_fast


def test_qp_reverse_extraction_matches_scqubits_x_qp() -> None:
    params = (3.469, 0.952, 0.582)
    flux = 0.3
    temp = 0.06
    x_qp = 1e-6
    cutoff = 30
    qub_dim = 4

    fluxonium = Fluxonium(*params, flux=flux, cutoff=cutoff, truncated_dim=qub_dim)
    evals, evecs = fluxonium.eigensys(evals_count=qub_dim)
    omega = 2 * np.pi * (evals[1] - evals[0])
    sin_oper = calc_qp_oper(params, flux, return_dim=qub_dim, esys=(evals, evecs))

    t1_ns = calculate_eff_t1_fast(
        flux,
        params,
        [("t1_quasiparticle_tunneling", {"x_qp": x_qp})],
        temp,
        cutoff=cutoff,
        qub_dim=qub_dim,
    )
    extracted_q_qp = calc_Qqp_vs_omega(
        params,
        np.array([omega], dtype=np.float64),
        np.array([t1_ns], dtype=np.float64),
        np.array([sin_oper], dtype=np.complex128),
        T1errs=None,
        Temp=temp,
    )[0]

    assert extracted_q_qp == pytest.approx(1 / x_qp, rel=1e-9)


def test_qp_error_arrays_preserve_values_and_inputs() -> None:
    params = (3.469, 0.952, 0.582)
    omegas = np.array([0.8, 2.4, 5.1], dtype=np.float64)
    t1s = np.array([1200.0, 3500.0, 8100.0], dtype=np.float64)
    t1errs = np.array([0.0, 140.0, 810.0], dtype=np.float64)
    sin2_elements = np.zeros((3, 2, 2), dtype=np.complex128)
    sin2_elements[:, 0, 1] = [0.2 + 0.1j, 0.3 - 0.2j, 0.5 + 0.4j]
    inputs = (omegas, t1s, sin2_elements, t1errs)
    originals = tuple(array.copy() for array in inputs)

    no_error_values = calc_Qqp_vs_omega(
        params, omegas, t1s, sin2_elements, T1errs=None, Temp=0.035, Delta_eV=4e-4
    )
    values, errors = calc_Qqp_vs_omega(
        params, omegas, t1s, sin2_elements, T1errs=t1errs, Temp=0.035, Delta_eV=4e-4
    )

    np.testing.assert_allclose(values, no_error_values, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(errors, values * t1errs / t1s, rtol=1e-14, atol=0.0)
    assert errors[0] == 0.0
    assert np.all(values > 0.0)
    for output in (no_error_values, values, errors):
        assert output.shape == (3,)
        assert output.dtype == np.dtype(np.float64)
        assert np.all(np.isfinite(output))
        for array in inputs:
            assert not np.shares_memory(output, array)
    for array, original in zip(inputs, originals):
        np.testing.assert_array_equal(array, original)
