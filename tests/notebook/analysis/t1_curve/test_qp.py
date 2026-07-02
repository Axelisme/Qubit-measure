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
