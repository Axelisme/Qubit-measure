"""Recover standard loss parameters from external scqubits ns data."""

from __future__ import annotations

import numpy as np
import pytest
from scqubits.core.fluxonium import Fluxonium
from zcu_tools.notebook.analysis.t1_curve.fit import T1FitParams, fit_t1_noise_params
from zcu_tools.notebook.analysis.t1_curve.Qcap import calc_Qcap_vs_omega
from zcu_tools.notebook.analysis.t1_curve.Qind import calc_Qind_vs_omega
from zcu_tools.notebook.analysis.t1_curve.Qqp import calc_Qqp_vs_omega

_PARAMS = (3.469, 0.952, 0.582)


@pytest.mark.parametrize("mechanism", ["capacitive", "inductive", "quasiparticle"])
@pytest.mark.parametrize("temp", [0.02, 0.08])
def test_pointwise_loss_recovers_raw_scqubits_truth(
    mechanism: str, temp: float
) -> None:
    fluxs = np.array([0.11, 0.27, 0.43])
    qubit = Fluxonium(*_PARAMS, flux=0, cutoff=30, truncated_dim=4)
    omegas, times, elements = [], [], []
    q = 2.7e6
    for flux in fluxs:
        qubit.flux = flux
        esys = qubit.eigensys(evals_count=4)
        omegas.append(2 * np.pi * (esys[0][1] - esys[0][0]))
        if mechanism == "capacitive":
            times.append(qubit.t1_capacitive(Q_cap=q, T=temp, total=True, esys=esys))
            elements.append(qubit.n_operator(energy_esys=esys))
        elif mechanism == "inductive":
            times.append(qubit.t1_inductive(Q_ind=q, T=temp, total=True, esys=esys))
            elements.append(qubit.phi_operator(energy_esys=esys))
        else:
            times.append(
                qubit.t1_quasiparticle_tunneling(
                    x_qp=1 / q, T=temp, total=True, esys=esys
                )
            )
            elements.append(
                qubit.sin_phi_operator(alpha=0.5, beta=np.pi * flux, energy_esys=esys)
            )
    extract = {
        "capacitive": calc_Qcap_vs_omega,
        "inductive": calc_Qind_vs_omega,
        "quasiparticle": calc_Qqp_vs_omega,
    }[mechanism]
    t1s = np.array(times, dtype=np.float64)
    values, errors = extract(
        _PARAMS,
        np.array(omegas),
        t1s,
        np.array(elements, dtype=np.complex128),
        T1errs=0.03 * t1s,
        Temp=temp,
    )
    np.testing.assert_allclose(values, q, rtol=1e-9)
    np.testing.assert_allclose(errors, 0.03 * q, rtol=1e-9)


def test_combined_fit_recovers_external_loss_parameters() -> None:
    fluxs = np.linspace(0.05, 0.47, 21)
    truth = T1FitParams(Temp=0.055, Q_cap=1.6e6, Q_ind=4e8, x_qp=1.2e-6)
    noise = [
        ("t1_capacitive", {"Q_cap": truth.Q_cap}),
        ("t1_inductive", {"Q_ind": truth.Q_ind}),
        ("t1_quasiparticle_tunneling", {"x_qp": truth.x_qp}),
    ]
    qubit = Fluxonium(*_PARAMS, flux=0, cutoff=30, truncated_dim=4)
    times = []
    for flux in fluxs:
        qubit.flux = flux
        times.append(
            qubit.t1_effective(
                noise_channels=noise,
                common_noise_options={"T": truth.Temp, "total": True},
            )
        )
    t1s = np.array(times, dtype=np.float64)
    result = fit_t1_noise_params(
        fluxs,
        t1s,
        _PARAMS,
        init=T1FitParams(Temp=truth.Temp, Q_cap=2.4e6, Q_ind=2.5e8, x_qp=8e-7),
        fixed=("Temp",),
        cutoff=30,
        qub_dim=4,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    assert result.success
    assert result.params.Q_cap == pytest.approx(truth.Q_cap, rel=1e-6)
    assert result.params.Q_ind == pytest.approx(truth.Q_ind, rel=1e-6)
    assert result.params.x_qp == pytest.approx(truth.x_qp, rel=1e-6)
    np.testing.assert_allclose(result.model_T1s, t1s, rtol=1e-8)
