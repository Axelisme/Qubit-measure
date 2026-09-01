from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import zcu_tools.experiment.v2.twotone.ckp as ckp_module
from matplotlib.figure import Figure
from zcu_tools.experiment.v2.twotone.ckp import CKP_Exp, CKP_Result
from zcu_tools.utils.fitting import FitDiagnostics, SharedFitResult, lorfunc


def _synthetic_result() -> CKP_Result:
    res_freqs = np.linspace(5994.0, 6006.0, 61)
    qub_freqs = np.linspace(3999.0, 4007.0, 321)
    g_centers = lorfunc(res_freqs, 4000.0, 0.0, 6.0, 5998.0, 1.6)
    e_centers = lorfunc(res_freqs, 4000.0, 0.0, 6.0, 6002.0, 1.6)
    signals = np.stack(
        [
            np.stack(
                [lorfunc(qub_freqs, 0.0, 0.0, 1.0, center, 0.12) for center in centers]
            )
            for centers in (g_centers, e_centers)
        ]
    ).astype(np.complex128)
    return CKP_Result(res_freqs=res_freqs, qub_freqs=qub_freqs, signals=signals)


def _controlled_shared_result() -> SharedFitResult:
    names = (
        "baseline",
        "g_slope",
        "e_slope",
        "scale",
        "g_res_freq",
        "e_res_freq",
        "width",
    )
    covariance = np.zeros((7, 7), dtype=np.float64)
    covariance[4, 4] = 4.0
    covariance[5, 5] = 9.0
    covariance[4, 5] = covariance[5, 4] = 1.0
    covariance[6, 6] = 0.25
    return SharedFitResult(
        parameter_names=names,
        values={
            "baseline": 4000.0,
            "g_slope": 0.0,
            "e_slope": 0.0,
            "scale": 6.0,
            "g_res_freq": 5998.0,
            "e_res_freq": 6002.0,
            "width": 1.6,
        },
        covariance=covariance,
        correlation=np.eye(7),
        diagnostics=FitDiagnostics(
            valid=True,
            edm=0.0,
            covariance_accurate=True,
            reached_call_limit=False,
            hesse_failed=False,
            reduced_chi_square=1.0,
        ),
        profile_intervals={},
    )


def test_ckp_public_analysis_recovers_numeric_result_and_figure() -> None:
    chi, kappa, res_freq, figure = CKP_Exp().analyze(_synthetic_result())

    assert chi == pytest.approx(2.0, abs=0.05)
    assert kappa == pytest.approx(3.2, abs=0.1)
    assert res_freq == pytest.approx(6001.2, abs=0.1)
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    title = figure.get_suptitle().lower()
    assert "nan" not in title
    assert "inf" not in title
    plt.close(figure)


def test_ckp_uncertainties_use_named_global_covariance_cross_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled = _controlled_shared_result()
    monkeypatch.setattr(ckp_module, "fit_shared", lambda *args, **kwargs: controlled)

    _, _, _, figure = CKP_Exp().analyze(_synthetic_result())

    title = figure.get_suptitle()
    assert r"\pm 1.658" in title
    assert r"\pm 1.000" in title
    assert np.all(controlled.covariance[1:3, :] == 0.0)
    assert np.all(controlled.covariance[:, 1:3] == 0.0)
    plt.close(figure)


def test_ckp_analysis_fast_fails_all_nan_signal_data() -> None:
    result = _synthetic_result()
    invalid = CKP_Result(
        res_freqs=result.res_freqs,
        qub_freqs=result.qub_freqs,
        signals=np.full_like(result.signals, np.nan + 1j * np.nan),
    )

    with pytest.raises(RuntimeError, match="requires finite signal data"):
        CKP_Exp().analyze(invalid)


def test_ckp_analysis_fast_fails_invalid_shared_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled = _controlled_shared_result()
    invalid = replace(
        controlled,
        diagnostics=replace(controlled.diagnostics, valid=False, edm=12.0),
    )
    monkeypatch.setattr(ckp_module, "fit_shared", lambda *args, **kwargs: invalid)

    with pytest.raises(RuntimeError, match="CKP shared fit is invalid.*EDM=12"):
        CKP_Exp().analyze(_synthetic_result())
