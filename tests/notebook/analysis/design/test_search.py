"""Anchor tests for the fluxonium design-search per-row calculators.

These lock the numeric output and the DataFrame interface (column names,
dtypes, ``esys`` object shape, NaN behaviour) of the ``calculate_*`` /
``avoid_*`` helpers so that the de-``apply`` optimisation in ``search.py`` is a
pure performance change: same scqubits calls, same per-cell results.

The expected arrays were captured from the original ``progress_apply`` /
``apply`` implementation on a fixed 9-cell grid (EJ in {4.0, 4.1, 4.2},
EC = 1.0, EL in {0.9, 1.0, 1.1}, flux = 0.5). atol = 1e-9 throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from zcu_tools.notebook.analysis.design import search as S

_ATOL = 1e-9

# --- baselines captured from the original implementation -------------------

_EXPECTED_F01 = np.array(
    [
        0.499910130908,
        0.581848996435,
        0.669229328268,
        0.473290775488,
        0.551816169027,
        0.635895061217,
        0.448020021850,
        0.523194537129,
        0.604001181015,
    ]
)

_EXPECTED_M01 = np.array(
    [
        0.138409140171,
        0.154990962123,
        0.171579629143,
        0.132388827184,
        0.148627388934,
        0.164962577713,
        0.126558482311,
        0.142429987536,
        0.158483355881,
    ]
)

_EXPECTED_T1 = np.array(
    [
        671831.5570755431,
        558234.6390127883,
        472207.0583024861,
        721108.9958025506,
        596956.7393701386,
        503103.4914646165,
        774423.6999777600,
        638809.0948964875,
        536448.9821699847,
    ]
)

# row0 eigenvalues (15,), used to lock the esys object shape/content.
_EXPECTED_EVALS_ROW0 = np.array(
    [
        1.909090087621,
        2.409000218529,
        5.838991072979,
        8.292899435632,
        11.496868144318,
        14.742167311614,
        18.022989456410,
        21.189738769171,
        24.128318331953,
        26.739722318261,
        29.038996874824,
        31.206440528655,
        33.437723983550,
        35.790981943733,
        38.244107142558,
    ]
)

# avoid_collision with avoid_freqs=[0.45], threshold=0.02 -> single hit on row6.
_EXPECTED_COLLISION = np.array(
    [False, False, False, False, False, False, True, False, False]
)
_EXPECTED_VALID_AFTER_COLLISION = np.array(
    [True, True, True, True, True, True, False, True, True]
)

_NOISE_CHANNELS = [("t1_capacitive", {}), ("t1_flux_bias_line", {})]


@pytest.fixture
def grid() -> pd.DataFrame:
    """Fixed deterministic 9-cell parameter grid (no esys yet)."""
    return S.generate_params_table((4.0, 4.3), 1.0, (0.9, 1.1), flux=0.5, precision=0.1)


@pytest.fixture
def grid_with_esys(grid: pd.DataFrame) -> pd.DataFrame:
    S.calculate_esys(grid)
    return grid


# --- generate_params_table interface ---------------------------------------


def test_generate_params_table_interface(grid: pd.DataFrame) -> None:
    assert list(grid.columns) == ["flux", "EJ", "EC", "EL", "valid"]
    assert len(grid) == 9
    assert grid["flux"].dtype == np.float64
    assert grid["EJ"].dtype == np.float64
    assert grid["EC"].dtype == np.float64
    assert grid["EL"].dtype == np.float64
    assert grid["valid"].dtype == np.bool_
    assert bool(grid["valid"].all())


# --- calculate_esys ---------------------------------------------------------


def test_calculate_esys_object_shape(grid_with_esys: pd.DataFrame) -> None:
    assert "esys" in grid_with_esys.columns
    assert grid_with_esys["esys"].dtype == object

    evals, evecs = grid_with_esys["esys"].iloc[0]
    assert isinstance(evals, np.ndarray)
    assert isinstance(evecs, np.ndarray)
    assert evals.shape == (S.DESIGN_EVALS_COUNT,)
    assert evecs.shape == (S.DESIGN_CUTOFF, S.DESIGN_EVALS_COUNT)
    assert evals.dtype == np.float64
    assert evecs.dtype == np.float64
    np.testing.assert_allclose(evals, _EXPECTED_EVALS_ROW0, atol=_ATOL)


def test_calculate_esys_requires_nothing_extra(grid: pd.DataFrame) -> None:
    # esys is the first stage; it does not depend on any other column.
    S.calculate_esys(grid)
    assert bool(grid["esys"].notna().all())


# --- calculate_f01 ----------------------------------------------------------


def test_calculate_f01_values(grid_with_esys: pd.DataFrame) -> None:
    S.calculate_f01(grid_with_esys)
    assert grid_with_esys["f01"].dtype == np.float64
    np.testing.assert_allclose(
        grid_with_esys["f01"].to_numpy(), _EXPECTED_F01, atol=_ATOL
    )


def test_calculate_f01_requires_esys(grid: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="esys"):
        S.calculate_f01(grid)


# --- calculate_m01 ----------------------------------------------------------


def test_calculate_m01_values(grid_with_esys: pd.DataFrame) -> None:
    S.calculate_m01(grid_with_esys)
    assert grid_with_esys["m01"].dtype == np.float64
    np.testing.assert_allclose(
        grid_with_esys["m01"].to_numpy(), _EXPECTED_M01, atol=_ATOL
    )


def test_calculate_m01_requires_esys(grid: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="esys"):
        S.calculate_m01(grid)


# --- calculate_t1 -----------------------------------------------------------


def test_calculate_t1_values(grid_with_esys: pd.DataFrame) -> None:
    S.calculate_t1(grid_with_esys, noise_channels=_NOISE_CHANNELS, Temp=0.05)
    assert grid_with_esys["t1"].dtype == np.float64
    # t1 spans ~5e5; use rtol so atol=1e-9 does not over-constrain magnitude.
    np.testing.assert_allclose(grid_with_esys["t1"].to_numpy(), _EXPECTED_T1, rtol=1e-9)


def test_calculate_t1_requires_esys(grid: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="esys"):
        S.calculate_t1(grid, noise_channels=_NOISE_CHANNELS, Temp=0.05)


def test_t1_default_warning_context_restores_on_exception(monkeypatch) -> None:
    import scqubits.settings as scq

    monkeypatch.setattr(scq, "T1_DEFAULT_WARNING", True)

    with pytest.raises(RuntimeError, match="boom"):
        with S._t1_default_warning_disabled():
            assert scq.T1_DEFAULT_WARNING is False
            raise RuntimeError("boom")

    assert scq.T1_DEFAULT_WARNING is True


def test_calculate_t1_restores_warning_when_fluxonium_raises(monkeypatch) -> None:
    import scqubits.core.fluxonium as fluxonium_mod
    import scqubits.settings as scq

    class FakeFluxonium:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def t1_effective(self, *_args, **_kwargs) -> float:
            assert scq.T1_DEFAULT_WARNING is False
            raise RuntimeError("boom")

    monkeypatch.setattr(scq, "T1_DEFAULT_WARNING", True)
    monkeypatch.setattr(fluxonium_mod, "Fluxonium", FakeFluxonium)
    grid = S.generate_params_table(1.0, 1.0, 1.0)
    grid["esys"] = [None]

    with pytest.raises(RuntimeError, match="boom"):
        S.calculate_t1(grid, noise_channels=_NOISE_CHANNELS, Temp=0.05)

    assert scq.T1_DEFAULT_WARNING is True


def test_calculate_dispersive_shift_is_correctly_spelled() -> None:
    assert hasattr(S, "calculate_dispersive_shift")
    assert not hasattr(S, "calculate_dipersive_shift")


# --- avoid_collision --------------------------------------------------------


def test_avoid_collision_values(grid_with_esys: pd.DataFrame) -> None:
    S.avoid_collision(grid_with_esys, avoid_freqs=[0.45], threshold=0.02)
    assert grid_with_esys["collision"].dtype == np.bool_
    np.testing.assert_array_equal(
        grid_with_esys["collision"].to_numpy(), _EXPECTED_COLLISION
    )
    np.testing.assert_array_equal(
        grid_with_esys["valid"].to_numpy(), _EXPECTED_VALID_AFTER_COLLISION
    )


def test_avoid_collision_requires_esys(grid: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="esys"):
        S.avoid_collision(grid, avoid_freqs=[0.45], threshold=0.02)


# --- calculate_snr interface (Floquet not run) ------------------------------


def test_calculate_snr_requires_esys(grid: pd.DataFrame) -> None:
    # Only the cheap guard path is exercised; the Floquet body is too slow to
    # include in an anchor test, so its numerics are intentionally not locked.
    with pytest.raises(ValueError, match="esys"):
        S.calculate_snr(grid, g=0.1, r_f=7.0, rf_w=7e-3, max_photon=70)
