"""Reduced-flux scalar/batch queries and independent spectrum reference values."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

# (EJ, EC, EL) covering normal / wide / near-integer ranges.
PARAMS = [
    (5.0, 1.0, 0.5),
    (3.0, 1.5, 0.3),
    (12.0, 0.5, 1.8),
]

# Reduced flux spans repeated, mirrored and unwrapped coordinates.
VALUE_CASES = {
    "linspace": np.linspace(-0.5, 0.5, 11),
    "unsorted_with_dups": np.array([0.2, -0.1, 0.2, 0.5, 0.0, -0.4, 0.5]),
    "wide": np.linspace(-1.2, 1.2, 9),
}

# predict_freq supports arbitrary level pairs (eigenvals go up to max(level)+5).
FREQ_TRANSITIONS = [(0, 1), (1, 2), (0, 2)]

# predict_matrix_element is intentionally restricted to (0, 1): the predictor's
# scqubits object is built with truncated_dim=2, so n_operator / phi_operator are
# 2x2 and only the (0, 1) element exists. Higher levels are a separate feature, not
# part of this batching change, so the contract locked here stays at (0, 1).
MATRIX_TRANSITIONS = [(0, 1)]


def _make_predictor(params: tuple[float, float, float]) -> FluxoniumPredictor:
    return FluxoniumPredictor(params)


# --------------------------------------------------------------------------- #
# predict_freq: scalar/array type contract + self-consistency                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("params", PARAMS)
def test_predict_freq_scalar_returns_float(params) -> None:
    p = _make_predictor(params)
    out = p.predict_freq(0.13)
    assert isinstance(out, float)


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("transition", FREQ_TRANSITIONS)
@pytest.mark.parametrize("case", list(VALUE_CASES))
def test_predict_freq_array_matches_scalar(params, transition, case) -> None:
    p = _make_predictor(params)
    vals = VALUE_CASES[case]

    arr = p.predict_freq(vals, transition)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == vals.shape

    scalar = np.array([p.predict_freq(float(v), transition) for v in vals])
    np.testing.assert_allclose(arr, scalar, atol=1e-10, rtol=0)


def test_predict_freq_golden() -> None:
    # Reference values at reduced flux 0.6 so a physics regression is caught
    # independently of the scalar/array self-consistency check above. Computed
    # with scqubits Fluxonium(cutoff=40); in MHz.
    p = FluxoniumPredictor((5.0, 1.0, 0.5))
    assert p.predict_freq(0.6, (0, 1)) == pytest.approx(1707.3748654723192, abs=1e-7)
    assert p.predict_freq(0.6, (1, 2)) == pytest.approx(3416.7599127277226, abs=1e-7)


def test_predictor_queries_frequency_curves_in_mhz() -> None:
    """The neutral query surface returns flux coordinates and transition rows."""
    predictor = FluxoniumPredictor((5.0, 1.0, 0.5))
    fluxs, frequencies = predictor.predict_frequencies_mhz(
        np.array([0.6], dtype=np.float64), ((0, 1), (1, 2))
    )
    np.testing.assert_allclose(fluxs, [0.6])
    np.testing.assert_allclose(
        frequencies, [[1707.3748654723192], [3416.7599127277226]], atol=1e-7
    )


def test_predictor_queries_matrix_curves_above_level_one() -> None:
    """Batch matrix queries retain the engine's higher-transition support."""
    predictor = FluxoniumPredictor((5.0, 1.0, 0.5))
    fluxs, magnitudes = predictor.predict_matrix_elements(
        np.array([0.6, 0.7], dtype=np.float64), ((0, 1), (0, 3)), "n"
    )
    np.testing.assert_allclose(fluxs, [0.6, 0.7])
    assert magnitudes.shape == (2, 2)
    assert np.all(magnitudes > 0)
    assert magnitudes[0, 0] == pytest.approx(0.05111118894415298, abs=1e-9)


def test_predict_freq_reverse_transition_keeps_signed_contract() -> None:
    p = _make_predictor(PARAMS[0])
    forward = p.predict_freq(0.13, (0, 1))
    reverse = p.predict_freq(0.13, (1, 0))
    assert reverse == pytest.approx(-forward, abs=1e-10)


# --------------------------------------------------------------------------- #
# predict_matrix_element: scalar/array type contract + self-consistency       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("operator", ["n", "phi"])
def test_predict_matrix_element_scalar_returns_float(params, operator) -> None:
    p = _make_predictor(params)
    out = p.predict_matrix_element(0.13, (0, 1), operator)
    assert isinstance(out, float)


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("transition", MATRIX_TRANSITIONS)
@pytest.mark.parametrize("operator", ["n", "phi"])
@pytest.mark.parametrize("case", list(VALUE_CASES))
def test_predict_matrix_element_array_matches_scalar(
    params, transition, operator, case
) -> None:
    p = _make_predictor(params)
    vals = VALUE_CASES[case]

    arr = p.predict_matrix_element(vals, transition, operator)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert arr.shape == vals.shape

    scalar = np.array(
        [p.predict_matrix_element(float(v), transition, operator) for v in vals]
    )
    np.testing.assert_allclose(arr, scalar, atol=1e-10, rtol=0)


def test_predict_matrix_element_golden() -> None:
    # Reference values at reduced flux 0.6; |<i|O|j>| is non-negative.
    p = FluxoniumPredictor((5.0, 1.0, 0.5))
    assert p.predict_matrix_element(0.6, (0, 1), "n") == pytest.approx(
        0.05111118894415298, abs=1e-9
    )
    assert p.predict_matrix_element(0.6, (0, 1), "phi") == pytest.approx(
        0.23948432, abs=1e-7
    )


def test_predict_matrix_element_invalid_operator_raises() -> None:
    p = _make_predictor(PARAMS[0])

    with pytest.raises(ValueError, match="unsupported matrix operator"):
        # tolerated: Verify unsupported operator rejection through the scalar overload.
        p.predict_matrix_element(0.1, (0, 1), "bad")  # pyright: ignore[reportCallIssue, reportArgumentType]

    with pytest.raises(ValueError, match="unsupported matrix operator"):
        # tolerated: Verify unsupported operator rejection through the array overload.
        p.predict_matrix_element(np.array([0.1]), (0, 1), "bad")  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_predict_matrix_element_rejects_levels_above_one() -> None:
    p = _make_predictor(PARAMS[0])

    with pytest.raises(ValueError, match="support only levels 0 and 1"):
        p.predict_matrix_element(0.6, (1, 2), "n")

    with pytest.raises(ValueError, match="support only levels 0 and 1"):
        p.predict_matrix_element(np.array([0.1]), (1, 2), "n")


def test_clone_is_independent() -> None:
    p = _make_predictor(PARAMS[0])
    c = p.clone()
    c.params = PARAMS[1]
    assert p.params == PARAMS[0]
    assert p.predict_freq(0.1) != pytest.approx(c.predict_freq(0.1), abs=1.0)
