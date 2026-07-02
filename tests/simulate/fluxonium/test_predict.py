"""Behaviour-locking tests for FluxoniumPredictor.

These pin the public semantics of FluxoniumPredictor now that scalar and array
``predict_freq`` / ``predict_matrix_element`` both delegate to the prediction
engine. The contract that must survive that delegation:

  - scalar in -> scalar (Python float) out; array in -> ``NDArray`` out.
  - the array result is *exactly* the per-point scalar result, point by point
    (the batched path must not drift from the scalar path).
  - the flux affine (``value_to_flux``) and the bias-correction in ``calculate_bias``
    / ``update_bias`` keep their current numerical behaviour.

The cross-check is scalar-vs-array self-consistency at machine precision, plus a
few absolute golden values so a wholesale regression in the physics is caught too.
"""

from __future__ import annotations

import numpy as np
import pytest
import scqubits.settings as scq_settings
import zcu_tools.simulate.fluxonium.predict as predict_mod
from zcu_tools.meta_tool import ParamsProject, QubitParams
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

scq_settings.PROGRESSBAR_DISABLED = True

# (EJ, EC, EL) covering normal / wide / near-integer ranges.
PARAMS = [
    (5.0, 1.0, 0.5),
    (3.0, 1.5, 0.3),
    (12.0, 0.5, 1.8),
]

# A flux affine that is *not* the identity, so value<->flux conversion and the
# fold/mirror symmetry are genuinely exercised (flux_half != 0, period != 1).
FLUX_AFFINE = dict(flux_half=0.1, flux_period=0.8, flux_bias=0.03)

# Current values spanning a full period plus points beyond it, so the array path
# hits folded/mirrored fluxes and repeated values.
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
    return FluxoniumPredictor(params, **FLUX_AFFINE)


# --------------------------------------------------------------------------- #
# Coordinate transforms                                                       #
# --------------------------------------------------------------------------- #


def test_value_flux_roundtrip() -> None:
    p = _make_predictor(PARAMS[0])
    for v in (-0.3, 0.0, 0.17, 0.5):
        assert p.flux_to_value(p.value_to_flux(v)) == pytest.approx(v, abs=1e-12)


def test_value_to_flux_formula() -> None:
    p = _make_predictor(PARAMS[0])
    v = 0.25
    expected = (v + p.flux_bias - p.flux_half) / p.flux_period + 0.5
    assert p.value_to_flux(v) == pytest.approx(expected, abs=1e-12)


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
    # Absolute golden values (identity affine) so a physics regression is caught
    # independently of the scalar/array self-consistency check above. Computed
    # from the current implementation; in MHz.
    p = FluxoniumPredictor(
        (5.0, 1.0, 0.5), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )
    assert p.predict_freq(0.1, (0, 1)) == pytest.approx(1707.3748654723192, abs=1e-7)
    assert p.predict_freq(0.1, (1, 2)) == pytest.approx(3416.7599127277226, abs=1e-7)


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
    # Absolute golden values (identity affine); |<i|O|j>| is non-negative.
    p = FluxoniumPredictor(
        (5.0, 1.0, 0.5), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )
    assert p.predict_matrix_element(0.1, (0, 1), "n") == pytest.approx(
        0.05111118894415298, abs=1e-9
    )
    assert p.predict_matrix_element(0.1, (0, 1), "phi") == pytest.approx(
        0.23948432, abs=1e-7
    )


def test_predict_matrix_element_invalid_operator_raises() -> None:
    p = _make_predictor(PARAMS[0])

    with pytest.raises(ValueError, match="unsupported matrix operator"):
        p.predict_matrix_element(0.1, (0, 1), "bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unsupported matrix operator"):
        p.predict_matrix_element(
            np.array([0.1]),
            (0, 1),
            "bad",  # type: ignore[arg-type]
        )


def test_predict_matrix_element_rejects_levels_above_one() -> None:
    p = _make_predictor(PARAMS[0])

    with pytest.raises(ValueError, match="support only levels 0 and 1"):
        p.predict_matrix_element(0.1, (1, 2), "n")

    with pytest.raises(ValueError, match="support only levels 0 and 1"):
        p.predict_matrix_element(np.array([0.1]), (1, 2), "n")


# --------------------------------------------------------------------------- #
# calculate_bias / update_bias                                                #
# --------------------------------------------------------------------------- #


def test_calculate_bias_recovers_predicted_freq() -> None:
    # With zero true bias, asking calculate_bias to match the *predicted* freq at a
    # value must return a bias near 0 (the minimum-|bias| equivalent solution).
    p = FluxoniumPredictor(
        (5.0, 1.0, 0.5), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )
    cur_value = 0.12
    target = p.predict_freq(cur_value, (0, 1))
    bias = p.calculate_bias(cur_value, target, (0, 1))
    assert bias == pytest.approx(0.0, abs=1e-4)


def test_calculate_bias_detects_offset() -> None:
    # Build the predicted freq at a value shifted by a known offset, then ask
    # calculate_bias to recover that offset as the bias.
    p = FluxoniumPredictor(
        (5.0, 1.0, 0.5), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )
    cur_value = 0.12
    offset = 0.05
    # freq the qubit *actually* shows at cur_value if its true flux is shifted by offset
    observed = p.predict_freq(cur_value + offset, (0, 1))
    bias = p.calculate_bias(cur_value, observed, (0, 1))
    assert bias == pytest.approx(offset, abs=1e-3)


def test_calculate_bias_warns_when_root_find_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_root_scalar(*args, **kwargs):
        raise RuntimeError("root failed")

    monkeypatch.setattr(predict_mod, "root_scalar", fail_root_scalar)
    p = FluxoniumPredictor(
        (5.0, 1.0, 0.5), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )

    with pytest.warns(RuntimeWarning, match="Bias calibration failed"):
        bias = p.calculate_bias(0.12, 0.0, (0, 1))

    assert bias == pytest.approx(0.0, abs=1e-12)


def test_update_bias_shifts_prediction() -> None:
    # update_bias changes value_to_flux and therefore predict_freq consistently.
    p = FluxoniumPredictor(
        (5.0, 1.0, 0.5), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )
    base = p.predict_freq(0.1, (0, 1))
    p.update_bias(0.05)
    shifted = p.predict_freq(0.1, (0, 1))
    # predicting at 0.1 with bias 0.05 == predicting at 0.15 with bias 0
    q = FluxoniumPredictor(
        (5.0, 1.0, 0.5), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )
    assert shifted == pytest.approx(q.predict_freq(0.15, (0, 1)), abs=1e-9)
    assert shifted != pytest.approx(base, abs=1.0)


def test_clone_is_independent() -> None:
    p = _make_predictor(PARAMS[0])
    c = p.clone()
    c.update_bias(0.9)
    # mutating the clone's bias must not affect the original's prediction
    assert p.flux_bias != c.flux_bias
    assert p.predict_freq(0.1, (0, 1)) != pytest.approx(
        c.predict_freq(0.1, (0, 1)), abs=1.0
    )


def test_predictor_does_not_keep_shared_fluxonium_instance() -> None:
    p = _make_predictor(PARAMS[0])
    assert not hasattr(p, "fluxonium")


def test_from_file_requires_fluxdep_fit(tmp_path) -> None:
    path = tmp_path / "params.json"
    QubitParams(path).ensure_project(ParamsProject("ChipA", "Q1"))

    with pytest.raises(ValueError, match="fluxdep_fit"):
        FluxoniumPredictor.from_file(str(path))
