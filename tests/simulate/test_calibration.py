"""Public spectrum inversion: nearest offset correction, reachable frequency and clamping.

Reference literals were computed independently with scqubits Fluxonium(cutoff=40)
 and TunableTransmon(ng=0, ncut=30), not with the predictor under test.
"""

import numpy as np
import pytest
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.transmon import TransmonParams


@pytest.fixture(params=["fluxonium", "transmon"])
def calibration_case(request):
    if request.param == "fluxonium":
        return (
            FluxoniumParams(5.0, 1.0, 0.5),
            5494.94359134362,
            134.4317658250239,
            5532.891872902891,
        )
    return (
        TransmonParams(20.0, 0.3, 0.0),
        6441.346425031242,
        1200.0,
        6613.448853504742,
    )


def test_calibration_selects_smallest_absolute_offset_correction(calibration_case):
    params, target, _, _ = calibration_case
    model = params.build_predictor()
    result = model.calculate_offset_correction(0.24, target)
    assert hasattr(result, "clamped"), (
        "Calibration must report reachability, not just a bias"
    )
    assert result.offset_correction == pytest.approx(-0.14, abs=1e-8)
    assert result.frequency_mhz == pytest.approx(target, abs=1e-6)
    assert result.clamped is False
    assert model.predict_freq(0.24 + result.offset_correction) == pytest.approx(
        result.frequency_mhz, abs=1e-6
    )
    assert params.build_predictor().predict_freq(0.24) == model.predict_freq(0.24)


@pytest.mark.parametrize(
    "target, expected_bias, bound", [(-100.0, 0.26, 2), (20000.0, -0.24, 3)]
)
def test_unreachable_calibration_returns_global_nearest_frequency(
    calibration_case, target, expected_bias, bound
):
    params = calibration_case[0]
    result = params.build_predictor().calculate_offset_correction(0.24, target)
    assert hasattr(result, "clamped"), (
        "Unreachable targets must return an explicit clamped result"
    )
    assert result.clamped is True
    assert result.offset_correction == pytest.approx(expected_bias, abs=1e-7)
    assert result.frequency_mhz == pytest.approx(calibration_case[bound], abs=1e-6)
    actual = params.build_predictor().predict_freq(0.24 + result.offset_correction)
    assert actual == pytest.approx(result.frequency_mhz, abs=1e-6)


@pytest.mark.parametrize(
    "transition, target, bias, clamped",
    [
        ((0, 3), 9300.0, -0.23674418584874902, False),
        ((0, 3), 20000.0, -0.29206368288231523, True),
    ],
)
def test_nonmonotone_transition_searches_all_roots_and_interior_extrema(
    transition, target, bias, clamped
):
    params = FluxoniumParams(5.0, 1.0, 0.5)
    result = params.build_predictor().calculate_offset_correction(
        0.4, target, transition
    )
    assert hasattr(result, "clamped"), "The full spectrum must determine the result"
    assert result.clamped is clamped
    assert result.offset_correction == pytest.approx(bias, abs=2e-6)
    assert result.frequency_mhz == pytest.approx(
        9464.04845594907 if clamped else target, abs=1e-5
    )


@pytest.mark.parametrize(
    "target, clamped", [(6613.448853504742, False), (-100.0, True), (20000.0, True)]
)
def test_flux_independent_transmon_chooses_zero_bias(target, clamped):
    model = TransmonParams(20.0, 0.3, 1.0).build_predictor()
    result = model.calculate_offset_correction(100.0, target)
    assert hasattr(result, "clamped"), "Constant spectra also report reachability"
    assert result.offset_correction == pytest.approx(0.0, abs=1e-10)
    assert result.frequency_mhz == pytest.approx(6613.448853504742, abs=1e-6)
    assert result.clamped is clamped


def test_symmetric_squid_half_flux_is_free_rotor():
    model = TransmonParams(20.0, 0.3, 0.0).build_predictor()
    # Sorted n=0, ±1, ±2, ±3 energies: 0, 1.2, 1.2, 4.8, 4.8, 10.8 GHz.
    transitions = ((0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (0, 5))
    flux, frequencies = model.predict_frequencies_mhz(np.array([0.5]), transitions)
    assert flux == pytest.approx([0.5])
    assert frequencies[:, 0] == pytest.approx(
        [1200.0, 1200.0, 0.0, 4800.0, 4800.0, 10800.0], abs=1e-7
    )


@pytest.mark.parametrize(
    "value, frequency, transition",
    [
        (float("nan"), 100.0, (0, 1)),
        (0.1, float("inf"), (0, 1)),
        (0.1, 100.0, (1, 1)),
        (0.1, 100.0, (-1, 1)),
        (0.1, 100.0, (0, 1.5)),
    ],
)
def test_calibration_rejects_invalid_input_instead_of_clamping(
    calibration_case, value, frequency, transition
):
    with pytest.raises(ValueError, match="finite|[Ll]evel|Transition"):
        calibration_case[0].build_predictor().calculate_offset_correction(
            value, frequency, transition
        )
