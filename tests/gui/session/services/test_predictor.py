"""Read-only frame/device queries through the public predictor service/facet.

Physics goldens belong to simulate. These examples verify coordinate projection,
source independence, missing-device admission and additive calibration.
"""

from typing import Literal

import numpy as np
import pytest
from zcu_tools.analysis.flux import DevicePeriod, FluxFrame
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.services import predictor as service
from zcu_tools.gui.session.services.predictor import (
    PredictFreqRequest,
    PredictorService,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

FRAME = FluxFrame(
    0.25, {"coil": DevicePeriod(-2.0, "A"), "bias": DevicePeriod(4.0, "V")}
)
PARAMS = FluxoniumParams(5.0, 1.0, 0.5)


def test_prediction_projects_all_devices_without_installing_a_model() -> None:
    control = PredictorControlFacet(PredictorService())
    values = {"coil": -0.4, "bias": 0.6}
    # 0.25 + (-0.4)/(-2) + 0.6/4 = 0.6 Phi0.
    frequency = control.predict_freq(
        PARAMS, PredictFreqRequest(frame=FRAME, device_values=values, transition=(0, 1))
    )
    assert frequency == pytest.approx(1707.3748654723192, abs=1e-7)
    assert values == {"coil": -0.4, "bias": 0.6}
    assert FRAME.offset == 0.25
    assert FluxoniumParams(5.0, 1.0, 0.5) == PARAMS


def test_prediction_names_missing_devices_instead_of_assuming_zero() -> None:
    control = PredictorControlFacet(PredictorService())
    with pytest.raises(ValueError, match="bias"):
        control.predict_freq(PARAMS, PredictFreqRequest(FRAME, {"coil": -0.4}))


def test_calibration_returns_additive_flux_correction_without_mutation() -> None:
    control = PredictorControlFacet(PredictorService())
    values = {"coil": -0.402, "bias": 0.6}  # 0.601 Phi0, including offset 0.25.
    result = control.calibrate_offset(
        PARAMS, service.CalibrateOffsetRequest(FRAME, values, 1707.3748654723192)
    )
    assert result.offset_correction == pytest.approx(-0.001, abs=1e-7)
    assert result.frequency_mhz == pytest.approx(1707.3748654723192, abs=1e-6)
    assert not result.clamped
    assert FRAME.offset == 0.25
    assert values == {"coil": -0.402, "bias": 0.6}


@pytest.mark.parametrize("kind", ["frequency", "n", "phi"])
def test_curves_scan_an_explicit_axis_with_other_devices_fixed(
    kind: Literal["frequency", "n", "phi"],
) -> None:
    control = PredictorControlFacet(PredictorService())
    values = np.array([-0.4, -0.6, -0.4])
    fixed = {"bias": 0.6}
    if kind == "frequency":
        result = control.predict_freq_curve(
            PARAMS,
            service.PredictCurveRequest(FRAME, "coil", values, fixed, ((0, 1), (0, 3))),
        )
        assert result.freqs_mhz.shape == (2, 3)
        assert result.freqs_mhz[0, 0] == pytest.approx(1707.3748654723192, abs=1e-7)
    else:
        result = control.predict_matrix_element_curve(
            PARAMS,
            service.PredictMatrixCurveRequest(
                FRAME, "coil", values, fixed, ((0, 1), (0, 3)), kind
            ),
        )
        assert result.mags.shape == (2, 3)
        assert np.all(result.mags > 0)
    assert result.labels == ("0→1", "0→3")
    np.testing.assert_allclose(result.fluxs, [0.6, 0.7, 0.6])
    np.testing.assert_array_equal(result.values, values)
    np.testing.assert_array_equal(values, [-0.4, -0.6, -0.4])
    assert fixed == {"bias": 0.6}


@pytest.mark.parametrize("transition", [(1, 1), (-1, 1), (2, 1)])
def test_calibration_rejects_nonincreasing_or_negative_levels(
    transition: tuple[int, int],
) -> None:
    with pytest.raises(InvalidInputError, match="from-level"):
        PredictorService().calibrate_offset(
            PARAMS,
            service.CalibrateOffsetRequest(
                FRAME, {"coil": 0.0, "bias": 0.0}, 1707.0, transition
            ),
        )


def test_partial_energy_inputs_refuse_calculation() -> None:
    control = PredictorControlFacet(PredictorService())
    partial = FluxoniumParams(4.1, 1.2, None)
    with pytest.raises(ValueError, match="EL"):
        control.predict_freq(
            partial, PredictFreqRequest(FRAME, {"coil": 0.0, "bias": 0.0})
        )
    with pytest.raises(ValueError, match="EL"):
        control.calibrate_offset(
            partial,
            service.CalibrateOffsetRequest(FRAME, {"coil": 0.0, "bias": 0.0}, 512.3),
        )


@pytest.mark.parametrize("transitions", [(), ((-1, 1),), ((2, 1),)])
def test_curves_reject_invalid_transitions(
    transitions: tuple[tuple[int, int], ...],
) -> None:
    control = PredictorControlFacet(PredictorService())
    with pytest.raises(ValueError, match="transitions|Transition"):
        control.predict_freq_curve(
            PARAMS,
            service.PredictCurveRequest(
                FRAME, "coil", np.array([0.1]), {"bias": 0.0}, transitions
            ),
        )
    with pytest.raises(ValueError, match="transitions|Transition"):
        control.predict_matrix_element_curve(
            PARAMS,
            service.PredictMatrixCurveRequest(
                FRAME, "coil", np.array([0.1]), {"bias": 0.0}, transitions, "n"
            ),
        )


@pytest.mark.parametrize("kind", ["frequency", "n"])
@pytest.mark.parametrize(
    "axis,fixed,message", [("absent", {"bias": 0.0}, "absent"), ("coil", {}, "bias")]
)
def test_curves_reject_unknown_axes_and_missing_fixed_devices(
    kind: str, axis: str, fixed: dict[str, float], message: str
) -> None:
    control = PredictorControlFacet(PredictorService())
    if kind == "frequency":
        with pytest.raises(ValueError, match=message):
            control.predict_freq_curve(
                PARAMS,
                service.PredictCurveRequest(
                    FRAME, axis, np.array([0.1]), fixed, ((0, 1),)
                ),
            )
    else:
        with pytest.raises(ValueError, match=message):
            control.predict_matrix_element_curve(
                PARAMS,
                service.PredictMatrixCurveRequest(
                    FRAME, axis, np.array([0.1]), fixed, ((0, 1),), "n"
                ),
            )
