"""Qt-free detached predictor drafts and explicit query assembly contracts."""

import numpy as np
import pytest
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.predictor_trial import PredictorTrial
from zcu_tools.gui.session.services.predictor import (
    CalibrateOffsetRequest,
    PredictFreqRequest,
    PredictMatrixCurveRequest,
    PredictMatrixCurveResult,
    PredictorService,
)
from zcu_tools.resources.predictor_file import PredictorFlux
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.predictor import (
    CalibrationResult,
    PredictorInputError,
    PredictorParams,
)
from zcu_tools.simulate.transmon import TransmonParams


@pytest.mark.parametrize("model", ["fluxonium", "transmon"])
@pytest.mark.parametrize(
    "value",
    [
        1e-13,
        5e-324,
        1e-200,
        1.2345678901234567,
        1e200,
        1.7976931348623157e308,
        -1e200,
        None,
    ],
)
def test_energy_drafts_preserve_finite_float_values(
    model: str, value: float | None
) -> None:
    from zcu_tools.gui.session.predictor_trial import PredictorTrial

    trial = PredictorTrial(PredictorControlFacet(PredictorService()))
    params = (
        FluxoniumParams(value, None, value)
        if model == "fluxonium"
        else TransmonParams(value, None, 0.12345678901234567)
    )
    trial.set_parameters(params)
    assert trial.parameters() == params
    assert trial.energy_text(model, "EC") == ""
    name = "EJ" if model == "fluxonium" else "EJmax"
    assert trial.energy_text(model, name) == ("" if value is None else repr(value))
    for text in ("oops", "1e", "nan", "inf"):
        trial.set_energy_text(model, name, text)
        assert trial.energy_text(model, name) == text
        assert (
            trial.energy_error(model, name)
            == f"{name}: enter a finite number or leave blank."
        )
        with pytest.raises(PredictorInputError, match=name):
            trial.parameters()
    trial.set_energy_text(model, name, "0.1")
    assert trial.energy_error(model, name) is None
    trial.select_model("transmon" if model == "fluxonium" else "fluxonium")
    trial.select_model(model)
    assert trial.energy_text(model, name) == "0.1"


@pytest.fixture
def calibration_trial() -> PredictorTrial:
    trial = PredictorTrial(PredictorControlFacet(PredictorService()))
    trial.set_flux(PredictorFlux(0.2, {"coil": 2.0}, {"coil": "A"}), {"coil": 0.3})
    return trial


def test_calibration_readiness_live_mode_takes_precedence(
    calibration_trial: PredictorTrial,
) -> None:
    calibration_trial.set_energy_text("fluxonium", "EJ", "oops")
    calibration_trial.set_flux(None, {})
    block = calibration_trial.calibration_readiness(
        live_mode=True, selected=(), frequency=0.0
    )
    assert block is not None
    assert block.reason == "Live mode keeps calibration read-only."
    assert not block.is_error


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("oops", "EJ: enter a finite number or leave blank."),
        ("", "Missing parameters: EJ"),
    ],
)
def test_calibration_readiness_invalid_energy_precedes_missing_frame(
    calibration_trial: PredictorTrial, text: str, reason: str
) -> None:
    calibration_trial.set_energy_text("fluxonium", "EJ", text)
    calibration_trial.set_flux(None, {})
    block = calibration_trial.calibration_readiness(
        live_mode=False, selected=(), frequency=0.0
    )
    assert block is not None
    assert block.reason == reason
    assert block.is_error


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("frame", "Load a predictor file to set the flux frame"),
        ("device", "coil: enter a finite number"),
        ("offset", "offset: enter a finite number"),
    ],
)
def test_calibration_readiness_requires_frame_and_native_values(
    calibration_trial: PredictorTrial, source: str, reason: str
) -> None:
    if source == "frame":
        calibration_trial.set_flux(None, {})
    elif source == "device":
        calibration_trial.set_device_text("coil", "")
    else:
        calibration_trial.set_offset_text("nan")
    block = calibration_trial.calibration_readiness(
        live_mode=False, selected=(), frequency=0.0
    )
    assert block is not None
    assert block.reason == reason
    assert block.is_error


@pytest.mark.parametrize("selected", [(), ((0, 1), (0, 2))])
def test_calibration_readiness_requires_exactly_one_transition(
    calibration_trial: PredictorTrial, selected: tuple[tuple[int, int], ...]
) -> None:
    block = calibration_trial.calibration_readiness(
        live_mode=False, selected=selected, frequency=0.0
    )
    assert block is not None
    assert block.reason == "Select exactly one transition to calibrate."
    assert block.is_error


@pytest.mark.parametrize("expr", ["", "missing.path", "11", "-1"])
def test_calibration_readiness_blocks_unresolved_or_out_of_range_frequency(
    calibration_trial: PredictorTrial, expr: str
) -> None:
    from zcu_tools.gui.session.expression import EvalRef

    block = calibration_trial.calibration_readiness(
        live_mode=False, selected=((0, 1),), frequency=EvalRef(expr, float, 0, 10)
    )
    assert block is not None
    assert block.reason == "Calibration frequency is unresolved."
    assert block.is_error


class RecordingQueries(PredictorService):
    """Record requests at the control seam without executing numerical solvers."""

    def __init__(self) -> None:
        super().__init__()
        self.calibrations: list[CalibrateOffsetRequest] = []
        self.points: list[PredictFreqRequest] = []
        self.matrices: list[PredictMatrixCurveRequest] = []
        self.failure: Exception | None = None
        self.matrix_failure: Exception | None = None

    def calibrate_offset(
        self, params: PredictorParams, req: CalibrateOffsetRequest
    ) -> CalibrationResult:
        self.calibrations.append(req)
        if self.failure is not None:
            raise self.failure
        return CalibrationResult(0.03, 4321.0, True)

    def predict_freq(self, params: PredictorParams, req: PredictFreqRequest) -> float:
        self.points.append(req)
        if self.failure is not None:
            raise self.failure
        return 1234.5

    def predict_matrix_element_curve(
        self, params: PredictorParams, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        self.matrices.append(req)
        if self.matrix_failure is not None:
            raise self.matrix_failure
        if self.failure is not None:
            raise self.failure
        return PredictMatrixCurveResult(
            ("0→1",),
            req.values,
            req.values,
            np.array([[0.2 if req.operator == "n" else 0.3]]),
        )


@pytest.mark.parametrize("frequency", [1200.0, "1000 + 200"])
def test_calibration_frequency_success_and_readiness(
    calibration_trial: PredictorTrial, frequency: float | str
) -> None:
    from zcu_tools.gui.session.expression import EvalRef

    raw = (
        EvalRef(frequency, float, 0, 2000) if isinstance(frequency, str) else frequency
    )
    resolved = calibration_trial.resolve_calibration_frequency_mhz(raw)
    assert resolved.frequency_mhz == 1200.0
    assert resolved.error is None
    assert (
        calibration_trial.calibration_readiness(
            live_mode=False, selected=((0, 1),), frequency=raw
        )
        is None
    )


@pytest.mark.parametrize("expr", ["11", "-1"])
def test_calibration_frequency_retains_detailed_range_error(
    calibration_trial: PredictorTrial, expr: str, caplog: pytest.LogCaptureFixture
) -> None:
    from zcu_tools.gui.session.expression import EvalRef

    resolved = calibration_trial.resolve_calibration_frequency_mhz(
        EvalRef(expr, float, 0, 10)
    )
    assert resolved.frequency_mhz is None
    assert resolved.error == f"value {float(expr)} out of range"
    assert not caplog.records


def test_calibration_frequency_isolates_provider_failure_and_captures_lazily(
    calibration_trial: PredictorTrial, caplog: pytest.LogCaptureFixture
) -> None:
    from zcu_tools.gui.session.expression import EvalRef
    from zcu_tools.gui.session.types import ParameterContext

    calls: list[str] = []

    def unavailable() -> ParameterContext | None:
        calls.append("capture")
        raise RuntimeError("capture offline")

    raw = EvalRef("1200", float, 0, 2000)
    for live_mode, selected in [(True, ((0, 1),)), (False, ())]:
        assert (
            calibration_trial.calibration_readiness(
                live_mode=live_mode,
                selected=selected,
                frequency=raw,
                parameter_provider=unavailable,
            )
            is not None
        )
    assert (
        calibration_trial.resolve_calibration_frequency_mhz(
            1200.0, unavailable
        ).frequency_mhz
        == 1200.0
    )
    assert not calls
    resolved = calibration_trial.resolve_calibration_frequency_mhz(raw, unavailable)
    assert resolved.frequency_mhz is None
    assert resolved.error == "capture offline"
    block = calibration_trial.calibration_readiness(
        live_mode=False,
        selected=((0, 1),),
        frequency=raw,
        parameter_provider=unavailable,
    )
    assert block is not None
    assert block.reason == "Calibration frequency is unresolved."
    assert calls == ["capture", "capture"]
    assert not caplog.records


@pytest.mark.parametrize(
    "failure", [PredictorInputError("invalid query"), RuntimeError("solver offline")]
)
def test_calibration_control_failure_propagates_without_draft_changes(
    failure: Exception,
) -> None:
    service = RecordingQueries()
    service.failure = failure
    trial = PredictorTrial(PredictorControlFacet(service))
    trial.set_flux(PredictorFlux(0.2, {"coil": 2.0}, {"coil": "A"}), {"coil": 0.3})
    with pytest.raises(type(failure)) as raised:
        trial.calibrate(4300.0)
    assert raised.value is failure
    assert trial.offset_text == "0.2"
    assert trial.device_text("coil") == "0.3"


def test_calibration_adds_correction_to_local_offset_with_explicit_inputs() -> None:
    service = RecordingQueries()
    trial = PredictorTrial(PredictorControlFacet(service))
    trial.set_flux(
        PredictorFlux(0.2, {"coil": 2.0, "trim": -4.0}, {"coil": "A", "trim": "V"}),
        {"coil": 0.3, "trim": -0.4},
    )
    result = trial.calibrate(4300.0, (1, 2))
    assert result == CalibrationResult(0.03, 4321.0, True)
    assert trial.offset_text == "0.23"
    assert trial.flux().offset == 0.23
    assert len(service.calibrations) == 1
    request = service.calibrations[0]
    assert request.frequency_mhz == 4300.0
    assert request.transition == (1, 2)
    assert request.frame.offset == 0.2
    assert (
        request.frame
        == PredictorFlux(
            0.2, {"coil": 2.0, "trim": -4.0}, {"coil": "A", "trim": "V"}
        ).to_frame()
    )
    assert request.device_values == {"coil": 0.3, "trim": -0.4}


@pytest.mark.parametrize("valid", [True, False])
def test_table_batch_and_curve_coordinates_share_explicit_inputs(valid: bool) -> None:
    from zcu_tools.gui.session.predictor_trial import PredictorTrial, TrialInputs

    class CountedTrial(PredictorTrial):
        input_calls = 0

        def inputs(self, *, for_curves: bool = False) -> TrialInputs:
            self.input_calls += 1
            return super().inputs(for_curves=for_curves)

    service = RecordingQueries()
    trial = CountedTrial(PredictorControlFacet(service))
    trial.set_parameters(FluxoniumParams(5.0, 1.0, 0.5))
    flux = PredictorFlux(0.2, {"coil": 2.0, "trim": -4.0}, {"coil": "A", "trim": "V"})
    trial.set_flux(flux, {"coil": 0.3, "trim": -0.4} if valid else {})
    trial.set_axis_device("trim")
    rows = trial.row_values(((0, 1), (1, 3)), 0.7)
    assert trial.input_calls == 1
    if not valid:
        assert [(r.freq_mhz, r.mag_n, r.mag_phi) for r in rows] == [
            (None, None, None),
            (None, None, None),
        ]
        assert not service.points and not service.matrices
        return
    assert [(r.freq_mhz, r.mag_n, r.mag_phi) for r in rows] == [
        (1234.5, 0.2, 0.3),
        (1234.5, 0.2, 0.3),
    ]
    assert [req.transition for req in service.points] == [(0, 1), (1, 3)]
    assert [req.transitions for req in service.matrices] == [
        ((0, 1),),
        ((0, 1),),
        ((1, 3),),
        ((1, 3),),
    ]
    assert [req.operator for req in service.matrices] == ["n", "phi", "n", "phi"]
    for req in service.matrices:
        assert req.axis_device == "trim"
        assert req.device_values == {"coil": 0.3, "trim": -0.4}
        assert req.frame == flux.to_frame()
        np.testing.assert_array_equal(req.values, [0.7])
    coordinates = trial.curve_coordinates()
    assert coordinates.frame == flux.to_frame()
    assert coordinates.axis_device == "trim"
    assert coordinates.device_values == {"coil": 0.3, "trim": -0.4}
    assert trial.predict_frequency((1, 2)) == 1234.5
    assert service.points[-1] == PredictFreqRequest(
        flux.to_frame(), {"coil": 0.3, "trim": -0.4}, (1, 2)
    )


@pytest.mark.parametrize(
    "error", [FailedPreconditionError("unavailable"), PredictorInputError("invalid")]
)
def test_table_expected_failures_are_unavailable_per_value(error: Exception) -> None:
    from zcu_tools.gui.session.predictor_trial import PredictorTrial

    service = RecordingQueries()
    trial = PredictorTrial(PredictorControlFacet(service))
    assert [
        (r.freq_mhz, r.mag_n, r.mag_phi)
        for r in trial.row_values(((0, 1), (1, 2)), 0.0)
    ] == [
        (None, None, None),
        (None, None, None),
    ]
    assert not service.points and not service.matrices
    trial.set_flux(PredictorFlux(0.2, {"coil": 2.0}, {"coil": "A"}), {"coil": 0.3})
    service.matrix_failure = error
    (row,) = trial.row_values(((0, 1),), 0.7)
    assert (row.freq_mhz, row.mag_n, row.mag_phi) == (1234.5, None, None)
    service.failure = error
    (row,) = trial.row_values(((0, 1),), 0.7)
    assert (row.freq_mhz, row.mag_n, row.mag_phi) == (None, None, None)
    # Constant-flux frames still support scalar prediction without an axis.
    service.failure = None
    trial.set_flux(PredictorFlux(0.2, {}, {}), {})
    (row,) = trial.row_values(((0, 1),), 0.7)
    assert (row.freq_mhz, row.mag_n, row.mag_phi) == (1234.5, None, None)


@pytest.mark.parametrize("error", [ValueError("invariant"), RuntimeError("solver")])
@pytest.mark.parametrize("matrix", [False, True])
def test_table_unexpected_failures_propagate(error: Exception, matrix: bool) -> None:
    from zcu_tools.gui.session.predictor_trial import PredictorTrial

    service = RecordingQueries()
    trial = PredictorTrial(PredictorControlFacet(service))
    trial.set_flux(PredictorFlux(0.2, {"coil": 2.0}, {"coil": "A"}), {"coil": 0.3})
    if matrix:
        service.matrix_failure = error
    else:
        service.failure = error
    with pytest.raises(type(error), match=str(error)) as failure:
        trial.row_values(((0, 1),), 0.7)
    assert failure.value is error


def test_parameters_return_energy_only_trial() -> None:
    from zcu_tools.gui.session.predictor_trial import PredictorTrial

    trial = PredictorTrial(PredictorControlFacet(PredictorService()))
    params = FluxoniumParams(EJ=6.3, EC=None, EL=0.7)
    trial.set_parameters(params)
    assert trial.parameters() == params


def test_flux_formats_prefilled_and_updated_numeric_fields() -> None:
    from zcu_tools.gui.session.predictor_trial import PredictorTrial

    trial = PredictorTrial(PredictorControlFacet(PredictorService()))
    trial.set_flux(
        PredictorFlux(0.22999999999999998, {"coil": 2.0}, {"coil": "A"}),
        {"coil": 0.22999999999999998},
    )
    assert (trial.offset_text, trial.device_text("coil")) == ("0.23", "0.23")
    for value, expected in (
        (0.1472990092, "0.1472990092"),
        (0.123456789012345, "0.123456789012345"),
        (1e-7, "1e-07"),
        (0.22999999999999998, "0.23"),
        (5e-324, "4.94065645841247e-324"),
        (1e200, "1e+200"),
    ):
        trial.set_offset(value)
        trial.set_device_value("coil", value)
        assert (trial.offset_text, trial.device_text("coil")) == (expected, expected)
        assert trial.flux().offset == float(expected)
        assert trial.device_values()["coil"] == float(expected)
    trial.set_flux(
        PredictorFlux(0.2, {"coil": 2.0, "trim": -4.0}, {"coil": "A", "trim": "V"}),
        {"coil": 0.3, "extra": 9.0},
    )
    assert trial.axis_device() == "coil"
    trial.set_axis_device("trim")
    assert trial.axis_device() == "trim"
    assert trial.known_device_values() == {"coil": 0.3}
    for text in ("oops", "1e", "nan", "inf", ""):
        trial.set_device_text("trim", text)
        assert trial.device_text("trim") == text
        with pytest.raises(PredictorInputError, match="trim"):
            trial.device_values()
        trial.set_offset_text(text)
        assert trial.offset_text == text
        with pytest.raises(PredictorInputError, match="offset"):
            trial.flux()
    trial.set_flux(None, {})
    assert not trial.has_frame()
    assert trial.device_values() == {}
    with pytest.raises(PredictorInputError, match="horizontal axis"):
        trial.axis_device()
    with pytest.raises(ValueError, match="Unknown axis"):
        trial.set_axis_device("other")
    with pytest.raises(KeyError):
        trial.set_device_value("other", 0.0)


def test_missing_frame_and_device_require_explicit_source() -> None:
    from zcu_tools.gui.session.predictor_trial import PredictorTrial

    trial = PredictorTrial(PredictorControlFacet(PredictorService()))
    trial.set_parameters(TransmonParams(20.0, 0.3, 0.1))
    with pytest.raises(PredictorInputError, match="flux frame"):
        trial.predict_frequency()
    trial.set_flux(PredictorFlux(0.2, {"coil": 2.0}, {"coil": "A"}), {})
    with pytest.raises(PredictorInputError, match="coil: enter a finite number"):
        trial.predict_frequency()
