"""Read-only simulator wire adapter: exactly one entry filename or named params.

Source selection and wire error projection belong to this adapter. Predictor
control owns entry-local file loading; simulate owns numeric parameter validation.
Computations never read a dialog, cache, component, alias or installed model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, TypedDict

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.services.predictor import (
    CalibrateOffsetRequest,
    PredictFreqRequest,
)
from zcu_tools.resources.predictor_file import PredictorFlux, PredictorFluxValues
from zcu_tools.simulate.parameters import ModelName, ModelParams, params_class
from zcu_tools.simulate.predictor import PredictorInputError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


class PredictorFilesReply(TypedDict):
    """files lists sorted *.json filenames relative to the opened entry's predictors/.

    An unopened entry or absent directory yields an empty list. Content is not
    loaded or validated, so listing a filename does not prove it can be used.
    """

    files: list[str]


class PredictionReply(TypedDict):
    """freq_mhz is the requested transition frequency in MHz."""

    freq_mhz: float


class OffsetCalibrationReply(TypedDict):
    """Candidate calibration; no source is updated.

    offset_correction is additive Phi0 for the caller's current offset.
    frequency_mhz is the actual attained MHz. clamped is true only when the
    target was unattainable; nearest-frequency ties use minimum correction.
    """

    offset_correction: float
    frequency_mhz: float
    clamped: bool


class PredictorInfoReply(TypedDict):
    """Independent input description without building a numerical model.

    model is fluxonium or transmon. params holds that model's nullable GHz
    energies/asymmetry. missing lists unknown energy/asymmetry fields in order.
    flux is recorded or explicit coordinates, or None for inline info without flux.
    """

    model: ModelName
    params: Mapping[str, object]
    missing: list[str]
    flux: PredictorFluxValues | None


def h_predictor_list_files(
    adapter: RemoteControlAdapter, _params: Mapping[str, object]
) -> PredictorFilesReply:
    """Return {files: [filename, ...]} without requiring a selected point.

    params is empty. Filenames include .json and are relative to the opened
    entry's predictors directory, suitable for predictor.info/predict/calibrate.
    No entry or absent directory returns {files: []}. Other I/O failures
    propagate. Listing neither reads file contents nor changes session state.
    """
    return {"files": list(adapter.predictor_control.list_files())}


@dataclass(frozen=True)
class _SourceInputs:
    params: ModelParams
    flux: PredictorFlux | None


def _source_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> _SourceInputs:
    filename, direct = params.get("filename"), params.get("params")
    if (filename is None) == (direct is None):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "Supply exactly one source: filename or params",
            reason="invalid_predictor_source",
        )
    if filename is not None:
        if "flux" in params:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                "filename uses its recorded flux; do not supply flux",
                reason="invalid_predictor_source",
            )
        if not isinstance(filename, str):
            raise TypeError("Invalid validated predictor filename")
        try:
            predictor_file = adapter.predictor_control.load_file(filename)
        except (InvalidInputError, ValueError) as exc:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_file"
            ) from exc
        except FailedPreconditionError as exc:
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED, str(exc), reason="no_entry"
            ) from exc
        except OSError as exc:
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED,
                str(exc),
                reason="predictor_file_unavailable",
            ) from exc
        return _SourceInputs(predictor_file.params, predictor_file.flux)
    if not isinstance(direct, dict):
        raise TypeError("Invalid validated parameter object")
    model = direct.get("model")
    try:
        inputs = params_class(str(model)).from_mapping(
            {name: value for name, value in direct.items() if name != "model"}
        )
    except PredictorInputError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_params"
        ) from exc
    return _SourceInputs(inputs, _inline_flux(params))


def _inline_flux(params: Mapping[str, object]) -> PredictorFlux | None:
    if "flux" not in params:
        return None
    value = params["flux"]
    try:
        if not isinstance(value, dict):
            raise ValueError("flux must be an offset/period/unit object")
        return PredictorFlux.from_mapping(value)
    except ValueError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_flux"
        ) from exc


def _complete_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> tuple[ModelParams, PredictorFlux]:
    inputs = _source_params(adapter, params)
    missing = inputs.params.missing()
    if missing:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "Missing parameters: " + ", ".join(missing),
            reason="missing_predictor_params",
            data={"missing": list(missing)},
        )
    if inputs.flux is None:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "Inline prediction/calibration requires flux",
            reason="missing_predictor_flux",
        )
    return inputs.params, inputs.flux


def _device_values(params: Mapping[str, object]) -> dict[str, float]:
    values = params["device_values"]
    if not isinstance(values, dict):
        raise TypeError("Invalid validated device_values object")
    decoded: dict[str, float] = {}
    for name, value in values.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"Invalid device {name!r} value: expected a finite number",
                reason="invalid_predictor_params",
            )
        try:
            number = float(value)
        except OverflowError as exc:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"Invalid device {name!r} value: expected a finite number",
                reason="invalid_predictor_params",
            ) from exc
        if not isfinite(number):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"Invalid device {name!r} value: expected a finite number",
                reason="invalid_predictor_params",
            )
        decoded[name] = number
    return decoded


def h_predictor_predict(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PredictionReply:
    """Return MHz for params with filename or inline energy params plus flux.

    device_values supplies every frame device in its declared native unit.
    from_level/to_level select nonnegative integer levels (default 0/1).
    Missing devices are named errors, never read from hardware or cache.
    Energies use GHz; frequencies use MHz.
    INVALID_PARAMS names source conflicts, missing fields or invalid inputs.
    No entry or unreadable files fail with PRECONDITION_FAILED. Queries leave
    files and supplied parameters unchanged.
    """
    inputs, flux = _complete_params(adapter, params)
    values = _device_values(params)
    frm, to = params["from_level"], params["to_level"]
    if type(frm) is not int or type(to) is not int:
        raise TypeError("Invalid validated predictor request")
    try:
        frequency = adapter.predictor_control.predict_freq(
            inputs, PredictFreqRequest(flux.to_frame(), values, (frm, to))
        )
    except PredictorInputError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_params"
        ) from exc
    return {"freq_mhz": frequency}


def h_predictor_calibrate(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> OffsetCalibrationReply:
    """Return additive Phi0 correction, actual MHz and reachability for params.

    Supply filename or inline energy params plus flux, device_values in every
    frame device's native unit, measured frequency_mhz and nonnegative integer
    from_level < to_level (default 0/1). Add offset_correction to the caller's
    current offset. Source/field failures are INVALID_PARAMS. Missing entry or
    unreadable files fail with PRECONDITION_FAILED. No source or state changes.
    The model chooses the smallest absolute correction. Unreachable targets
    clamp to nearest MHz; numerical failures propagate without fallback.
    """
    inputs, flux = _complete_params(adapter, params)
    values, frequency = _device_values(params), params["frequency_mhz"]
    frm, to = params["from_level"], params["to_level"]
    if (
        not isinstance(frequency, (int, float))
        or type(frm) is not int
        or type(to) is not int
    ):
        raise TypeError("Invalid validated calibration request")
    try:
        result = adapter.predictor_control.calibrate_offset(
            inputs,
            CalibrateOffsetRequest(
                flux.to_frame(), values, float(frequency), (frm, to)
            ),
        )
    except PredictorInputError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_params"
        ) from exc
    return {
        "offset_correction": result.offset_correction,
        "frequency_mhz": result.frequency_mhz,
        "clamped": result.clamped,
    }


def h_predictor_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PredictorInfoReply:
    """Return model, energy params, missing energy fields and flux for params.

    Supply exactly one filename or inline energy params source. Inline flux
    is optional: omitted returns None, supplied is validated and returned.
    Filename uses recorded flux and rejects an additional flux argument.
    model is fluxonium or transmon; unknown energies stay null. No model is built.
    INVALID_PARAMS names source conflicts, invalid files or malformed direct
    inputs. No entry or unreadable files fail with PRECONDITION_FAILED.
    Filename sources read the opened entry without requiring a selected point.
    Neither source reads components, aliases, dialog state or cache.
    """
    inputs = _source_params(adapter, params)
    return {
        "model": inputs.params.model,
        "params": inputs.params.to_dict(),
        "missing": list(inputs.params.missing()),
        "flux": inputs.flux.to_dict() if inputs.flux is not None else None,
    }
