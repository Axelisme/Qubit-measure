"""Read-only simulator wire adapter: exactly one entry filename or named params.

Source selection and wire error projection belong to this adapter. Predictor
control owns entry-local file loading; simulate owns numeric parameter validation.
Computations never read a dialog, cache, component, alias or installed model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.services.predictor import (
    CalibrateFluxBiasRequest,
    PredictFreqRequest,
)
from zcu_tools.simulate.parameters import ModelParams, params_class
from zcu_tools.simulate.predictor import PredictorInputError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def _source_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> ModelParams:
    filename, direct = params.get("filename"), params.get("params")
    if (filename is None) == (direct is None):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "Supply exactly one source: filename or params",
            reason="invalid_predictor_source",
        )
    if filename is not None:
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
        return predictor_file.params
    if not isinstance(direct, dict):
        raise TypeError("Invalid validated parameter object")
    model = direct.get("model")
    try:
        return params_class(str(model)).from_mapping(
            {name: value for name, value in direct.items() if name != "model"}
        )
    except PredictorInputError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_params"
        ) from exc


def _complete_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> ModelParams:
    inputs = _source_params(adapter, params)
    missing = inputs.missing()
    if missing:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "Missing parameters: " + ", ".join(missing),
            reason="missing_predictor_params",
            data={"missing": list(missing)},
        )
    return inputs


def h_predictor_predict(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Return {freq_mhz} for one complete filename/params source at device_value.

    Device values use native units; energies use GHz; frequencies use MHz.
    INVALID_PARAMS names source conflicts, missing fields or invalid inputs.
    No entry or unreadable files fail with PRECONDITION_FAILED. Queries leave
    files and supplied parameters unchanged.
    """
    inputs = _complete_params(adapter, params)
    value, frm, to = params["device_value"], params["from_level"], params["to_level"]
    if (
        not isinstance(value, (int, float))
        or type(frm) is not int
        or type(to) is not int
    ):
        raise TypeError("Invalid validated predictor request")
    try:
        frequency = adapter.predictor_control.predict_freq(
            inputs, PredictFreqRequest(float(value), (frm, to))
        )
    except PredictorInputError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_params"
        ) from exc
    return {"freq_mhz": frequency}


def h_predictor_calibrate(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Return {flux_bias_before, flux_bias_after, frequency_mhz, clamped}.

    device_value is the measured native setpoint, frequency_mhz the measured MHz
    frequency; from_level < to_level are nonnegative integers. Source/field
    failures are INVALID_PARAMS. No entry or unreadable files fail with
    PRECONDITION_FAILED. No file, dialog, hardware or cache writes occur.
    The model selects smallest |bias|; unreachable targets clamp to the
    nearest frequency. Numerical failures propagate without fallback.
    """
    inputs = _complete_params(adapter, params)
    value, frequency = params["device_value"], params["frequency_mhz"]
    frm, to = params["from_level"], params["to_level"]
    if (
        not isinstance(value, (int, float))
        or not isinstance(frequency, (int, float))
        or type(frm) is not int
        or type(to) is not int
    ):
        raise TypeError("Invalid validated calibration request")
    try:
        result = adapter.predictor_control.calibrate_flux_bias(
            inputs, CalibrateFluxBiasRequest(float(value), float(frequency), (frm, to))
        )
    except PredictorInputError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_params"
        ) from exc
    return {
        "flux_bias_before": inputs.flux_bias,
        "flux_bias_after": result.flux_bias,
        "frequency_mhz": result.frequency_mhz,
        "clamped": result.clamped,
    }


def h_predictor_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Return {model, params, missing} for one possibly incomplete source.

    model is fluxonium or transmon. Params retain null unknown fields in
    GHz/native units. No model is built.
    INVALID_PARAMS names source conflicts, invalid files or malformed direct
    inputs. No entry or unreadable files fail with PRECONDITION_FAILED.
    Filename sources read the opened entry without requiring a selected point.
    Neither source reads components, aliases, dialog state or cache.
    """
    inputs = _source_params(adapter, params)
    return {
        "model": inputs.model,
        "params": inputs.to_dict(),
        "missing": list(inputs.missing()),
    }
