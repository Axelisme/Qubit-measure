"""Read-only simulator wire adapter: exactly one concrete qubit or named params.

Source selection and model discrimination belong to this adapter. simulate owns
named parameter decoding/projection and physical validation. Computations use
PredictorControlPort; they never read a dialog, cache or installed model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.services.predictor import (
    CalibrateFluxBiasRequest,
    PredictFreqRequest,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def _source_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> FluxoniumParams:
    qubit, direct = params.get("qubit"), params.get("params")
    if (qubit is None) == (direct is None):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "Supply exactly one source: qubit or params",
            reason="invalid_predictor_source",
        )
    if qubit is not None:
        if not isinstance(qubit, str):
            raise TypeError("Invalid validated qubit name")
        try:
            return adapter.predictor_control.params_for_qubit(qubit)
        except ValueError as exc:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_qubit"
            ) from exc
    if not isinstance(direct, dict):
        raise TypeError("Invalid validated parameter object")
    if direct.get("model") != "fluxonium":
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "params.model must be 'fluxonium'",
            reason="invalid_predictor_params",
        )
    try:
        return FluxoniumParams.from_mapping(
            {name: value for name, value in direct.items() if name != "model"}
        )
    except ValueError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_params"
        ) from exc


def _complete_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> FluxoniumParams:
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
    """Return {freq_mhz} for one complete qubit/params source at device_value.

    Device values use native units; energies use GHz; frequencies use MHz.
    INVALID_PARAMS names source conflicts, missing fields or invalid inputs.
    Queries leave components and supplied parameters unchanged.
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
    except ValueError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_params"
        ) from exc
    return {"freq_mhz": frequency}


def h_predictor_calibrate(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Return native {flux_bias_before, flux_bias_after} for one complete source.

    device_value is the measured native setpoint, frequency_mhz the measured MHz
    frequency; from_level < to_level are nonnegative integers. Source/field
    failures are INVALID_PARAMS. No component, dialog, hardware or cache writes
    occur. Numerical root failures follow the model's warning/fallback contract.
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
    except ValueError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, str(exc), reason="invalid_predictor_params"
        ) from exc
    return {"flux_bias_before": inputs.flux_bias, "flux_bias_after": result.flux_bias}


def h_predictor_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Return {model, params, missing} for one possibly incomplete source.

    Params retain null unknown fields in GHz/native units. No model is built.
    INVALID_PARAMS names source conflicts, unknown qubits or malformed direct
    inputs. The selected point is read as-is, without dialog/alias/cache access.
    """
    inputs = _source_params(adapter, params)
    return {
        "model": "fluxonium",
        "params": inputs.to_dict(),
        "missing": list(inputs.missing()),
    }
