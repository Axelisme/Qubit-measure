"""Read-only predictor handlers using explicit concrete qubit names.

The independent params wire alternative and source-error contract are completed
by the dedicated MCP slice. There is no installed session model or write route.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING

from zcu_tools.gui.session.services.predictor import (
    CalibrateFluxBiasRequest,
    PredictFreqRequest,
)

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def h_predictor_predict(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Return freq_mhz at native device_value for the supplied concrete qubit."""
    qubit, value = params["qubit"], params["device_value"]
    frm, to = params["from_level"], params["to_level"]
    if (
        not isinstance(qubit, str)
        or not isinstance(value, (int, float))
        or type(frm) is not int
        or type(to) is not int
    ):
        raise TypeError("Invalid validated predictor request")
    inputs = adapter.predictor_control.params_for_qubit(qubit)
    return {
        "freq_mhz": adapter.predictor_control.predict_freq(
            inputs, PredictFreqRequest(float(value), (frm, to))
        )
    }


def h_predictor_calibrate(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Return candidate native bias without changing the qubit or any dialog."""
    qubit, value, frequency = (
        params["qubit"],
        params["device_value"],
        params["frequency_mhz"],
    )
    frm, to = params["from_level"], params["to_level"]
    if (
        not isinstance(qubit, str)
        or not isinstance(value, (int, float))
        or not isinstance(frequency, (int, float))
        or type(frm) is not int
        or type(to) is not int
    ):
        raise TypeError("Invalid validated calibration request")
    inputs = adapter.predictor_control.params_for_qubit(qubit)
    result = adapter.predictor_control.calibrate_flux_bias(
        inputs, CalibrateFluxBiasRequest(float(value), float(frequency), (frm, to))
    )
    return {"flux_bias_before": inputs.flux_bias, "flux_bias_after": result.flux_bias}


def h_predictor_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    """Return named GHz/native inputs and missing fields for one concrete qubit."""
    qubit = params["qubit"]
    if not isinstance(qubit, str):
        raise TypeError("Invalid validated qubit name")
    inputs = adapter.predictor_control.params_for_qubit(qubit)
    return {
        "model": "fluxonium",
        "params": asdict(inputs),
        "missing": list(inputs.missing()),
    }
