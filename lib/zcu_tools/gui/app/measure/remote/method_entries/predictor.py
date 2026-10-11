"""Predictor remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    default_integer,
    required_number,
    required_string,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "predictor.predict",
        "predictor:h_predictor_predict",
        MethodSpec(
            10.0,
            "Predict a transition frequency at a device-value setpoint. Returns "
            "{freq_mhz}.",
            (
                required_string(
                    "qubit", "Concrete qubit component name at the current point"
                ),
                required_number(
                    "device_value",
                    "Device-value setpoint in the instrument's native unit (e.g. "
                    "current in A for YOKOGS200) — NOT a flux quantum. The predictor "
                    "applies an internal value-to-flux affine conversion; passing a "
                    "flux quantum (e.g. 0.5) will silently yield a wrong frequency.",
                ),
                default_integer("from_level", 0, "From level"),
                default_integer("to_level", 1, "To level"),
            ),
        ),
    ),
    method_entry(
        "predictor.calibrate",
        "predictor:h_predictor_calibrate",
        MethodSpec(
            30.0,
            "Compute a candidate qubit flux_bias from one measured transition "
            "frequency in MHz at a device-value setpoint in native instrument units. "
            "Uses the shared PredictorService and returns {flux_bias_before, "
            "flux_bias_after}; no hardware read or write is performed.",
            (
                required_string(
                    "qubit", "Concrete qubit component name at the current point"
                ),
                required_number(
                    "device_value", "Measured setpoint in native instrument units"
                ),
                required_number(
                    "frequency_mhz", "Measured transition frequency in MHz"
                ),
                default_integer("from_level", 0, "Transition from level"),
                default_integer("to_level", 1, "Transition to level"),
            ),
        ),
    ),
    method_entry(
        "predictor.info",
        "predictor:h_predictor_info",
        MethodSpec(
            5.0,
            "Read current qubit parameters and missing fields without building a model.",
            (required_string("qubit", "Concrete qubit component name"),),
        ),
    ),
)
