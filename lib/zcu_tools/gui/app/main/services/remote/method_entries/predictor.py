"""Predictor remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _int_default,
    _num,
    _num_default,
    _str,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "predictor.load",
        "predictor:_h_predictor_load",
        MethodSpec(
            30.0,
            "Install a FluxoniumPredictor from a params.json file (its fluxdep_fit "
            "section). Replaces any currently loaded predictor. Echoes the installed "
            "model: {loaded: true, path, flux_bias, flux_half, flux_period, EJ, EC, EL}.",
            (
                _str("path", "Predictor file path"),
                _num_default("flux_bias", 0.0, "Flux bias"),
            ),
            tool_name="gui_predictor_install_from_file",
        ),
    ),
    method_entry(
        "predictor.set_model_params",
        "predictor:_h_predictor_set_model_params",
        MethodSpec(
            10.0,
            "Build+install a FluxoniumPredictor directly from typed model params "
            "(no params.json). EJ/EC/EL are the Fluxonium energies in GHz "
            "(e.g. 4:1:1); flux_half/flux_period are the value->flux affine anchors "
            "in device-value units; flux_bias is the bias correction. Replaces any "
            "currently loaded predictor. Echoes the installed model: {loaded: true, "
            "path: null, flux_bias, flux_half, flux_period, EJ, EC, EL} (path is null "
            "because this predictor has no backing file).",
            (
                _num("EJ", "Josephson energy E_J (GHz)"),
                _num("EC", "Charging energy E_C (GHz)"),
                _num("EL", "Inductive energy E_L (GHz)"),
                _num(
                    "flux_half", "Half-flux (Phi0/2) value->flux anchor (device units)"
                ),
                _num("flux_period", "Flux period (device units); must be non-zero"),
                _num_default("flux_bias", 0.0, "Flux bias correction (device units)"),
            ),
            tool_name="gui_predictor_install_params",
        ),
    ),
    method_entry(
        "predictor.clear",
        "predictor:_h_predictor_clear",
        MethodSpec(
            5.0,
            "Unload the current predictor (idempotent — succeeds with no predictor "
            "loaded). Returns {loaded: false}.",
            tool_name="gui_predictor_unload",
        ),
    ),
    method_entry(
        "predictor.predict",
        "predictor:_h_predictor_predict",
        MethodSpec(
            10.0,
            "Predict a transition frequency at a device-value setpoint. Returns "
            "{freq_mhz}.",
            (
                _num(
                    "device_value",
                    "Device-value setpoint in the instrument's native unit (e.g. "
                    "current in A for YOKOGS200) — NOT a flux quantum. The predictor "
                    "applies an internal value-to-flux affine conversion; passing a "
                    "flux quantum (e.g. 0.5) will silently yield a wrong frequency.",
                ),
                _int_default("from_level", 0, "From level"),
                _int_default("to_level", 1, "To level"),
            ),
        ),
    ),
    method_entry(
        "predictor.info",
        "predictor:_h_predictor_info",
        MethodSpec(
            5.0,
            "Read the current predictor's installed model. Returns {loaded: false} "
            "when none is loaded, else {loaded: true, path, flux_bias, flux_half, "
            "flux_period, EJ, EC, EL} (path is null for an in-memory install).",
        ),
    ),
)
