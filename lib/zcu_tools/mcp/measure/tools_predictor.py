"""Fixed public predictor tools for the shared measure GUI session."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext

_MODEL = {
    "type": "object",
    "properties": {
        name: {"type": "number"}
        for name in ("EJ", "EC", "EL", "flux_half", "flux_period")
    },
    "required": ["EJ", "EC", "EL", "flux_half", "flux_period"],
}
_TRANSITION = {
    "type": "array",
    "items": {"type": "integer"},
    "minItems": 2,
    "maxItems": 2,
}


def predictor_info(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    del arguments
    info = ctx.session.read_internal("predictor.info", {})
    if not info["loaded"]:
        return {"loaded": False}
    return {
        "loaded": True,
        "source": info["path"] if info["path"] is not None else "model",
        **{
            key: info[key]
            for key in ("EJ", "EC", "EL", "flux_half", "flux_period", "flux_bias")
        },
    }


def predictor_load(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    has_path = "path" in arguments
    has_model = "model" in arguments
    if has_path == has_model:
        raise ValueError("provide exactly one of path or model")

    flux_bias = arguments.get("flux_bias", 0.0)
    if has_path:
        installed = ctx.send_gui_rpc(
            "predictor.load", {"path": arguments["path"], "flux_bias": flux_bias}
        )
    else:
        installed = ctx.send_gui_rpc(
            "predictor.set_model_params",
            {**arguments["model"], "flux_bias": flux_bias},
        )
    return {
        "loaded": True,
        "source": installed["path"] if installed["path"] is not None else "model",
        **{
            key: installed[key]
            for key in ("EJ", "EC", "EL", "flux_half", "flux_period", "flux_bias")
        },
    }


def predict(ctx: MeasureToolContext, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    value = arguments["value"]
    transitions = arguments.get("transitions", [[0, 1]])
    return [
        {
            "transition": transition,
            "freq_mhz": ctx.send_gui_rpc(
                "predictor.predict",
                {
                    "device_value": value,
                    "from_level": transition[0],
                    "to_level": transition[1],
                },
            )["freq_mhz"],
        }
        for transition in transitions
    ]


def predictor_calibrate(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Calibrate the installed GUI predictor from one measured frequency."""
    transition = arguments.get("transition", [0, 1])
    return ctx.send_gui_rpc(
        "predictor.calibrate",
        {
            "device_value": arguments["value"],
            "frequency_mhz": arguments["freq_mhz"],
            "from_level": transition[0],
            "to_level": transition[1],
        },
    )


PREDICTOR_TOOLS: dict[str, dict[str, Any]] = {
    "predictor_info": {
        "handler": predictor_info,
        "description": "Read the GUI's installed predictor model and bias, or loaded=false.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "predictor_load": {
        "handler": predictor_load,
        "description": "Install a predictor from params.json path or direct model (exactly one). Optional flux_bias defaults to zero. Replaces the current predictor and returns predictor_info.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "model": _MODEL,
                "flux_bias": {"type": "number", "default": 0.0},
            },
        },
    },
    "predictor_calibrate": {
        "handler": predictor_calibrate,
        "description": "Calibrate the GUI predictor flux bias from one measured transition frequency in MHz at a native-unit device value; returns before/after bias.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "freq_mhz": {"type": "number"},
                "transition": {**_TRANSITION, "default": [0, 1]},
            },
            "required": ["value", "freq_mhz"],
        },
    },
    "predict": {
        "handler": predict,
        "description": "Predict transition frequencies in MHz at a setting in the instrument's native unit; do not write predictions as measured results.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "transitions": {
                    "type": "array",
                    "items": _TRANSITION,
                    "minItems": 1,
                    "default": [[0, 1]],
                },
            },
            "required": ["value"],
        },
    },
}


def build_predictor_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        name: {**entry, "handler": partial(entry["handler"], ctx)}
        for name, entry in PREDICTOR_TOOLS.items()
    }
