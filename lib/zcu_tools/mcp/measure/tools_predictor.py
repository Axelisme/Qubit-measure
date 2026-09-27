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
    del ctx, arguments
    raise NotImplementedError("04 predictor_load dispatch is not implemented")


def predict(ctx: MeasureToolContext, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    del ctx, arguments
    raise NotImplementedError("04 predict dispatch is not implemented")


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
