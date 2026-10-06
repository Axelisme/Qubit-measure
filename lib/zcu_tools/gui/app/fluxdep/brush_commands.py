"""Shared wire decoding for the existing grid and joint-cloud brush plugins."""

from __future__ import annotations

from collections.abc import Mapping

from zcu_tools.analysis.fluxdep.stroke import BrushPoint, BrushStroke, BrushTool
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.gui.remote.param_spec import (
    JsonType,
    NumberPairs,
    ParamSpec,
    validate_params,
)

# These declarations accompany the payload decoders; both plugins expose them.
TOOL_PARAMS = (
    ParamSpec("width", JsonType.NUMBER, required=False),
    ParamSpec("mode", JsonType.STRING, required=False, enum=("select", "erase")),
)
STROKE_PARAMS = (
    ParamSpec("vertices", JsonType.NUMBER_PAIRS),
    ParamSpec("width", JsonType.NUMBER),
    ParamSpec("mode", JsonType.STRING, enum=("select", "erase")),
)


def decode_parameters(
    specs: tuple[ParamSpec, ...], params: Mapping[str, object]
) -> Mapping[str, object]:
    """Validate plugin command fields, returning the shared codec's named values.

    specs declares allowed names, required fields and JSON representations.
    params may be raw JSON-compatible values or already-decoded values.
    Unknown names and codec failures raise InvalidInputError with no mutation.
    Domain ranges remain the numerical transition owner's responsibility.
    """
    unknown = params.keys() - {spec.name for spec in specs}
    if unknown:
        raise InvalidInputError(f"unknown parameters: {sorted(unknown)!r}")
    try:
        return validate_params(specs, params)
    except RemoteError as exc:
        raise InvalidInputError(str(exc)) from exc


def decode_brush_tool(params: Mapping[str, object]) -> BrushTool:
    """Decode TOOL_PARAMS to a partial update, without touching state/history.

    width is an optional numeric normalized radius; mode is select or erase.
    Either may be None. The transition rejects an empty update or bad ranges.
    Raise InvalidInputError for unknown fields or wrong representations.
    """
    values = decode_parameters(TOOL_PARAMS, params)
    width, mode = values["width"], values["mode"]
    if width is not None and not isinstance(width, float):
        raise InvalidInputError("width must be numeric")
    if mode is not None and mode not in ("select", "erase"):
        raise InvalidInputError("mode must be select or erase")
    return BrushTool(
        width, "select" if mode == "select" else "erase" if mode == "erase" else None
    )


def decode_brush_stroke(params: Mapping[str, object]) -> BrushStroke:
    """Decode STROKE_PARAMS to one native-axis gesture, without mutation.

    vertices is a nonempty sequence of finite x/y pairs or a validated
    NumberPairs. width is a normalized radius and mode is select or erase.
    Coordinates retain the caller's device-or-flux/GHz units. Domain range,
    geometry and sample-budget validation belong to the numerical kernel.
    Raise InvalidInputError for unknown fields or invalid representations.
    """
    values = decode_parameters(STROKE_PARAMS, params)
    vertices, width, mode = values["vertices"], values["width"], values["mode"]
    if not isinstance(vertices, NumberPairs) or not isinstance(width, float):
        raise InvalidInputError("stroke requires vertices and width")
    if not isinstance(mode, str) or mode not in ("select", "erase"):
        raise InvalidInputError("mode must be select or erase")
    return BrushStroke(tuple(BrushPoint(x, y) for x, y in vertices.values), width, mode)
