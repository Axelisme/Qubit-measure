"""Explicit, editable recipe registration; no discovery or hot reload."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from zcu_tools.mcp.measure.recipe_context import RecipeContext

from .lookback import lookback
from .onetone import onetone_spectrum, onetone_spectrum_over_flux


@dataclass(frozen=True)
class RecipeDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[RecipeContext, dict[str, Any]], None]


RECIPES = (
    RecipeDefinition(
        name="lookback",
        description=(
            "Run lookback once, save raw data, analyze and save figures. "
            "Wait up to 300 seconds; unfinished work continues by execution ID."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **{
                    name: {"type": ["string", "null"], "minLength": 1}
                    for name in (
                        "reuse_tab_id",
                        "readout_ref",
                        "use_reset",
                        "init_pulse_ref",
                    )
                },
                **{
                    name: {"type": ["number", "null"]}
                    for name in (
                        "frequency_mhz",
                        "readout_length_us",
                        "trigger_offset_us",
                    )
                },
                "rounds": {"type": ["integer", "null"]},
            },
        },
        run=lookback,
    ),
    RecipeDefinition(
        name="onetone_spectrum",
        description="Run one onetone spectrum and save raw data and Primary analysis.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **{
                    name: {"type": ["string", "null"], "minLength": 1}
                    for name in ("reuse_tab_id", "readout_ref")
                },
                **{
                    name: {"type": ["number", "null"]}
                    for name in ("center_mhz", "span_mhz", "gain")
                },
                **{
                    name: {"type": ["integer", "null"]}
                    for name in ("points", "reps", "rounds")
                },
            },
        },
        run=onetone_spectrum,
    ),
    RecipeDefinition(
        name="onetone_spectrum_over_flux",
        description="Run one frequency/physical-flux survey, save raw and Primary analysis.",
        input_schema={
            "type": "object", "additionalProperties": False,
            "properties": {
                **{name: {"type": ["string", "null"], "minLength": 1}
                   for name in ("reuse_tab_id", "readout_ref", "flux_device")},
                **{name: {"type": ["number", "null"]}
                   for name in ("center_mhz", "span_mhz", "gain")},
                **{name: {"type": ["integer", "null"]}
                   for name in ("freq_points", "flux_points", "reps", "rounds")},
                "flux_range": {"type": ["array", "null"], "items": {"type": "number"},
                               "minItems": 2, "maxItems": 2},
            },
        },
        run=onetone_spectrum_over_flux,
    ),
)
