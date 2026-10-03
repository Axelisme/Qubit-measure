"""Explicit, editable recipe registration; no discovery or hot reload."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from zcu_tools.mcp.measure.recipe_context import RecipeContext

from .drive import time_rabi, twotone_spectrum
from .lookback import lookback
from .onetone import (
    onetone_spectrum,
    onetone_spectrum_over_flux,
    onetone_spectrum_over_power,
)


@dataclass(frozen=True)
class RecipeDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[RecipeContext, dict[str, Any]], None]


_ONETONE_COMMON_PROPERTIES = {
    **{
        name: {"type": ["string", "null"], "minLength": 1}
        for name in ("reuse_tab_id", "readout_ref")
    },
    **{name: {"type": ["number", "null"]} for name in ("center_mhz", "span_mhz")},
    **{name: {"type": ["integer", "null"]} for name in ("reps", "rounds")},
}


RECIPES = (
    RecipeDefinition(
        name="time_rabi",
        description="Run one length Rabi sweep and save raw data and Primary analysis.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **{
                    name: {"type": ["string", "null"], "minLength": 1}
                    for name in (
                        "reuse_tab_id",
                        "readout_ref",
                        "drive_ref",
                        "use_reset",
                    )
                },
                **{
                    name: {"type": ["number", "null"]}
                    for name in ("frequency_mhz", "gain", "max_length_us")
                },
                **{
                    name: {"type": ["integer", "null"]}
                    for name in ("points", "reps", "rounds")
                },
            },
        },
        run=time_rabi,
    ),
    RecipeDefinition(
        name="twotone_spectrum",
        description="Run one two-tone spectrum and save raw data and Primary analysis.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **{
                    name: {"type": ["string", "null"], "minLength": 1}
                    for name in (
                        "reuse_tab_id",
                        "readout_ref",
                        "drive_ref",
                        "use_reset",
                    )
                },
                **{
                    name: {"type": ["number", "null"]}
                    for name in ("center_mhz", "span_mhz", "gain", "pulse_length_us")
                },
                **{
                    name: {"type": ["integer", "null"]}
                    for name in ("points", "reps", "rounds")
                },
            },
        },
        run=twotone_spectrum,
    ),
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
                **_ONETONE_COMMON_PROPERTIES,
                "gain": {"type": ["number", "null"]},
                "points": {"type": ["integer", "null"]},
            },
        },
        run=onetone_spectrum,
    ),
    RecipeDefinition(
        name="onetone_spectrum_over_flux",
        description="Run one frequency/physical-flux survey, save raw and Primary analysis.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **_ONETONE_COMMON_PROPERTIES,
                "gain": {"type": ["number", "null"]},
                "flux_device": {"type": ["string", "null"], "minLength": 1},
                "freq_points": {"type": ["integer", "null"]},
                "flux_points": {"type": ["integer", "null"]},
                "flux_range": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
        run=onetone_spectrum_over_flux,
    ),
    RecipeDefinition(
        name="onetone_spectrum_over_power",
        description="Run one frequency/gain survey, save raw and return its Run preview. No analysis or writeback.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **_ONETONE_COMMON_PROPERTIES,
                "freq_points": {"type": ["integer", "null"]},
                "gain_points": {"type": ["integer", "null"]},
                "gain_range": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
        run=onetone_spectrum_over_power,
    ),
)
