"""Explicit, editable recipe registration; no discovery or hot reload."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from zcu_tools.mcp.measure.recipe_context import RecipeContext

from .coherence import t1, t2echo, t2ramsey
from .drive import amplitude_rabi, time_rabi, twotone_spectrum
from .ge import singleshot_ge
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
        name="singleshot_ge",
        description="Run GE single-shot calibration with Primary and Post analysis.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **{
                    name: {"type": ["string", "null"], "minLength": 1}
                    for name in (
                        "reuse_tab_id", "readout_ref", "pi_ref",
                        "use_reset", "init_pulse_ref",
                    )
                },
                "shots": {"type": ["integer", "null"]},
            },
        },
        run=singleshot_ge,
    ),
    *(
        RecipeDefinition(
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    **{
                        key: {"type": ["string", "null"], "minLength": 1}
                        for key in (
                            "reuse_tab_id",
                            "readout_ref",
                            "use_reset",
                            *pulse_refs,
                        )
                    },
                    **{
                        key: {"type": ["number", "null"]}
                        for key in ("max_delay_us", "detune_ratio")
                    },
                    **{
                        key: {"type": ["integer", "null"]}
                        for key in ("points", "reps", "rounds")
                    },
                },
            },
            run=run,
        )
        for name, description, pulse_refs, run in (
            (
                "t2ramsey",
                "Run one calibrated Ramsey delay sweep.",
                ("pi2_ref",),
                t2ramsey,
            ),
            (
                "t2echo",
                "Run one calibrated Echo total-delay sweep.",
                ("pi_ref", "pi2_ref"),
                t2echo,
            ),
        )
    ),
    RecipeDefinition(
        name="t1",
        description="Run one calibrated T1 delay sweep and save raw and Primary analysis.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **{
                    name: {"type": ["string", "null"], "minLength": 1}
                    for name in ("reuse_tab_id", "readout_ref", "pi_ref", "use_reset")
                },
                "max_delay_us": {"type": ["number", "null"]},
                **{
                    name: {"type": ["integer", "null"]}
                    for name in ("points", "reps", "rounds")
                },
            },
        },
        run=t1,
    ),
    RecipeDefinition(
        name="amplitude_rabi",
        description="Run one gain Rabi sweep and save raw data and Primary analysis.",
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
                    for name in ("frequency_mhz", "pulse_length_us")
                },
                **{
                    name: {"type": ["integer", "null"]}
                    for name in ("points", "reps", "rounds")
                },
                "gain_range": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
        run=amplitude_rabi,
    ),
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
