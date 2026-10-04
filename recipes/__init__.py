"""Explicit, editable recipe registration; no discovery or hot reload."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
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
    adapter_name: str = field(kw_only=True)
    summary_parameters: tuple[SummaryParameter, ...] = field(kw_only=True)
    summary_estimates: tuple[SummaryEstimate, ...] = field(kw_only=True)


_READOUT = (
    SummaryParameter("readout_frequency_mhz", "modules.readout.pulse_cfg.freq", "MHz"),
    SummaryParameter("ro_frequency_mhz", "modules.readout.ro_cfg.ro_freq", "MHz"),
    SummaryParameter("readout_ref", "modules.readout"),
    SummaryParameter("use_reset", "modules.reset"),
)
_AVERAGING = (SummaryParameter("reps", "reps"), SummaryParameter("rounds", "rounds"))
_DRIVE = (
    *_READOUT,
    *_AVERAGING,
    SummaryParameter("frequency_mhz", "modules.qub_pulse.freq", "MHz"),
    SummaryParameter("drive_ref", "modules.qub_pulse"),
)
_PULSE_LENGTH = SummaryParameter(
    "pulse_length_us", "modules.qub_pulse.waveform.length", "us"
)
_ONETONE_PARAMETERS = (
    SummaryParameter("center_mhz", "center_mhz", "MHz"),
    SummaryParameter("span_mhz", "span_mhz", "MHz"),
    SummaryParameter("frequency_sweep", "sweep.freq", "MHz"),
    SummaryParameter("gain", "modules.readout.pulse_cfg.gain"),
    SummaryParameter("readout_ref", "modules.readout"),
    *_AVERAGING,
)
_ONETONE_ESTIMATES = (
    SummaryEstimate("freq", "freq", unit="MHz"),
    SummaryEstimate("fwhm", "fwhm", unit="MHz"),
)
_COHERENCE_PARAMETERS = {
    "t2ramsey": (
        *_READOUT,
        *_AVERAGING,
        SummaryParameter("delay", "sweep.length", "us"),
        SummaryParameter("detune_ratio", "detune_ratio"),
        SummaryParameter("pi2_ref", "modules.pi2_pulse"),
    ),
    "t2echo": (
        *_READOUT,
        *_AVERAGING,
        SummaryParameter("delay", "sweep.length", "us"),
        SummaryParameter("detune_ratio", "detune_ratio"),
        SummaryParameter("pi_ref", "modules.pi_pulse"),
        SummaryParameter("pi2_ref", "modules.pi2_pulse"),
    ),
}
_COHERENCE_ESTIMATES = {
    "t2ramsey": (
        SummaryEstimate("t2r", "t2r", "t2r_err", "us"),
        SummaryEstimate("detune", "detune", unit="MHz"),
    ),
    "t2echo": (SummaryEstimate("t2e", "t2e", "t2e_err", "us"),),
}


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
                        "reuse_tab_id",
                        "readout_ref",
                        "pi_ref",
                        "use_reset",
                        "init_pulse_ref",
                    )
                },
                "shots": {"type": ["integer", "null"]},
            },
        },
        run=singleshot_ge,
        adapter_name="singleshot/ge",
        summary_parameters=(
            *_READOUT,
            SummaryParameter("shots", "shots"),
            SummaryParameter("pi_ref", "modules.probe_pulse"),
            SummaryParameter("init_pulse_ref", "modules.init_pulse"),
        ),
        summary_estimates=(
            SummaryEstimate("fidelity", "fidelity"),
            SummaryEstimate("theta", "theta"),
            SummaryEstimate("threshold", "threshold"),
            SummaryEstimate("ge_s", "ge_s"),
        ),
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
            adapter_name=f"twotone/{name}",
            summary_parameters=_COHERENCE_PARAMETERS[name],
            summary_estimates=_COHERENCE_ESTIMATES[name],
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
        adapter_name="twotone/t1",
        summary_parameters=(
            *_READOUT,
            *_AVERAGING,
            SummaryParameter("delay", "sweep.length", "us"),
            SummaryParameter("pi_ref", "modules.pi_pulse"),
        ),
        summary_estimates=(
            SummaryEstimate("t1", "t1", "t1_err", "us"),
            SummaryEstimate("t1b", "t1b", "t1b_err", "us"),
        ),
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
        adapter_name="twotone/rabi/amp_rabi",
        summary_parameters=(
            *_DRIVE,
            _PULSE_LENGTH,
            SummaryParameter("gain_sweep", "sweep.gain"),
        ),
        summary_estimates=(
            SummaryEstimate("pi_gain", "pi_gain", "pi_gain_err"),
            SummaryEstimate("pi2_gain", "pi2_gain", "pi2_gain_err"),
        ),
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
        adapter_name="twotone/rabi/len_rabi",
        summary_parameters=(
            *_DRIVE,
            SummaryParameter("gain", "modules.qub_pulse.gain"),
            SummaryParameter("length_sweep", "sweep.length", "us"),
        ),
        summary_estimates=(
            SummaryEstimate("pi_len", "pi_len", "pi_len_err", "us"),
            SummaryEstimate("pi2_len", "pi2_len", "pi2_len_err", "us"),
            SummaryEstimate("rabi_f", "rabi_f", "rabi_f_err", "MHz"),
        ),
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
        adapter_name="twotone/freq",
        summary_parameters=(
            *_READOUT,
            *_AVERAGING,
            _PULSE_LENGTH,
            SummaryParameter("gain", "modules.qub_pulse.gain"),
            SummaryParameter("frequency_sweep", "sweep.freq", "MHz"),
            SummaryParameter("center_mhz", "center_mhz", "MHz"),
            SummaryParameter("span_mhz", "span_mhz", "MHz"),
            SummaryParameter("drive_ref", "modules.qub_pulse"),
        ),
        summary_estimates=(
            SummaryEstimate("freq", "freq", "freq_err", "MHz"),
            SummaryEstimate("fwhm", "fwhm", "fwhm_err", "MHz"),
        ),
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
        adapter_name="lookback",
        summary_parameters=(
            SummaryParameter("frequency_mhz", "modules.readout.pulse_cfg.freq", "MHz"),
            SummaryParameter(
                "ro_frequency_mhz", "modules.readout.ro_cfg.ro_freq", "MHz"
            ),
            SummaryParameter(
                "readout_length_us", "modules.readout.ro_cfg.ro_length", "us"
            ),
            SummaryParameter(
                "trigger_offset_us", "modules.readout.ro_cfg.trig_offset", "us"
            ),
            SummaryParameter("rounds", "rounds"),
            SummaryParameter("readout_ref", "modules.readout"),
            SummaryParameter("use_reset", "modules.reset"),
            SummaryParameter("init_pulse_ref", "modules.init_pulse"),
        ),
        summary_estimates=(
            SummaryEstimate("predict_offset", "predict_offset", unit="us"),
        ),
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
        adapter_name="onetone/freq",
        summary_parameters=_ONETONE_PARAMETERS,
        summary_estimates=_ONETONE_ESTIMATES,
    ),
    RecipeDefinition(
        name="onetone_spectrum_over_flux",
        description="Run one frequency/flux survey, save raw and Primary analysis. Physical units must match; FakeDevice requires flux_unit=native.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                **_ONETONE_COMMON_PROPERTIES,
                "gain": {"type": ["number", "null"]},
                "flux_device": {"type": ["string", "null"], "minLength": 1},
                "flux_unit": {
                    "type": ["string", "null"],
                    "minLength": 1,
                    "description": "Optional device-unit assertion. FakeDevice requires explicit native coordinates.",
                },
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
        adapter_name="onetone/flux_dep",
        summary_parameters=(
            *_ONETONE_PARAMETERS,
            SummaryParameter("flux_sweep", "sweep.flux"),
            SummaryParameter("flux_device", "dev.flux_dev"),
        ),
        summary_estimates=(),
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
        adapter_name="onetone/power_dep",
        summary_parameters=(
            *_ONETONE_PARAMETERS,
            SummaryParameter("gain_sweep", "sweep.gain"),
        ),
        summary_estimates=(),
    ),
)
