"""Read-only simulator methods; catalog descriptions are the MCP input contract."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import default_integer, optional_string, required_number
from ._registry import RemoteMethodEntry, method_entry

_SOURCES = (
    optional_string(
        "filename",
        "Predictor JSON filename including .json (e.g. fitted.json) in the opened "
        "entry's predictors directory, not a path, component or alias. No selected "
        "point is required. Supply exactly one of filename or params.",
    ),
    ParamSpec(
        "params",
        JsonType.OBJECT,
        required=False,
        description="Independent named inputs. Fluxonium: {model: 'fluxonium', EJ, "
        "EC, EL, flux_half, flux_period, flux_bias}. SQUID transmon: {model: "
        "'transmon', EJmax, EC, d, flux_half, flux_period, flux_bias}, where d is "
        "the junction asymmetry. Energies use GHz; affine values use native device "
        "units. Omitted/null fields are unknown. Supply exactly one of filename or "
        "params; inline inputs need no entry. No dialog/cache or alias is read.",
    ),
)

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "predictor.predict",
        "predictor:h_predictor_predict",
        MethodSpec(
            10.0,
            "Read-only transition prediction from exactly one filename or params source. "
            "Returns {freq_mhz}. Requires complete valid parameters. INVALID_PARAMS "
            "reasons: invalid_predictor_source, invalid_predictor_file, "
            "invalid_predictor_params, missing_predictor_params (data.missing lists fields). "
            "Filename sources can fail with PRECONDITION_FAILED/no_entry or "
            "predictor_file_unavailable.",
            _SOURCES
            + (
                required_number(
                    "device_value",
                    "Setpoint in source-native device units, NOT flux quanta. "
                    "flux = (device_value + flux_bias - flux_half)/flux_period + 0.5; "
                    "no unit conversion or hardware read occurs.",
                ),
                default_integer("from_level", 0, "Nonnegative transition from level"),
                default_integer("to_level", 1, "Nonnegative transition to level"),
            ),
            readonly=True,
        ),
    ),
    method_entry(
        "predictor.calibrate",
        "predictor:h_predictor_calibrate",
        MethodSpec(
            30.0,
            "Compute {flux_bias_before, flux_bias_after, frequency_mhz, clamped} from exactly one "
            "filename or params source. Complete valid parameters are required; missing "
            "fields return INVALID_PARAMS/missing_predictor_params with data.missing. "
            "Source conflicts return invalid_predictor_source; malformed params or "
            "files return invalid_predictor_params or invalid_predictor_file. "
            "Filename sources can fail with PRECONDITION_FAILED/no_entry or "
            "predictor_file_unavailable. Only returns a candidate: never writes a dialog, file or "
            "hardware. Selects smallest absolute native bias across all roots. "
            "Unreachable targets return the nearest attainable frequency_mhz and "
            "clamped=true, ties choose smallest absolute bias. Reachable targets "
            "return clamped=false. Invalid inputs are rejected, not clamped.",
            _SOURCES
            + (
                required_number(
                    "device_value", "Measured setpoint in native device units"
                ),
                required_number(
                    "frequency_mhz", "Measured transition frequency in MHz"
                ),
                default_integer(
                    "from_level", 0, "Nonnegative transition from level, below to_level"
                ),
                default_integer("to_level", 1, "Transition to level, above from_level"),
            ),
        ),
    ),
    method_entry(
        "predictor.info",
        "predictor:h_predictor_info",
        MethodSpec(
            5.0,
            "Read-only {model, params, missing} from exactly one filename or params "
            "source. Null fields remain unknown; no complete model is required or "
            "built. Source conflicts return INVALID_PARAMS/invalid_predictor_source; "
            "invalid filenames or content return invalid_predictor_file; malformed "
            "direct inputs return invalid_predictor_params. Filename sources can fail "
            "with PRECONDITION_FAILED/no_entry or predictor_file_unavailable. "
            "Never reads components, aliases or dialog/cache.",
            _SOURCES,
            readonly=True,
        ),
    ),
)
