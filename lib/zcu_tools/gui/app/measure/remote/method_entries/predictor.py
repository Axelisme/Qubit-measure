"""Read-only simulator methods; catalog descriptions are the MCP input contract."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import default_integer, optional_string, required_number
from ._registry import RemoteMethodEntry, method_entry

_SOURCES = (
    optional_string(
        "qubit",
        "Concrete qubit component name (e.g. Q2) at the current selected point, "
        "not an alias/path. Supply exactly one of qubit or params.",
    ),
    ParamSpec(
        "params",
        JsonType.OBJECT,
        required=False,
        description="Independent named inputs. Fluxonium: {model: 'fluxonium', EJ, "
        "EC, EL, flux_half, flux_period, flux_bias}. SQUID transmon: {model: "
        "'transmon', EJmax, EC, d, flux_half, flux_period, flux_bias}, where d is "
        "the junction asymmetry. Energies use GHz; affine values use native device "
        "units. Omitted/null fields are unknown. Supply exactly one of qubit or "
        "params; no dialog/cache or alias is read.",
    ),
)

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "predictor.predict",
        "predictor:h_predictor_predict",
        MethodSpec(
            10.0,
            "Read-only transition prediction from exactly one qubit or params source. "
            "Returns {freq_mhz}. Requires complete valid parameters. INVALID_PARAMS "
            "reasons: invalid_predictor_source, invalid_predictor_qubit, "
            "invalid_predictor_params, missing_predictor_params (data.missing lists fields).",
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
        ),
    ),
    method_entry(
        "predictor.calibrate",
        "predictor:h_predictor_calibrate",
        MethodSpec(
            30.0,
            "Compute native {flux_bias_before, flux_bias_after} from exactly one "
            "qubit or params source. Complete valid parameters are required; missing "
            "fields return INVALID_PARAMS/missing_predictor_params with data.missing. "
            "Source conflicts return invalid_predictor_source; malformed params or "
            "qubits return invalid_predictor_params or invalid_predictor_qubit. "
            "Only returns a candidate: never writes a component, dialog, file or "
            "hardware. Numerical root failure warns and follows model fallback, "
            "so a returned bias is not proof of a fit.",
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
            "Read-only {model, params, missing} from exactly one qubit or params "
            "source. Null fields remain unknown; no complete model is required or "
            "built. Source conflicts return INVALID_PARAMS/invalid_predictor_source; "
            "unknown/invalid concrete names return invalid_predictor_qubit; malformed "
            "direct inputs return invalid_predictor_params. Never reads dialog/cache.",
            _SOURCES,
        ),
    ),
)
