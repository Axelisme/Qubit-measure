"""Read-only simulator methods; catalog descriptions are the MCP input contract."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import default_integer, optional_string, required_number
from ._registry import RemoteMethodEntry, method_entry

_SOURCES = (
    optional_string(
        "filename",
        "V2 predictor JSON filename including .json in the opened entry's predictors "
        "directory, not a path, component or alias. No selected point is required. "
        "Supply exactly one non-null source: filename or params. Omission/null means "
        "no source. Filename uses recorded flux and forbids a separate flux argument.",
    ),
    ParamSpec(
        "params",
        JsonType.OBJECT,
        required=False,
        description="Independent energy inputs: fluxonium {model: 'fluxonium', EJ, EC, EL} "
        "or SQUID transmon {model: 'transmon', EJmax, EC, d}, where d is junction "
        "asymmetry. Energies use GHz. Omitted/null fields are unknown. Supply exactly "
        "one non-null source: filename or params. Inline inputs need no entry; "
        "predict/calibrate also require explicit top-level flux. No dialog/cache or alias reads.",
    ),
    ParamSpec(
        "flux",
        JsonType.OBJECT,
        required=False,
        nullable=True,
        omit_when_missing=True,
        description="Inline coordinates: {offset: finite Phi0, period: {device: signed "
        "nonzero native units per Phi0}, unit: {device: nonblank unit}}. Period/unit "
        "keys must match. Required for inline predict/calibrate, optional for info. "
        "Null is invalid. Forbidden with filename; its recorded flux is authoritative.",
    ),
)
_DEVICE_VALUES = ParamSpec(
    "device_values",
    JsonType.OBJECT,
    description="Every flux device's finite native value, keyed by name. Missing "
    "devices are errors naming the device, never zero-filled. Extra devices are "
    "ignored. Reduced flux = offset + sum(value/period); no unit conversion or "
    "hardware/cache reads.",
)
_SOURCE_ERRORS = (
    "INVALID_PARAMS reasons: invalid_predictor_source for missing/conflicting sources "
    "or filename plus flux; invalid_predictor_file for malformed files; "
    "invalid_predictor_params for malformed energies/device values; "
    "invalid_predictor_flux for malformed inline flux. Filename sources can fail "
    "with PRECONDITION_FAILED/no_entry or predictor_file_unavailable. "
)
_QUERY_ERRORS = (
    "Complete valid energies and flux are required. Missing energies return "
    "INVALID_PARAMS/missing_predictor_params with data.missing listing fields; "
    "missing inline flux returns missing_predictor_flux. Missing required device "
    "values return invalid_predictor_params naming the devices. "
)

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "predictor.list_files",
        "predictor:h_predictor_list_files",
        MethodSpec(
            5.0,
            "Read-only discovery of the opened entry's predictor files. Returns "
            "{files: [filename, ...]} sorted by filename, including .json, relative "
            "to that entry's predictors directory. Use a filename with predictor.info, "
            "predictor.predict or predictor.calibrate. No selected point is required. "
            "No entry or absent directory returns {files: []}. Listing does not read "
            "or validate content and never creates files or directories. Other I/O "
            "failures propagate as unexpected errors.",
            readonly=True,
        ),
    ),
    method_entry(
        "predictor.predict",
        "predictor:h_predictor_predict",
        MethodSpec(
            10.0,
            "Read-only transition prediction from exactly one filename or params source. "
            "Returns {freq_mhz}. " + _SOURCE_ERRORS + _QUERY_ERRORS,
            _SOURCES
            + (
                _DEVICE_VALUES,
                default_integer("from_level", 0, "Nonnegative transition from level"),
                default_integer(
                    "to_level", 1, "Transition to level, at least from_level"
                ),
            ),
            readonly=True,
        ),
    ),
    method_entry(
        "predictor.calibrate",
        "predictor:h_predictor_calibrate",
        MethodSpec(
            30.0,
            "Compute {offset_correction, frequency_mhz, clamped} from exactly one "
            "filename or params source. Add offset_correction (Phi0) to the caller's "
            "current offset. Returns a candidate only: never writes a dialog, file or "
            "hardware. Selects smallest absolute correction across all roots. "
            "Unreachable targets return the nearest attainable frequency_mhz and "
            "clamped=true; ties choose smallest absolute correction. Reachable targets "
            "return clamped=false. Invalid inputs are rejected, not clamped. "
            + _SOURCE_ERRORS
            + _QUERY_ERRORS,
            _SOURCES
            + (
                _DEVICE_VALUES,
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
            "Read-only {model, params, missing, flux} from exactly one filename or params "
            "source. Null energy fields remain unknown; no complete model is required or "
            "built. missing lists energy fields only. flux is the file's recorded coordinates, "
            "validated inline flux if supplied, or null for inline inputs without flux. "
            "Never reads components, aliases or dialog/cache. " + _SOURCE_ERRORS,
            _SOURCES,
            readonly=True,
        ),
    ),
)
