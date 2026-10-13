"""Public facade for experiment data persistence.

Exports load their owner on first access, so path helpers do not require HDF5.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis_document import AnalysisDocument, encode_analysis_document
    from .comment import LabberComment, decode_labber_comment, encode_labber_comment
    from .fluxdep_database import FluxdepDatabaseInfo, read_fluxdep_database_info
    from .grouped import load_grouped_labber_data, save_grouped_labber_data
    from .labber import (
        load_labber_data,
        save_labber_data,
        save_labber_trace_data,
    )
    from .labber_schema import cast_labber_values, validate_labber_payload
    from .labber_writer import write_labber
    from .models import (
        Axis,
        DataVariable,
        GroupedLabberData,
        LabberData,
        LabberMetadata,
        LabberPayload,
    )
    from .native import load_run_data, save_run_data, validate_experiment_payload
    from .native_metadata import json_values_equal, validate_cfg_snapshot
    from .native_models import (
        AxisSchema,
        CfgSnapshot,
        ExperimentPayload,
        JsonObject,
        NativeExtensions,
        RunMetadata,
        RunSnapshot,
        SoftwareProvenance,
        StoredRun,
        VariableSchema,
    )
    from .paths import (
        create_datafolder,
        format_ext,
        get_datafolder_path,
        remove_ext,
        reserve_labber_filepath,
    )
    from .streaming import (
        StreamingGroupedLabberWriter,
        StreamingLabberVariableSpec,
        StreamingLabberWriter,
        open_streaming_grouped_labber_data,
        open_streaming_labber_data,
    )

__all__ = [
    "FluxdepDatabaseInfo",
    "read_fluxdep_database_info",
    "AnalysisDocument",
    "encode_analysis_document",
    "LabberComment",
    "encode_labber_comment",
    "decode_labber_comment",
    "AxisSchema",
    "VariableSchema",
    "ExperimentPayload",
    "CfgSnapshot",
    "RunSnapshot",
    "SoftwareProvenance",
    "RunMetadata",
    "NativeExtensions",
    "StoredRun",
    "JsonObject",
    "json_values_equal",
    "validate_labber_payload",
    "cast_labber_values",
    "write_labber",
    "save_run_data",
    "load_run_data",
    "validate_experiment_payload",
    "validate_cfg_snapshot",
    "Axis",
    "LabberPayload",
    "LabberMetadata",
    "LabberData",
    "DataVariable",
    "GroupedLabberData",
    "save_labber_data",
    "load_labber_data",
    "save_grouped_labber_data",
    "load_grouped_labber_data",
    "StreamingLabberVariableSpec",
    "StreamingGroupedLabberWriter",
    "StreamingLabberWriter",
    "open_streaming_labber_data",
    "open_streaming_grouped_labber_data",
    "save_labber_trace_data",
    "format_ext",
    "remove_ext",
    "reserve_labber_filepath",
    "get_datafolder_path",
    "create_datafolder",
]

_EXPORT_MODULES = {
    "FluxdepDatabaseInfo": ".fluxdep_database",
    "read_fluxdep_database_info": ".fluxdep_database",
    "AnalysisDocument": ".analysis_document",
    "encode_analysis_document": ".analysis_document",
    "LabberComment": ".comment",
    "decode_labber_comment": ".comment",
    "encode_labber_comment": ".comment",
    "load_grouped_labber_data": ".grouped",
    "save_grouped_labber_data": ".grouped",
    "load_labber_data": ".labber",
    "save_labber_data": ".labber",
    "save_labber_trace_data": ".labber",
    "cast_labber_values": ".labber_schema",
    "validate_labber_payload": ".labber_schema",
    "write_labber": ".labber_writer",
    "Axis": ".models",
    "DataVariable": ".models",
    "GroupedLabberData": ".models",
    "LabberData": ".models",
    "LabberMetadata": ".models",
    "LabberPayload": ".models",
    "load_run_data": ".native",
    "save_run_data": ".native",
    "validate_experiment_payload": ".native",
    "json_values_equal": ".native_metadata",
    "validate_cfg_snapshot": ".native_metadata",
    "AxisSchema": ".native_models",
    "CfgSnapshot": ".native_models",
    "ExperimentPayload": ".native_models",
    "JsonObject": ".native_models",
    "NativeExtensions": ".native_models",
    "RunMetadata": ".native_models",
    "RunSnapshot": ".native_models",
    "SoftwareProvenance": ".native_models",
    "StoredRun": ".native_models",
    "VariableSchema": ".native_models",
    "create_datafolder": ".paths",
    "format_ext": ".paths",
    "get_datafolder_path": ".paths",
    "remove_ext": ".paths",
    "reserve_labber_filepath": ".paths",
    "StreamingGroupedLabberWriter": ".streaming",
    "StreamingLabberVariableSpec": ".streaming",
    "StreamingLabberWriter": ".streaming",
    "open_streaming_grouped_labber_data": ".streaming",
    "open_streaming_labber_data": ".streaming",
}


def __getattr__(name: str) -> object:
    """Load an exported persistence tool only when the caller selects it."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include the public exports without loading their owners."""
    return sorted(set(globals()) | set(__all__))
