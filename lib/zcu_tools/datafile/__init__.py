"""Standalone numeric-array/JSON containers and independent Labber codecs.

Exports load their owner on first use, so path helpers do not require HDF5.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .append import AppendWriter, open_append
    from .comment import LabberComment, decode_labber_comment, encode_labber_comment
    from .container import inspect, load, read_doc, save
    from .container_models import (
        Axis,
        Container,
        ContainerAttrs,
        ContainerInfo,
        FormatError,
        GroupSource,
        JsonObject,
        JsonScalar,
        Variable,
        VariableSchema,
    )
    from .grouped import load_grouped_labber_data, save_grouped_labber_data
    from .json_values import json_values_equal
    from .labber import save_labber_data, save_labber_trace_data
    from .labber_container import read_labber
    from .labber_reader import load_labber_data, load_labber_logs
    from .labber_schema import cast_labber_values, validate_labber_payload
    from .labber_snapshot import LabberSnapshotError, load_labber_device_snapshot
    from .labber_writer import write_labber
    from .models import (
        DataVariable,
        GroupedLabberData,
        LabberData,
        LabberMetadata,
        LabberPayload,
    )
    from .paths import (
        create_datafolder,
        format_ext,
        get_datafolder_path,
        remove_ext,
        reserve_labber_filepath,
    )

__all__ = [
    "save",
    "load",
    "read_doc",
    "inspect",
    "open_append",
    "AppendWriter",
    "Axis",
    "Variable",
    "VariableSchema",
    "Container",
    "ContainerAttrs",
    "ContainerInfo",
    "FormatError",
    "GroupSource",
    "JsonObject",
    "JsonScalar",
    "LabberComment",
    "encode_labber_comment",
    "decode_labber_comment",
    "LabberPayload",
    "LabberMetadata",
    "LabberData",
    "DataVariable",
    "GroupedLabberData",
    "write_labber",
    "read_labber",
    "validate_labber_payload",
    "cast_labber_values",
    "save_labber_data",
    "save_labber_trace_data",
    "load_labber_data",
    "load_labber_logs",
    "save_grouped_labber_data",
    "load_grouped_labber_data",
    "LabberSnapshotError",
    "load_labber_device_snapshot",
    "json_values_equal",
    "format_ext",
    "remove_ext",
    "reserve_labber_filepath",
    "create_datafolder",
    "get_datafolder_path",
]

_EXPORT_MODULES = {
    "save": ".container",
    "load": ".container",
    "read_doc": ".container",
    "inspect": ".container",
    "open_append": ".append",
    "AppendWriter": ".append",
    "Axis": ".container_models",
    "Variable": ".container_models",
    "VariableSchema": ".container_models",
    "Container": ".container_models",
    "ContainerAttrs": ".container_models",
    "ContainerInfo": ".container_models",
    "FormatError": ".container_models",
    "GroupSource": ".container_models",
    "JsonObject": ".container_models",
    "JsonScalar": ".container_models",
    "LabberComment": ".comment",
    "encode_labber_comment": ".comment",
    "decode_labber_comment": ".comment",
    "LabberPayload": ".models",
    "LabberMetadata": ".models",
    "LabberData": ".models",
    "DataVariable": ".models",
    "GroupedLabberData": ".models",
    "write_labber": ".labber_writer",
    "read_labber": ".labber_container",
    "validate_labber_payload": ".labber_schema",
    "cast_labber_values": ".labber_schema",
    "save_labber_data": ".labber",
    "save_labber_trace_data": ".labber",
    "load_labber_data": ".labber_reader",
    "load_labber_logs": ".labber_reader",
    "save_grouped_labber_data": ".grouped",
    "load_grouped_labber_data": ".grouped",
    "LabberSnapshotError": ".labber_snapshot",
    "load_labber_device_snapshot": ".labber_snapshot",
    "json_values_equal": ".json_values",
    "format_ext": ".paths",
    "remove_ext": ".paths",
    "reserve_labber_filepath": ".paths",
    "create_datafolder": ".paths",
    "get_datafolder_path": ".paths",
}


def __getattr__(name: str) -> object:
    """Load an exported codec or value type only when the caller selects it."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List public exports without loading their owners."""
    return sorted(set(globals()) | set(__all__))
