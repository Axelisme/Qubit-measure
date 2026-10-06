"""Explicit legacy Labber import for the offline storage migration caller."""

import os
import tempfile
from pathlib import Path

import h5py

from zcu_tools.format_version import FormatError, FormatVersion, validate_header

from .grouped import GROUPED_VERSION_ATTR, load_grouped_labber_data
from .labber import load_labber_data
from .labber_schema import cast_labber_values
from .models import LabberPayload
from .native import load_run_data, validate_experiment_payload
from .native_metadata import read_json, text_attr, write_json
from .native_models import ExperimentPayload, VariableSchema
from .native_nodes import known_node


def migrate_native_run_v1(source: Path, destination: Path) -> Path:
    """Convert one native 1.x file into a new native 2.0 file.

    Both paths are exact caller-owned paths. Source remains byte-for-byte intact.
    Preserve data, cfg, run/provenance evidence and unrelated unknown HDF5/JSON,
    including local links and references. Remove only the obsolete context.params
    JSON object; do not decode its retired parameter-source schema.
    Validate the complete new file before atomic, no-overwrite publication.
    Same resolved paths raise ValueError; an existing destination raises
    FileExistsError. Wrong source format/major retains FormatError subtype;
    malformed known content or shared cfg/context datasets raise a located
    ValueError: removing context.params cannot also preserve cfg in that object. I/O errors
    propagate. No in-place rewrite, runtime fallback or power-loss guarantee.
    """
    if source.resolve() == destination.resolve():
        raise ValueError(f"{source}: source and destination must be different paths")
    if destination.exists():
        raise FileExistsError(destination)
    try:
        with h5py.File(source, "r") as file:
            validate_header(
                {
                    "format": text_attr(file, "format"),
                    "format_version": text_attr(file, "format_version"),
                },
                expected_format="zcu.experiment-data",
                supported_version=FormatVersion(1, 0),
                source=source,
            )
            cfg_node = known_node(file, "cfg")
            context_node = known_node(file, "context")
            if (
                cfg_node is not None
                and context_node is not None
                and cfg_node.id == context_node.id
            ):
                raise ValueError(
                    "/context: shared dataset with /cfg; cannot preserve cfg "
                    "while removing context.params"
                )
            context = read_json(file, "context")
            if not isinstance(context.get("params"), dict):
                raise ValueError("/context/params: expected required JSON object")
            del context["params"]
            # Copy the image, not individual nodes: unknown references stay valid.
            image = file.id.get_file_image()
    except ValueError as error:
        if isinstance(error, FormatError):
            raise
        raise ValueError(f"{source}: {error}") from error

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temp = Path(temporary)
    try:
        temp.write_bytes(image)
        with h5py.File(temp, "r+") as file:
            file.attrs["format_version"] = "2.0"
            write_json(file, "context", context)
        load_run_data(temp)
        # link rejects a destination created after the initial existence check.
        os.link(temp, destination)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error
    finally:
        temp.unlink(missing_ok=True)
    return destination


def load_legacy_labber_payload(
    source: Path, *, schema: tuple[VariableSchema, ...]
) -> ExperimentPayload:
    """Read supported legacy Labber layouts using an explicit disk schema.

    source is an exact existing Labber file; schema names its experiment's
    variables, axes, signal labels, SI/discrete units and numeric dtypes.
    Marked grouped files use the canonical/streaming grouped reader; unmarked
    files must declare one variable and use the single-log reader. No filename
    or channel guesses, unit conversion or runtime fallback is performed.
    Return a schema-validated generic payload with source metadata preserved.
    Numeric reader containers are cast through the Labber dtype contract.
    Unsupported layout/schema raises ValueError; filesystem errors propagate.
    This migration-only entry never modifies source.
    """
    if not schema:
        raise ValueError(f"{source}: missing disk schema")
    with h5py.File(source, "r") as file:
        grouped = GROUPED_VERSION_ATTR in file.attrs
    if grouped:
        loaded = load_grouped_labber_data(
            str(source), required_variables=[item.variable for item in schema]
        )
        variables = loaded.variables
        metadata = loaded.metadata
    else:
        if len(schema) != 1:
            raise ValueError(
                f"{source}: unmarked file cannot identify multiple variables"
            )
        single = load_labber_data(str(source))
        variables = {schema[0].variable: single.payload}
        metadata = single.metadata
    converted = {
        item.variable: LabberPayload(
            (
                variables[item.variable].data.name,
                variables[item.variable].data.unit,
                cast_labber_values(
                    variables[item.variable].z,
                    item.signal_dtype,
                    context=f"{source}/{item.variable}",
                ),
            ),
            [
                (
                    actual.name,
                    actual.unit,
                    cast_labber_values(
                        actual.values,
                        declared.dtype,
                        context=f"{source}/{item.variable}/{actual.name}",
                    ),
                )
                for actual, declared in zip(
                    variables[item.variable].axes, item.axes, strict=True
                )
            ],
            timestamps=variables[item.variable].timestamps,
        )
        for item in schema
    }
    payload = ExperimentPayload(
        variables=converted,
        metadata=metadata,
        representation="grouped" if grouped else "single",
    )
    validate_experiment_payload(payload, schema=schema)
    return payload
