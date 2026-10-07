"""Native experiment data public I/O contracts."""

import json
import os
import tempfile
from pathlib import Path

import h5py as h5
import numpy as np
from pydantic import TypeAdapter, ValidationError

from zcu_tools.format_version import FormatError, FormatVersion, validate_header

from .models import DataVariable, LabberMetadata, LabberPayload
from .native_metadata import (
    json_object,
    read_metadata,
    text_attr,
    validate_metadata,
    write_metadata,
)
from .native_models import (
    CfgSnapshot,
    ExperimentPayload,
    NativeExtensions,
    RunMetadata,
    StoredRun,
    VariableSchema,
)
from .native_nodes import known_group, known_node


def save_run_data(
    destination: Path,
    payload: ExperimentPayload,
    metadata: RunMetadata,
    *,
    cfg: CfgSnapshot,
    replace: bool = False,
    extensions: NativeExtensions | None = None,
) -> None:
    """Save a native zcu.experiment-data file at the caller's exact destination.

    payload contains SI/discrete arrays; metadata and cfg are historical inputs,
    not instructions to capture live state. Validate before publication.
    Existing destinations raise FileExistsError unless replace=True. Write a
    same-directory temporary file, close it, then atomically publish; failures
    remove this call's temp and preserve the prior destination.
    extensions reuses a detached validated image for same-shape generic rewrites;
    None creates version 1.0. Known nodes must resolve within this file. Existing
    fixed UTF-8 JSON retains unchanged raw text; edits beyond byte capacity fail.
    Invalid known data raises a destination/location ValueError. I/O errors
    propagate. No cross-file or power-loss guarantee.
    """
    if not replace and destination.exists():
        raise FileExistsError(destination)
    try:
        _validate_payload_arrays(payload)
        validate_metadata(metadata, cfg)
        labber_json = _LABBER_METADATA.dump_json(payload.metadata).decode("utf-8")
        _LABBER_METADATA.validate_json(labber_json, strict=True)
    except ValueError as error:
        raise ValueError(f"{destination}: {error}") from error
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temp = Path(temporary)
    try:
        if extensions is not None:
            temp.write_bytes(extensions.file_image)
        with h5.File(temp, "r+" if extensions is not None else "w") as file:
            if extensions is not None:
                _validate_header(file, destination)
            else:
                file.attrs["format"] = "zcu.experiment-data"
                file.attrs["format_version"] = "1.0"
            write_metadata(file, metadata, cfg)
            _write_payload(file, payload, labber_json)
        if replace:
            os.replace(temp, destination)
        else:
            # link publishes without overwriting a destination created after validation.
            os.link(temp, destination)
    except ValueError as error:
        if isinstance(error, FormatError):
            raise
        raise ValueError(f"{destination}: {error}") from error
    finally:
        temp.unlink(missing_ok=True)


def load_run_data(source: Path, *, preserve_unknown: bool = False) -> StoredRun:
    """Read and close one native file, returning known payload, metadata and cfg.

    Reject legacy/Labber, wrong major, missing required attrs/datasets, invalid
    wire values, names, shapes, dtypes and units with source/location ValueError
    (format-version errors retain their existing public subtype). Same-major
    minor versions are readable. cfg preserves the entire JSON object.
    preserve_unknown=True additionally captures a detached HDF5 image for an
    explicit generic lossless rewrite; False returns extensions=None.
    Unknown nodes are not decoded or used to traverse external link targets.
    Known nodes must also be local, including any soft-link targets; external
    targets fail before dereferencing. I/O errors propagate. No live instruments
    or registry are consulted.
    """
    with h5.File(source, "r") as file:
        try:
            _validate_header(file, source)
            metadata, cfg = read_metadata(file, source)
            payload = _read_payload(file)
            _validate_payload_arrays(payload)
        except ValueError as error:
            # Shared header failures retain their public FormatError/VersionError subtype.
            if isinstance(error, FormatError):
                raise
            raise ValueError(f"{source}: {error}") from error
        extensions = (
            NativeExtensions(file_image=file.id.get_file_image())
            if preserve_unknown
            else None
        )
        return StoredRun(
            payload=payload, metadata=metadata, cfg=cfg, extensions=extensions
        )


_LABBER_METADATA = TypeAdapter(LabberMetadata)


def _validate_header(file: h5.File, source: Path) -> None:
    validate_header(
        {
            "format": text_attr(file, "format"),
            "format_version": text_attr(file, "format_version"),
        },
        expected_format="zcu.experiment-data",
        supported_version=FormatVersion(1, 0),
        source=source,
    )


def _write_array(
    group: h5.Group, name: str, values: np.ndarray, unit: str
) -> h5.Dataset:
    existing = known_node(group, name)
    if existing is not None:
        if not isinstance(existing, h5.Dataset):
            raise ValueError(f"{group.name}/{name}: expected numeric dataset")
        if existing.shape == values.shape and existing.dtype == values.dtype:
            existing[...] = values
            existing.attrs["units"] = unit
            return existing
        # Structural rewrites do not promise remapping unknown references.
        del group[name]
    dataset = group.create_dataset(name, data=values)
    dataset.attrs["units"] = unit
    return dataset


def _write_payload(file: h5.File, payload: ExperimentPayload, labber_json: str) -> None:
    data = known_group(file, "data")
    data.attrs["representation"] = payload.representation
    if "metadata" in data.attrs:
        prior = json_object(text_attr(data, "metadata"), "/data: metadata")
        prior.update(json_object(labber_json, "/data: metadata"))
        labber_json = json.dumps(prior, ensure_ascii=False, allow_nan=False)
    data.attrs["metadata"] = labber_json
    for variable, item in payload.variables.items():
        group = known_group(data, variable)
        group.attrs["signal"] = item.data.name
        group.attrs["axes"] = np.array(
            [axis.name for axis in item.axes], dtype=h5.string_dtype("utf-8")
        )
        scales: list[h5.Dataset] = []
        for axis in item.axes:
            dataset = _write_array(group, axis.name, np.asarray(axis.values), axis.unit)
            if not dataset.is_scale:
                dataset.make_scale(axis.name)
            scales.append(dataset)
        signal = _write_array(group, item.data.name, np.asarray(item.z), item.data.unit)
        for index, scale in enumerate(reversed(scales)):
            attached = list(signal.dims[index].values())
            if not any(old.id == scale.id for old in attached):
                signal.dims[index].attach_scale(scale)
        if item.timestamps is not None:
            timestamps = _write_array(
                group, "timestamps", np.asarray(item.timestamps), "s"
            )
            # A separate scale lets h5netcdf expose per-trace timestamps as a coordinate.
            if not timestamps.is_scale:
                timestamps.make_scale("timestamps")
        elif "timestamps" in group:
            del group["timestamps"]


def _numeric_dataset(group: h5.Group, name: str, *, real: bool = False) -> h5.Dataset:
    location = f"{group.name}/{name}"
    dataset = known_node(group, name)
    if not isinstance(dataset, h5.Dataset):
        raise ValueError(f"{location}: missing numeric dataset")
    if dataset.dtype.kind not in ("iuf" if real else "iufc"):
        raise ValueError(f"{location}: expected numeric dtype")
    return dataset


def _axis_names(group: h5.Group) -> tuple[str, ...]:
    if "axes" not in group.attrs:
        raise ValueError(f"{group.name}: missing axes attr")
    names = np.asarray(group.attrs["axes"])
    if names.ndim != 1:
        raise ValueError(f"{group.name}: axes must be a string array")
    result: list[str] = []
    for value in names:
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{group.name}: axes must be UTF-8") from error
        if not isinstance(value, str):
            raise ValueError(f"{group.name}: axes must be strings")
        result.append(value)
    return tuple(result)


def _read_variable(group: h5.Group) -> LabberPayload:
    signal_name = text_attr(group, "signal")
    names = _axis_names(group)
    _validate_names(names, signal_name, f"{group.name}")
    signal = _numeric_dataset(group, signal_name)
    axes: list[tuple[str, str, np.ndarray]] = []
    datasets: list[h5.Dataset] = []
    for name in names:
        axis = _numeric_dataset(group, name)
        if axis.ndim != 1 or not axis.is_scale:
            raise ValueError(f"{axis.name}: expected 1D dimension scale")
        datasets.append(axis)
        axes.append((name, text_attr(axis, "units"), np.asarray(axis[()])))
    if signal.ndim != len(datasets):
        raise ValueError(f"{signal.name}: shape does not match axes")
    for index, axis in enumerate(reversed(datasets)):
        scales = list(signal.dims[index].values())
        if len(scales) != 1 or scales[0].id != axis.id:
            raise ValueError(
                f"{signal.name}: dimension {index} does not attach {axis.name}"
            )
    timestamps = None
    if "timestamps" in group:
        dataset = _numeric_dataset(group, "timestamps", real=True)
        if text_attr(dataset, "units") != "s":
            raise ValueError(f"{dataset.name}: timestamps units must be s")
        timestamps = np.asarray(dataset[()])
    item = LabberPayload(
        (signal_name, text_attr(signal, "units"), np.asarray(signal[()])),
        axes=axes,
        timestamps=timestamps,
    )
    _validate_variable_arrays(item, f"{group.name}")
    return item


def _read_payload(file: h5.File) -> ExperimentPayload:
    data = known_node(file, "data")
    if not isinstance(data, h5.Group):
        raise ValueError("/data: missing group")
    representation = text_attr(data, "representation")
    if representation != "single" and representation != "grouped":
        raise ValueError("/data: representation must be single or grouped")
    labber_json = text_attr(data, "metadata")
    fields = json_object(labber_json, "/data: metadata")
    missing = {"comment", "tags", "project", "user", "creation_time"} - fields.keys()
    if missing:
        raise ValueError(f"/data: metadata missing fields {sorted(missing)}")
    try:
        metadata = _LABBER_METADATA.validate_json(labber_json, strict=True)
    except ValidationError as error:
        raise ValueError(f"/data: metadata: {error}") from error
    variables: dict[DataVariable, LabberPayload] = {}
    for name in data:
        if not isinstance(name, str):
            raise ValueError("/data: child names must be strings")
        # Unknown external targets are not opened or incorporated into the image.
        if not isinstance(data.get(name, getlink=True), h5.HardLink):
            continue
        node = data[name]
        if not isinstance(node, h5.Group):
            continue
        if "signal" not in node.attrs and "axes" not in node.attrs:
            continue
        try:
            variable = DataVariable(name)
        except ValueError as error:
            raise ValueError(f"{node.name}: {error}") from error
        variables[variable] = _read_variable(node)
    return ExperimentPayload(
        variables=variables, metadata=metadata, representation=representation
    )


def validate_experiment_payload(
    payload: ExperimentPayload, *, schema: tuple[VariableSchema, ...]
) -> None:
    """Validate exact declared variable identities, axes and channel wire arrays.

    schema specifies inner-first labels, units and NumPy disk dtypes. Validate
    numeric arrays and reversed-axis signal shapes; optional timestamps are flat
    real numeric Unix epoch seconds, one per outer trace, including NaN.
    Empty/duplicate declarations, incompatible payload or malformed names raise
    a located ValueError. Different variables may use different grids.
    This function does not mutate arrays or convert memory units.
    """
    _validate_payload_arrays(payload)
    if not schema:
        raise ValueError("/data: schema must declare at least one variable")
    declarations = {item.variable: item for item in schema}
    if len(declarations) != len(schema):
        raise ValueError("/data: duplicate schema variable identities")
    if set(declarations) != set(payload.variables):
        raise ValueError(
            f"/data: schema variables {tuple(declarations)} do not match "
            f"payload variables {tuple(payload.variables)}"
        )
    for variable, declaration in declarations.items():
        location = f"/data/{variable}"
        item = payload.variables[variable]
        _validate_names(
            tuple(axis.name for axis in declaration.axes),
            declaration.signal_name,
            location,
        )
        if len(item.axes) != len(declaration.axes):
            raise ValueError(f"{location}: axis count does not match schema")
        for actual, expected in zip(item.axes, declaration.axes, strict=True):
            axis_location = f"{location}/{expected.name}"
            if actual.name != expected.name or actual.unit != expected.unit:
                raise ValueError(
                    f"{axis_location}: axis name or units do not match schema"
                )
            _validate_dtype(
                np.asarray(actual.values).dtype, expected.dtype, axis_location
            )
        if item.data.name != declaration.signal_name:
            raise ValueError(f"{location}: signal name does not match schema")
        signal_location = f"{location}/{declaration.signal_name}"
        if item.data.unit != declaration.signal_unit:
            raise ValueError(f"{signal_location}: units do not match schema")
        _validate_dtype(
            np.asarray(item.z).dtype, declaration.signal_dtype, signal_location
        )


def _validate_dtype(
    actual: np.dtype[np.generic], expected: np.dtype[np.generic], location: str
) -> None:
    if expected.kind not in "iufc":
        raise ValueError(f"{location}: schema dtype {expected} is not numeric")
    if actual != expected:
        raise ValueError(f"{location}: dtype {actual} does not match schema {expected}")


def _validate_names(axes: tuple[object, ...], signal: object, location: str) -> None:
    seen: set[str] = set()
    for name in (*axes, signal):
        if not isinstance(name, str) or not name or "/" in name or "\x00" in name:
            raise ValueError(f"{location}: invalid axis or signal name {name!r}")
        if name in {".", "..", "timestamps"}:
            raise ValueError(f"{location}: reserved axis or signal name {name!r}")
        if name in seen:
            raise ValueError(f"{location}: axis and signal names collide")
        seen.add(name)


def _validate_payload_arrays(payload: ExperimentPayload) -> None:
    if not payload.variables:
        raise ValueError("/data: payload variables must not be empty")
    if payload.representation not in {"single", "grouped"}:
        raise ValueError("/data: representation must be single or grouped")
    if payload.representation == "single" and len(payload.variables) != 1:
        raise ValueError("/data: single representation requires exactly one variable")
    for variable, item in payload.variables.items():
        location = f"/data/{variable}"
        try:
            DataVariable(variable)
        except ValueError as error:
            raise ValueError(f"{location}: {error}") from error
        _validate_variable_arrays(item, location)


def _validate_variable_arrays(item: LabberPayload, location: str) -> None:
    _validate_names(tuple(axis.name for axis in item.axes), item.data.name, location)
    for channel in (*item.axes, item.data):
        if not isinstance(channel.unit, str):
            raise ValueError(f"{location}/{channel.name}: units must be a string")
    lengths: list[int] = []
    for axis in item.axes:
        values = np.asarray(axis.values)
        if values.ndim != 1 or values.dtype.kind not in "iufc":
            raise ValueError(f"{location}/{axis.name}: axis must be numeric and 1D")
        lengths.append(len(values))
    signal = np.asarray(item.z)
    if signal.dtype.kind not in "iufc":
        raise ValueError(f"{location}/{item.data.name}: signal dtype must be numeric")
    shape = tuple(reversed(lengths))
    if signal.shape != shape:
        raise ValueError(
            f"{location}/{item.data.name}: shape {signal.shape} != {shape}"
        )
    if item.timestamps is not None:
        timestamps = np.asarray(item.timestamps)
        count = int(np.prod(signal.shape[:-1], dtype=np.int64))
        if (
            timestamps.ndim != 1
            or timestamps.dtype.kind not in "iuf"
            or timestamps.size != count
        ):
            raise ValueError(
                f"{location}/timestamps: expected flat real numeric epoch seconds "
                f"with {count} entries"
            )
