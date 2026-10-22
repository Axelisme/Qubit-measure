"""Explicit legacy Labber import for offline storage migration."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from zcu_tools.datafile import (
    Axis,
    Container,
    ContainerAttrs,
    GroupSource,
    LabberMetadata,
    LabberPayload,
    Variable,
    cast_labber_values,
    load_grouped_labber_data,
    load_labber_logs,
)
from zcu_tools.datafile.grouped import GROUPED_VERSION_ATTR


@dataclass(frozen=True, kw_only=True)
class LegacyAxisSchema:
    """Declared historical axis name, unit and disk dtype, without coordinate values."""

    name: str
    unit: str
    dtype: np.dtype[np.generic]


@dataclass(frozen=True, kw_only=True)
class LegacyVariableSchema:
    """One historical log's explicit identity and inner-first channel declaration.

    variable is the output variable name. axes supplies ordered labels/units and
    dtypes, not shape or values. signal_name/unit/dtype identifies the legacy
    measured channel. These declarations never infer names or convert units.
    """

    variable: str
    axes: tuple[LegacyAxisSchema, ...]
    signal_name: str
    signal_unit: str
    signal_dtype: np.dtype[np.generic]


def load_legacy_labber_payload(
    source: Path, *, schema: tuple[LegacyVariableSchema, ...]
) -> tuple[Container, LabberMetadata]:
    """Read legacy Labber with explicit identities, returning (container, metadata).

    Marked files use their saved variable identities. Unmarked logs map to schema
    order: root, Log_2, Log_3, etc. Check counts, labels, units, shape and numeric
    dtype before converting inner-first axes to NumPy order. Do not infer names,
    units or cfg. Source remains unchanged. Labber metadata is returned separately;
    groups/labber retains the detached original, including per-trace timestamps.
    Missing/mismatched declarations raise ValueError; I/O errors propagate.
    """
    if not schema:
        raise ValueError(f"{source}: missing disk schema")
    with h5py.File(source, "r") as file:
        grouped = GROUPED_VERSION_ATTR in file.attrs
        image = file.id.get_file_image()
    if grouped:
        loaded = load_grouped_labber_data(
            str(source), required_variables=[item.variable for item in schema]
        )
        variables = {str(name): payload for name, payload in loaded.variables.items()}
        metadata = loaded.metadata
    else:
        logs = load_labber_logs(str(source))
        if len(logs) != len(schema):
            raise ValueError(
                f"{source}: log count {len(logs)} cannot identify multiple variables in schema of length {len(schema)}"
            )
        variables = {
            item.variable: log.payload for item, log in zip(schema, logs, strict=True)
        }
        metadata = logs[0].metadata
    converted: dict[str, Variable] = {}
    for item in schema:
        payload = variables[item.variable]
        context = f"{source}/{item.variable}"
        _validate_payload(payload, item, context)
        axes = tuple(
            Axis(
                actual.name,
                actual.unit,
                cast_labber_values(
                    actual.values, declared.dtype, context=f"{context}/{actual.name}"
                ),
            )
            for actual, declared in zip(payload.axes, item.axes, strict=True)
        )
        converted[item.variable] = Variable(
            cast_labber_values(payload.z, item.signal_dtype, context=context),
            tuple(reversed(axes)),
            payload.data.unit,
        )
    container = Container(
        converted,
        {},
        {"labber": GroupSource(image, "/")},
        ContainerAttrs("3.0", "migrations.legacy_labber", None, {}),
    )
    return container, metadata


def _validate_payload(
    payload: LabberPayload, schema: LegacyVariableSchema, context: str
) -> None:
    if len(payload.axes) != len(schema.axes):
        raise ValueError(f"{context}: axis count mismatch")
    shape: list[int] = []
    for actual, expected in zip(payload.axes, schema.axes, strict=True):
        if actual.name != expected.name:
            raise ValueError(
                f"{context}: axis label {actual.name!r}; expected {expected.name!r}"
            )
        if actual.unit != expected.unit:
            raise ValueError(
                f"{context}: axis unit {actual.unit!r}; expected {expected.unit!r}"
            )
        array = np.asarray(actual.values)
        if array.ndim != 1:
            raise ValueError(f"{context}: expected 1D axis")
        cast_labber_values(array, expected.dtype, context=context)
        shape.append(len(array))
    if payload.data.name != schema.signal_name:
        raise ValueError(
            f"{context}: signal label {payload.data.name!r}; expected {schema.signal_name!r}"
        )
    if payload.data.unit != schema.signal_unit:
        raise ValueError(
            f"{context}: signal unit {payload.data.unit!r}; expected {schema.signal_unit!r}"
        )
    if np.asarray(payload.z).shape != tuple(reversed(shape)):
        raise ValueError(f"{context}: signal shape mismatch")
