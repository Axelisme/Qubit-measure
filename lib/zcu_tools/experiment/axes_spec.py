"""Declarative, per-experiment persistence spec (ADR-0063).

An ``AxesSpec`` decouples an experiment's in-memory frozen Result dataclass from
its on-disk (Labber) representation, and drives the base ``save()``/``load()``
symmetrically: it names each sweep axis + the log channel, carries the per-axis
unit scale, and supplies the typed Result builder (``result_type``) plus the cfg
restorer (``cfg_type.validate_or_warn``).

Axes are declared **inner-first** to match ``labber_io``'s native convention
(``z.shape == tuple(len(ax) for ax in reversed(axes))`` — inner axis last), so
``save`` and ``load`` are exact inverses with zero caller-side transpose.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from zcu_tools.datafile import (
    AxisSchema,
    DataVariable,
    ExperimentPayload,
    GroupedLabberData,
    LabberMetadata,
    LabberPayload,
    VariableSchema,
    cast_labber_values,
    format_ext,
    load_grouped_labber_data,
    save_grouped_labber_data,
    validate_labber_payload,
    write_labber,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord

__all__ = [
    "Axis",
    "ZSpec",
    "AxesSpec",
    "VariableAxisSpec",
    "VariableZSpec",
    "VariableSpec",
    "LoadedVariableData",
    "GroupedLoadData",
    "GroupedAxesSpec",
    "IDENTITY",
    "MHZ_TO_HZ",
    "US_TO_S",
]

T_Result = TypeVar("T_Result")
T_Config = TypeVar("T_Config", bound=ExpCfgModel)

# On-disk scale convention: disk_value = memory_value * scale; load divides back.
IDENTITY = 1.0
MHZ_TO_HZ = 1e6
US_TO_S = 1e-6


@dataclass(frozen=True)
class Axis:
    """One sweep axis of an experiment Result."""

    field_name: str  # the Result field holding this axis array (in memory units)
    label: str  # on-disk axis display name
    unit: str  # on-disk unit
    scale: float = IDENTITY  # disk = memory * scale
    dtype: type = np.float64  # in-memory dtype the loaded axis is cast back to


@dataclass(frozen=True)
class ZSpec:
    """Result channel: field_name selects the array; label/unit describe disk.

    dtype is the reconstructed memory dtype. scale maps memory values to SI
    disk values (disk = memory * scale), and must be finite and nonzero.
    """

    field_name: str
    label: str
    unit: str
    dtype: type = np.complex128
    scale: float = IDENTITY

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale == 0.0:
            raise ValueError("ZSpec scale must be finite and non-zero")


@dataclass(frozen=True)
class AxesSpec(Generic[T_Result, T_Config]):
    """Map a Result dataclass and cfg model to one stable variable identity.

    axes are inner-first memory-field/label/unit/scale declarations; z selects
    the signal field and its memory dtype/unit conversion. result_type rebuilds
    Result; cfg_type validates cfg. tag is the persisted experiment identity.
    data_variable is the explicit native group key, defaulting to data, never
    inferred from a label. cfg_schema_version is declaration-owned major.minor,
    defaulting to 1.0. Invalid Result fields fail at declaration time. Existing
    save/load remain Labber; native save_run/load_run consume this same spec.
    """

    axes: tuple[Axis, ...]  # inner-first
    z: ZSpec
    result_type: type[T_Result]
    cfg_type: type[T_Config]
    tag: str  # on-disk hierarchical tag, e.g. 'twotone/freq'
    data_variable: DataVariable = DataVariable(
        "data"
    )  # stable single-variable identity
    cfg_schema_version: str = "1.0"  # declaration-owned cfg major.minor

    def __post_init__(self) -> None:
        _validate_cfg_schema_version(self.cfg_schema_version)
        # Fast-Fail at declaration time: the spec must reference real Result fields.
        if not is_dataclass(self.result_type):
            raise TypeError(f"result_type {self.result_type!r} must be a dataclass")
        result_fields = {f.name for f in fields(self.result_type)}  # type: ignore[arg-type]
        declared = {ax.field_name for ax in self.axes} | {self.z.field_name}
        missing = declared - result_fields
        if missing:
            raise ValueError(
                f"AxesSpec field_name(s) {sorted(missing)} not on "
                f"{self.result_type.__name__} (has {sorted(result_fields)})"
            )


@dataclass(frozen=True)
class VariableAxisSpec:
    """Declare one inner-first axis for a grouped Data Variable.

    label/unit identify the persisted axis. field_name selects a 1D Result field;
    generated="arange" instead derives indices from the corresponding z dimension.
    Exactly one source is required. Nonzero scale multiplies memory values on save
    and divides persisted values on load. dtype is the reconstructed memory dtype.
    Invalid source selection or zero scale raises ValueError at construction.
    """

    label: str
    unit: str
    field_name: str | None = None
    scale: float = IDENTITY
    dtype: type = np.float64
    generated: Literal["arange"] | None = None

    @classmethod
    def generated_arange(
        cls,
        label: str,
        unit: str,
        *,
        dtype: type = np.int64,
    ) -> VariableAxisSpec:
        """Declare indices 0..N-1 with the given persisted label/unit and dtype."""
        return cls(label=label, unit=unit, dtype=dtype, generated="arange")

    def __post_init__(self) -> None:
        if (self.field_name is None) == (self.generated is None):
            raise ValueError(
                "VariableAxisSpec requires exactly one of field_name or generated"
            )
        if self.scale == 0.0:
            raise ValueError("VariableAxisSpec scale must be non-zero")


@dataclass(frozen=True)
class VariableZSpec:
    """Declare the measured channel for one grouped Data Variable.

    field_name selects the Result array; label/unit identify its persisted channel.
    Nonzero scale maps memory to disk units; dtype is the reconstructed memory
    dtype. index=None uses the entire array; otherwise select index on index_axis
    before saving. The remaining shape must match the reversed axis lengths.
    Empty field_name or zero scale raises ValueError at construction.
    """

    field_name: str
    label: str
    unit: str
    scale: float = IDENTITY
    dtype: type = np.float64
    index: int | None = None
    index_axis: int = -1

    def __post_init__(self) -> None:
        if not self.field_name:
            raise ValueError("VariableZSpec field_name must be non-empty")
        if self.scale == 0.0:
            raise ValueError("VariableZSpec scale must be non-zero")


@dataclass(frozen=True)
class LoadedVariableData:
    """Validated arrays for one loaded Data Variable.

    variable is its lowercase snake_case identity. axes are 1D arrays in
    inner-first order and memory units. z is the measured memory-unit array,
    shaped by the reversed axis lengths, with the VariableZSpec memory dtype.
    """

    variable: DataVariable
    axes: tuple[np.ndarray, ...]
    z: np.ndarray


@dataclass(frozen=True)
class GroupedLoadData(Generic[T_Config]):
    """Validated grouped payload plus reconstructed cfg snapshot.

    variables maps each DataVariable identity to its memory-unit loaded arrays.
    metadata contains shared file metadata. cfg_snapshot is the reconstructed
    experiment config, or None when the saved comment cannot validate as that cfg.
    """

    variables: Mapping[DataVariable, LoadedVariableData]
    metadata: LabberMetadata
    cfg_snapshot: T_Config | None

    def variable(self, variable: str | DataVariable) -> LoadedVariableData:
        """Get loaded arrays by snake_case identity; invalid/missing names raise ValueError."""
        data_variable = DataVariable(variable)
        try:
            return self.variables[data_variable]
        except KeyError:
            raise ValueError(
                f"loaded grouped data is missing variable {variable!r}"
            ) from None


@dataclass(frozen=True)
class VariableSpec:
    """Map one named Data Variable between Result fields and a Labber payload.

    variable is its lowercase snake_case identity; invalid names raise ValueError
    at construction. axes declares inner-first axes; z declares the measured
    channel, memory dtype and unit conversion. Arrays use reversed-axis shape.
    """

    variable: str | DataVariable
    axes: tuple[VariableAxisSpec, ...]
    z: VariableZSpec

    def __post_init__(self) -> None:
        DataVariable(self.variable)

    @property
    def data_variable(self) -> DataVariable:
        """Return the validated snake_case identity for this mapping."""
        return DataVariable(self.variable)

    def validate_result_fields(
        self, result_fields: set[str], result_type_name: str
    ) -> None:
        """Check declared fields against Result field names; missing fields raise ValueError.

        result_type_name labels that error with the caller\'s Result type.
        """
        declared = {self.z.field_name}
        declared.update(axis.field_name for axis in self.axes if axis.field_name)
        missing = declared - result_fields
        if missing:
            raise ValueError(
                f"VariableSpec {self.data_variable!r} field_name(s) "
                f"{sorted(missing)} not on {result_type_name} "
                f"(has {sorted(result_fields)})"
            )

    def native_schema(self) -> VariableSchema:
        """Return this mapping's SI disk labels, units and numeric dtypes.

        NumPy scalar multiplication determines disk dtype after memory casting.
        Generated arange axes keep their declared integer dtype without scaling.
        This declaration does not inspect Result values or modify arrays.
        """
        return VariableSchema(
            variable=self.data_variable,
            axes=tuple(
                AxisSchema(
                    name=axis.label,
                    unit=axis.unit,
                    dtype=(
                        np.dtype(axis.dtype)
                        if axis.generated == "arange"
                        else (np.empty(0, dtype=axis.dtype) * axis.scale).dtype
                    ),
                )
                for axis in self.axes
            ),
            signal_name=self.z.label,
            signal_unit=self.z.unit,
            signal_dtype=(np.empty(0, dtype=self.z.dtype) * self.z.scale).dtype,
        )

    def payload_from_result(self, result: object, *, context: str) -> LabberPayload:
        """Map declared Result arrays into disk units; context labels validation errors.

        Missing fields raise AttributeError. Invalid dtype, index or axis/array
        shape raises ValueError. The returned payload retains inner-first axes.
        """
        z_values = self._z_from_result(result, context=context)
        axes = [
            (
                axis.label,
                axis.unit,
                self._axis_from_result(
                    result,
                    axis,
                    z_shape=z_values.shape,
                    axis_index=index,
                    context=context,
                ),
            )
            for index, axis in enumerate(self.axes)
        ]
        self._validate_shape(
            z_values, [axis_values for _, _, axis_values in axes], context
        )
        return LabberPayload((self.z.label, self.z.unit, z_values), axes=axes)

    def loaded_from_payload(
        self, payload: LabberPayload, *, context: str
    ) -> LoadedVariableData:
        """Validate persisted labels/units/shape and restore memory units and dtypes.

        context labels ValueError for invalid labels, units, arrays or generated
        indices. Return loaded arrays for this mapping\'s variable identity.
        """
        validate_labber_payload(payload, schema=self.native_schema(), context=context)

        loaded_axes: list[np.ndarray] = []
        for index, (loaded_axis, expected_axis) in enumerate(
            zip(payload.axes, self.axes, strict=True)
        ):
            axis_values = _cast_memory_values(
                np.asarray(loaded_axis.values) / expected_axis.scale,
                expected_axis.dtype,
                context=f"{context} axis {index}",
            )
            if expected_axis.generated == "arange":
                expected_values = np.arange(
                    axis_values.shape[0], dtype=axis_values.dtype
                )
                if not np.array_equal(axis_values, expected_values):
                    raise ValueError(f"{context} axis {index} must equal arange(N)")
            loaded_axes.append(axis_values)

        z_values = _cast_memory_values(
            np.asarray(payload.z) / self.z.scale,
            self.z.dtype,
            context=f"{context} z channel",
        )
        return LoadedVariableData(
            variable=self.data_variable,
            axes=tuple(loaded_axes),
            z=z_values,
        )

    def _z_from_result(self, result: object, *, context: str) -> np.ndarray:
        values = np.asarray(getattr(result, self.z.field_name))
        if self.z.index is not None:
            try:
                values = np.take(values, self.z.index, axis=self.z.index_axis)
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"{context} cannot select index {self.z.index} from "
                    f"field {self.z.field_name!r} on axis {self.z.index_axis}"
                ) from exc
        memory_values = _cast_memory_values(
            values,
            self.z.dtype,
            context=f"{context} result field {self.z.field_name!r}",
        )
        return np.asarray(memory_values * self.z.scale)

    def _axis_from_result(
        self,
        result: object,
        axis: VariableAxisSpec,
        *,
        z_shape: tuple[int, ...],
        axis_index: int,
        context: str,
    ) -> np.ndarray:
        if axis.generated == "arange":
            if len(z_shape) <= axis_index:
                raise ValueError(
                    f"{context} cannot generate axis {axis_index} from "
                    f"{len(z_shape)}D z data"
                )
            return np.arange(z_shape[-1 - axis_index], dtype=np.dtype(axis.dtype))

        assert axis.field_name is not None
        values = _cast_memory_values(
            getattr(result, axis.field_name),
            axis.dtype,
            context=f"{context} result axis field {axis.field_name!r}",
        )
        if values.ndim != 1:
            raise ValueError(
                f"{context} result axis field {axis.field_name!r} is "
                f"{values.ndim}D; expected 1D"
            )
        return np.asarray(values * axis.scale)

    def _validate_shape(
        self, z_values: np.ndarray, axis_values: list[np.ndarray], context: str
    ) -> None:
        expected_shape = tuple(axis.shape[0] for axis in reversed(axis_values))
        if z_values.shape != expected_shape:
            raise ValueError(
                f"{context} z shape {z_values.shape} != expected {expected_shape}"
            )


@dataclass(frozen=True)
class GroupedAxesSpec(Generic[T_Result, T_Config]):
    """Declare the grouped persistence mapping for one experiment (ADR-0063).

    variables is a nonempty ordered tuple of unique VariableSpec declarations.
    result_type is the Result dataclass; cfg_type validates saved cfg comments.
    tag is the default file tag. result_builder rebuilds result_type from loaded
    memory-unit arrays; result_validator, when supplied, checks results on save
    and load. Invalid declarations raise ValueError; non-dataclass types or wrong
    builder return types raise TypeError. cfg_schema_version is the declaration's
    major.minor cfg version, defaulting to 1.0. Existing Labber save/load require
    a common grid; native persistence permits different grids for each variable.
    """

    variables: tuple[VariableSpec, ...]
    result_type: type[T_Result]
    cfg_type: type[T_Config]
    tag: str
    result_builder: Callable[[GroupedLoadData[T_Config]], T_Result]
    result_validator: Callable[[T_Result], None] | None = None
    cfg_schema_version: str = "1.0"  # declaration-owned cfg major.minor

    def __post_init__(self) -> None:
        _validate_cfg_schema_version(self.cfg_schema_version)
        if not self.variables:
            raise ValueError("GroupedAxesSpec requires at least one variable")
        if not is_dataclass(self.result_type):
            raise TypeError(f"result_type {self.result_type!r} must be a dataclass")
        result_fields = {f.name for f in fields(self.result_type)}  # type: ignore[arg-type]

        seen: set[DataVariable] = set()
        for variable in self.variables:
            data_variable = variable.data_variable
            if data_variable in seen:
                raise ValueError(f"duplicate grouped data variable {data_variable!r}")
            seen.add(data_variable)
            variable.validate_result_fields(result_fields, self.result_type.__name__)

    @property
    def required_variables(self) -> tuple[DataVariable, ...]:
        """Return required variable identities in declaration order."""
        return tuple(variable.data_variable for variable in self.variables)

    def payloads_from_result(self, result: T_Result) -> dict[str, LabberPayload]:
        if self.result_validator is not None:
            self.result_validator(result)
        return {
            str(variable.data_variable): variable.payload_from_result(
                result,
                context=(
                    f"{self.result_type.__name__} grouped variable "
                    f"{str(variable.data_variable)!r}"
                ),
            )
            for variable in self.variables
        }

    def save_grouped_result(
        self,
        filepath: str,
        result: T_Result,
        *,
        comment: str = "",
        tag: str | None = None,
    ) -> str:
        return save_grouped_labber_data(
            filepath,
            self.payloads_from_result(result),
            metadata=LabberMetadata(comment=comment, tags=tag or self.tag),
        )

    def save(
        self,
        source: RunRecord[T_Config, T_Result],
        destination: Path,
        *,
        comment: str | None = None,
        tag: str | None = None,
    ) -> None:
        """Map this explicit record to one common-grid Labber export.

        source.cfg must be present. destination follows the existing Labber
        extension normalization. comment is optional user text; tag overrides
        this declaration's tag. Invalid cfg JSON or data raises ValueError;
        existing files raise FileExistsError and I/O errors propagate. No native
        file is written and no directory/name reservation is performed.
        """
        if source.cfg is None:
            raise ValueError("Cannot save a RunRecord without cfg")
        from zcu_tools.experiment.utils import make_labber_cfg_snapshot

        payload = ExperimentPayload(
            variables={
                DataVariable(variable): signal
                for variable, signal in self.payloads_from_result(source.result).items()
            },
            metadata=LabberMetadata(tags=tag or self.tag),
            representation="grouped",
        )
        write_labber(
            Path(format_ext(str(destination))),
            payload,
            cfg=make_labber_cfg_snapshot(
                source.cfg, schema_version=self.cfg_schema_version
            ),
            comment=comment,
        )

    def load(self, source: Path) -> RunRecord[T_Config, T_Result]:
        """Read canonical grouped Labber at an exact local path into a record.

        Required variables, axes, units or shape mismatches raise ValueError;
        I/O errors propagate. Missing cfg or non-envelope comment text returns
        cfg=None with valid Result data. Recognized envelopes with invalid
        cfg/comment/timestamp field types raise ValueError. A valid envelope
        whose cfg object fails cfg_type validation warns and retains Result data
        with cfg=None. This entry does not guess native or legacy layouts.
        """
        grouped = load_grouped_labber_data(
            str(source),
            required_variables=self.required_variables,
        )
        return self.record_from_grouped_data(grouped, source=str(source))

    def record_from_grouped_data(
        self,
        grouped: GroupedLabberData,
        *,
        source: str | None = None,
    ) -> RunRecord[T_Config, T_Result]:
        self._validate_grouped_variables(grouped)
        cfg_snapshot = self._cfg_from_comment(grouped.metadata.comment, source=source)
        loaded_variables = {
            variable.data_variable: variable.loaded_from_payload(
                grouped.variables[variable.data_variable],
                context=(
                    f"{self.result_type.__name__} grouped variable "
                    f"{str(variable.data_variable)!r}"
                ),
            )
            for variable in self.variables
        }
        load_data = GroupedLoadData(
            variables=loaded_variables,
            metadata=grouped.metadata,
            cfg_snapshot=cfg_snapshot,
        )
        result = self.result_builder(load_data)
        if not isinstance(result, self.result_type):
            raise TypeError(
                f"GroupedAxesSpec builder returned {type(result).__name__}; "
                f"expected {self.result_type.__name__}"
            )
        if self.result_validator is not None:
            self.result_validator(result)
        return RunRecord(cfg=cfg_snapshot, result=result)

    def _validate_grouped_variables(self, grouped: GroupedLabberData) -> None:
        expected = set(self.required_variables)
        present = set(grouped.variables)
        missing = expected - present
        unknown = present - expected
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"missing required data variable(s): {names}")
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown data variable(s): {names}")

    def _cfg_from_comment(
        self, comment: str, *, source: str | None = None
    ) -> T_Config | None:
        if not comment:
            return None
        from zcu_tools.experiment.utils import parse_comment

        cfg_dict, _, _ = parse_comment(comment)
        if cfg_dict is None:
            return None
        return self.cfg_type.validate_or_warn(
            cfg_dict,
            source=source or "<grouped>",
        )


def _validate_cfg_schema_version(version: str) -> None:
    if re.fullmatch(r"[0-9]+\.[0-9]+", version) is None:
        raise ValueError(f"cfg_schema_version must be major.minor, got {version!r}")


def _cast_memory_values(values: ArrayLike, dtype: type, *, context: str) -> np.ndarray:
    target_dtype = np.dtype(dtype)
    if target_dtype.kind in {"i", "u"}:
        # Integer coordinates use the typed mapping's existing rounding contract.
        real_array = cast_labber_values(values, np.dtype(np.float64), context=context)
        rounded = np.round(real_array)
        if not np.allclose(real_array, rounded):
            raise ValueError(f"{context} values must be integers")
        return rounded.astype(target_dtype)
    return cast_labber_values(values, target_dtype, context=context)
