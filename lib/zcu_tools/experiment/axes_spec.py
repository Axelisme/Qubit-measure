"""Declarative, per-experiment persistence spec (ADR-0027).

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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Generic, Literal, TypeVar

import numpy as np

from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.utils.datasaver import (
    DatasetRole,
    GroupedLabberData,
    LabberMetadata,
    LabberPayload,
    load_grouped_labber_data,
    save_grouped_labber_data,
)

__all__ = [
    "Axis",
    "ZSpec",
    "AxesSpec",
    "RoleAxisSpec",
    "RoleZSpec",
    "RoleSpec",
    "LoadedRoleData",
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
    """The log (z) channel of an experiment Result."""

    field_name: str
    label: str
    unit: str
    dtype: type = np.complex128


@dataclass(frozen=True)
class AxesSpec(Generic[T_Result, T_Config]):
    """The full per-experiment persistence declaration (see module docstring)."""

    axes: tuple[Axis, ...]  # inner-first
    z: ZSpec
    result_type: type[T_Result]
    cfg_type: type[T_Config]
    tag: str  # on-disk hierarchical tag, e.g. 'twotone/freq'

    def __post_init__(self) -> None:
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
        if "cfg_snapshot" not in result_fields:
            raise ValueError(
                f"{self.result_type.__name__} must declare a 'cfg_snapshot' field"
            )


@dataclass(frozen=True)
class RoleAxisSpec:
    """One inner-first axis for a grouped Dataset Role."""

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
    ) -> RoleAxisSpec:
        return cls(label=label, unit=unit, dtype=dtype, generated="arange")

    def __post_init__(self) -> None:
        if (self.field_name is None) == (self.generated is None):
            raise ValueError(
                "RoleAxisSpec requires exactly one of field_name or generated"
            )
        if self.scale == 0.0:
            raise ValueError("RoleAxisSpec scale must be non-zero")


@dataclass(frozen=True)
class RoleZSpec:
    """The z/data channel for one grouped Dataset Role."""

    field_name: str
    label: str
    unit: str
    scale: float = IDENTITY
    dtype: type = np.float64
    index: int | None = None
    index_axis: int = -1

    def __post_init__(self) -> None:
        if not self.field_name:
            raise ValueError("RoleZSpec field_name must be non-empty")
        if self.scale == 0.0:
            raise ValueError("RoleZSpec scale must be non-zero")


@dataclass(frozen=True)
class LoadedRoleData:
    """Validated, memory-unit arrays for one loaded Dataset Role."""

    role: DatasetRole
    axes: tuple[np.ndarray, ...]
    z: np.ndarray


@dataclass(frozen=True)
class GroupedLoadData(Generic[T_Config]):
    """Validated grouped payload plus reconstructed cfg snapshot."""

    roles: Mapping[DatasetRole, LoadedRoleData]
    metadata: LabberMetadata
    cfg_snapshot: T_Config | None

    def role(self, role: str | DatasetRole) -> LoadedRoleData:
        dataset_role = DatasetRole(role)
        try:
            return self.roles[dataset_role]
        except KeyError:
            raise ValueError(f"loaded grouped data is missing role {role!r}") from None


@dataclass(frozen=True)
class RoleSpec:
    """Mechanical Result-field mapping for one grouped Dataset Role."""

    role: str | DatasetRole
    axes: tuple[RoleAxisSpec, ...]
    z: RoleZSpec

    def __post_init__(self) -> None:
        DatasetRole(self.role)

    @property
    def dataset_role(self) -> DatasetRole:
        return DatasetRole(self.role)

    def validate_result_fields(
        self, result_fields: set[str], result_type_name: str
    ) -> None:
        declared = {self.z.field_name}
        declared.update(axis.field_name for axis in self.axes if axis.field_name)
        missing = declared - result_fields
        if missing:
            raise ValueError(
                f"RoleSpec {self.dataset_role!r} field_name(s) "
                f"{sorted(missing)} not on {result_type_name} "
                f"(has {sorted(result_fields)})"
            )

    def payload_from_result(self, result: object, *, context: str) -> LabberPayload:
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
    ) -> LoadedRoleData:
        if len(payload.axes) != len(self.axes):
            raise ValueError(
                f"{context} has {len(payload.axes)} axes; expected {len(self.axes)}"
            )

        loaded_axes: list[np.ndarray] = []
        for index, (loaded_axis, expected_axis) in enumerate(
            zip(payload.axes, self.axes, strict=True)
        ):
            if loaded_axis.name != expected_axis.label:
                raise ValueError(
                    f"{context} axis {index} label is {loaded_axis.name!r}; "
                    f"expected {expected_axis.label!r}"
                )
            if loaded_axis.unit != expected_axis.unit:
                raise ValueError(
                    f"{context} axis {index} unit is {loaded_axis.unit!r}; "
                    f"expected {expected_axis.unit!r}"
                )

            axis_values = _cast_memory_values(
                np.asarray(loaded_axis.values) / expected_axis.scale,
                expected_axis.dtype,
                context=f"{context} axis {index}",
            )
            if axis_values.ndim != 1:
                raise ValueError(
                    f"{context} axis {index} is {axis_values.ndim}D; expected 1D"
                )
            if expected_axis.generated == "arange":
                expected_values = np.arange(
                    axis_values.shape[0], dtype=axis_values.dtype
                )
                if not np.array_equal(axis_values, expected_values):
                    raise ValueError(f"{context} axis {index} must equal arange(N)")
            loaded_axes.append(axis_values)

        if payload.data.name != self.z.label:
            raise ValueError(
                f"{context} z channel label is {payload.data.name!r}; "
                f"expected {self.z.label!r}"
            )
        if payload.data.unit != self.z.unit:
            raise ValueError(
                f"{context} z channel unit is {payload.data.unit!r}; "
                f"expected {self.z.unit!r}"
            )

        z_values = _cast_memory_values(
            np.asarray(payload.z) / self.z.scale,
            self.z.dtype,
            context=f"{context} z channel",
        )
        self._validate_shape(z_values, loaded_axes, context)
        return LoadedRoleData(
            role=self.dataset_role,
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
        axis: RoleAxisSpec,
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
    """Experiment-level grouped persistence contract (ADR-0027)."""

    roles: tuple[RoleSpec, ...]
    result_type: type[T_Result]
    cfg_type: type[T_Config]
    tag: str
    result_builder: Callable[[GroupedLoadData[T_Config]], T_Result]
    result_validator: Callable[[T_Result], None] | None = None

    def __post_init__(self) -> None:
        if not self.roles:
            raise ValueError("GroupedAxesSpec requires at least one role")
        if not is_dataclass(self.result_type):
            raise TypeError(f"result_type {self.result_type!r} must be a dataclass")
        result_fields = {f.name for f in fields(self.result_type)}  # type: ignore[arg-type]
        if "cfg_snapshot" not in result_fields:
            raise ValueError(
                f"{self.result_type.__name__} must declare a 'cfg_snapshot' field"
            )

        seen: set[DatasetRole] = set()
        for role in self.roles:
            dataset_role = role.dataset_role
            if dataset_role in seen:
                raise ValueError(f"duplicate grouped dataset role {dataset_role!r}")
            seen.add(dataset_role)
            role.validate_result_fields(result_fields, self.result_type.__name__)

    @property
    def required_roles(self) -> tuple[DatasetRole, ...]:
        return tuple(role.dataset_role for role in self.roles)

    def payloads_from_result(self, result: T_Result) -> dict[str, LabberPayload]:
        if self.result_validator is not None:
            self.result_validator(result)
        return {
            str(role.dataset_role): role.payload_from_result(
                result,
                context=(
                    f"{self.result_type.__name__} grouped role "
                    f"{str(role.dataset_role)!r}"
                ),
            )
            for role in self.roles
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

    def save_experiment_result(
        self,
        filepath: str,
        result: T_Result,
        *,
        comment: str | None = None,
        tag: str | None = None,
        make_comment_fn: Callable[[T_Config, str | None], str] | None = None,
    ) -> str:
        cfg = getattr(result, "cfg_snapshot")
        if cfg is None:
            raise ValueError("cfg_snapshot is None")
        if make_comment_fn is None:
            from zcu_tools.experiment.utils import make_comment

            make_comment_fn = make_comment
        return self.save_grouped_result(
            filepath,
            result,
            comment=make_comment_fn(cfg, comment),
            tag=tag,
        )

    def load_result(self, filepath: str) -> T_Result:
        grouped = load_grouped_labber_data(
            filepath,
            required_roles=self.required_roles,
        )
        return self.result_from_grouped_data(grouped, source=filepath)

    def result_from_grouped_data(
        self,
        grouped: GroupedLabberData,
        *,
        source: str | None = None,
    ) -> T_Result:
        self._validate_grouped_roles(grouped)
        cfg_snapshot = self._cfg_from_comment(grouped.metadata.comment, source=source)
        loaded_roles = {
            role.dataset_role: role.loaded_from_payload(
                grouped.roles[role.dataset_role],
                context=(
                    f"{self.result_type.__name__} grouped role "
                    f"{str(role.dataset_role)!r}"
                ),
            )
            for role in self.roles
        }
        load_data = GroupedLoadData(
            roles=loaded_roles,
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
        return result

    def _validate_grouped_roles(self, grouped: GroupedLabberData) -> None:
        expected = set(self.required_roles)
        present = set(grouped.roles)
        missing = expected - present
        unknown = present - expected
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"missing required dataset role(s): {names}")
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown dataset role(s): {names}")

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


def _cast_memory_values(values: Any, dtype: type, *, context: str) -> np.ndarray:
    target_dtype = np.dtype(dtype)
    array = np.asarray(values)
    if target_dtype.kind != "c" and np.iscomplexobj(array):
        if np.any(np.imag(array) != 0.0):
            raise ValueError(
                f"{context} contains non-zero imaginary component; "
                f"cannot load as {target_dtype}"
            )
        array = np.real(array)

    if target_dtype.kind in {"i", "u"}:
        real_array = np.asarray(array, dtype=np.float64)
        rounded = np.round(real_array)
        if not np.allclose(real_array, rounded):
            raise ValueError(f"{context} values must be integers")
        return rounded.astype(target_dtype)

    return np.asarray(array, dtype=target_dtype)
