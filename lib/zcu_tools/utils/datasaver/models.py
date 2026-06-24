"""In-memory models for Labber-style experiment data."""

from __future__ import annotations

import re
from collections import namedtuple
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Axis = namedtuple("Axis", ["name", "unit", "values"])

_ROLE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def as_tag_list(tags: str | Sequence[str] | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags]
    return [str(tag) for tag in tags]


def unpack_triple(triple: Any, what: str) -> tuple[str, str, Any]:
    """Validate and unpack a ``(name, unit, values)`` triple."""
    try:
        name, unit, values = triple
    except (TypeError, ValueError):
        raise ValueError(
            f"`{what}` must be a (name, unit, values) tuple, got {triple!r}"
        ) from None
    return str(name), str(unit), values


def as_axis(triple: Axis | tuple[str, str, Any], what: str) -> Axis:
    """Coerce a ``(name, unit, values)`` tuple into an ``Axis``."""
    if isinstance(triple, Axis):
        return Axis(str(triple.name), str(triple.unit), triple.values)
    name, unit, values = unpack_triple(triple, what)
    return Axis(name, unit, values)


@dataclass(slots=True)
class LabberMetadata:
    """Metadata shared by one experiment data file."""

    comment: str = ""
    tags: list[str] = field(default_factory=list)
    project: str = ""
    user: str = ""
    creation_time: float | None = None

    def __init__(
        self,
        *,
        comment: str = "",
        tags: str | Sequence[str] | None = None,
        project: str = "",
        user: str = "",
        creation_time: float | None = None,
    ) -> None:
        self.comment = str(comment)
        self.tags = as_tag_list(tags)
        self.project = str(project)
        self.user = str(user)
        self.creation_time = creation_time


@dataclass(slots=True)
class LabberPayload:
    """One Labber dataset: log channel, inner-first axes, and timestamps."""

    data: Axis
    axes: list[Axis]
    timestamps: Any = None

    def __init__(
        self,
        data: Axis | tuple[str, str, Any],
        axes: Sequence[Axis | tuple[str, str, Any]],
        *,
        timestamps: Any = None,
    ) -> None:
        self.data = as_axis(data, "data")
        self.axes = [as_axis(axis, f"axes[{i}]") for i, axis in enumerate(axes)]
        self.timestamps = timestamps

    @property
    def z(self) -> Any:
        """The data array, alias for ``self.data.values``."""
        return self.data.values

    @property
    def x(self) -> Any:
        """Inner-axis values, or None."""
        return self.axes[0].values if len(self.axes) >= 1 else None

    @property
    def y(self) -> Any:
        """Second-axis values, or None."""
        return self.axes[1].values if len(self.axes) >= 2 else None

    @property
    def w(self) -> Any:
        """Third-axis values, or None."""
        return self.axes[2].values if len(self.axes) >= 3 else None

    def __iter__(self):
        return iter((self.z, self.x, self.y))

    def get_log_channels(self) -> list[dict[str, Any]]:
        vals = self.data.values
        complex_ = bool(
            np.iscomplexobj(np.asarray(vals[0] if isinstance(vals, list) else vals))
        )
        return [
            {
                "name": self.data.name,
                "unit": self.data.unit,
                "complex": complex_,
                "vector": isinstance(vals, list),
            }
        ]

    def get_step_channels(self) -> list[dict[str, Any]]:
        return [
            {
                "name": axis.name,
                "unit": axis.unit,
                "values": np.asarray(axis.values),
                "complex": False,
                "vector": False,
            }
            for axis in self.axes
        ]

    def get_num_entries(self) -> int:
        vals = self.data.values
        if isinstance(vals, list):
            return len(vals)
        if getattr(vals, "ndim", 1) <= 1:
            return 1
        return int(np.prod(vals.shape[:-1]))


class LabberData:
    """Single-file Labber dataset with payload plus metadata.

    The constructor keeps the historical ``LabberData(data, axes, ...)`` form
    while also accepting explicit ``payload=`` and ``metadata=`` values.
    """

    __slots__ = ("payload", "metadata")

    def __init__(
        self,
        data: Axis | tuple[str, str, Any] | None = None,
        axes: Sequence[Axis | tuple[str, str, Any]] | None = None,
        *,
        payload: LabberPayload | None = None,
        metadata: LabberMetadata | None = None,
        comment: str = "",
        tags: str | Sequence[str] | None = None,
        project: str = "",
        user: str = "",
        timestamps: Any = None,
        creation_time: float | None = None,
    ) -> None:
        if payload is None:
            if data is None or axes is None:
                raise TypeError("LabberData requires data and axes or payload")
            payload = LabberPayload(data, axes, timestamps=timestamps)
        elif data is not None or axes is not None:
            raise TypeError("Pass either payload or data/axes, not both")
        elif timestamps is not None:
            raise TypeError("Pass timestamps through LabberPayload when using payload")

        if metadata is None:
            metadata = LabberMetadata(
                comment=comment,
                tags=tags,
                project=project,
                user=user,
                creation_time=creation_time,
            )
        elif any(
            (
                comment,
                tags is not None,
                project,
                user,
                creation_time is not None,
            )
        ):
            raise TypeError("Pass either metadata or metadata fields, not both")

        self.payload = payload
        self.metadata = metadata

    @property
    def data(self) -> Axis:
        return self.payload.data

    @data.setter
    def data(self, value: Axis | tuple[str, str, Any]) -> None:
        self.payload.data = as_axis(value, "data")

    @property
    def axes(self) -> list[Axis]:
        return self.payload.axes

    @axes.setter
    def axes(self, value: Sequence[Axis | tuple[str, str, Any]]) -> None:
        self.payload.axes = [
            as_axis(axis, f"axes[{i}]") for i, axis in enumerate(value)
        ]

    @property
    def timestamps(self) -> Any:
        return self.payload.timestamps

    @timestamps.setter
    def timestamps(self, value: Any) -> None:
        self.payload.timestamps = value

    @property
    def comment(self) -> str:
        return self.metadata.comment

    @comment.setter
    def comment(self, value: str) -> None:
        self.metadata.comment = str(value)

    @property
    def tags(self) -> list[str]:
        return self.metadata.tags

    @tags.setter
    def tags(self, value: str | Sequence[str] | None) -> None:
        self.metadata.tags = as_tag_list(value)

    @property
    def project(self) -> str:
        return self.metadata.project

    @project.setter
    def project(self, value: str) -> None:
        self.metadata.project = str(value)

    @property
    def user(self) -> str:
        return self.metadata.user

    @user.setter
    def user(self, value: str) -> None:
        self.metadata.user = str(value)

    @property
    def creation_time(self) -> float | None:
        return self.metadata.creation_time

    @creation_time.setter
    def creation_time(self, value: float | None) -> None:
        self.metadata.creation_time = value

    @property
    def z(self) -> Any:
        return self.payload.z

    @property
    def x(self) -> Any:
        return self.payload.x

    @property
    def y(self) -> Any:
        return self.payload.y

    @property
    def w(self) -> Any:
        return self.payload.w

    def __iter__(self):
        return iter((self.z, self.x, self.y))

    def get_log_channels(self) -> list[dict[str, Any]]:
        return self.payload.get_log_channels()

    def get_step_channels(self) -> list[dict[str, Any]]:
        return self.payload.get_step_channels()

    def get_num_entries(self) -> int:
        return self.payload.get_num_entries()

    def save(self, path: str) -> str:
        if isinstance(self.data.values, list):
            from .labber import _save_labber_trace_data

            return _save_labber_trace_data(path, self)
        from .labber import _save_labber_data

        return _save_labber_data(path, self)

    @classmethod
    def load(cls, path: str) -> LabberData:
        from .labber import _load_labber_data

        return _load_labber_data(path)

    def __repr__(self) -> str:
        vals = self.data.values
        shape = (
            f"[{len(vals)} ragged]"
            if isinstance(vals, list)
            else (None if vals is None else vals.shape)
        )
        return (
            f"LabberData(data={self.data.name!r} shape={shape}, "
            f"axes={[a.name for a in self.axes]!r}, tags={self.tags!r})"
        )


class DatasetRole(str):
    """Lowercase snake_case dataset role value."""

    def __new__(cls, value: str | DatasetRole) -> DatasetRole:
        text = str(value)
        if not _ROLE_RE.fullmatch(text):
            raise ValueError(
                f"DatasetRole must be lowercase snake_case (got {value!r})"
            )
        return str.__new__(cls, text)


@dataclass(slots=True)
class GroupedLabberData:
    """Grouped experiment dataset: role payloads plus shared metadata."""

    roles: dict[DatasetRole, LabberPayload]
    metadata: LabberMetadata

    def __init__(
        self,
        roles: Mapping[str | DatasetRole, LabberPayload],
        *,
        metadata: LabberMetadata | None = None,
    ) -> None:
        if not roles:
            raise ValueError("GroupedLabberData requires at least one role")

        normalized: dict[DatasetRole, LabberPayload] = {}
        for raw_role, payload in roles.items():
            role = DatasetRole(raw_role)
            if role in normalized:
                raise ValueError(f"duplicate dataset role {role!r}")
            if isinstance(payload, LabberData):
                raise TypeError(
                    "GroupedLabberData role values must be LabberPayload, "
                    "not LabberData"
                )
            if not isinstance(payload, LabberPayload):
                raise TypeError(
                    "GroupedLabberData role values must be LabberPayload "
                    f"(role {role!r} got {type(payload).__name__})"
                )
            normalized[role] = payload

        if metadata is not None and not isinstance(metadata, LabberMetadata):
            raise TypeError(
                "GroupedLabberData metadata must be LabberMetadata "
                f"(got {type(metadata).__name__})"
            )

        self.roles = normalized
        self.metadata = metadata if metadata is not None else LabberMetadata()
