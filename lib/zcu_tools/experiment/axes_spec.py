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

from dataclasses import dataclass, fields, is_dataclass
from typing import Generic, TypeVar

import numpy as np

from zcu_tools.experiment.cfg_model import ExpCfgModel

__all__ = ["Axis", "ZSpec", "AxesSpec", "IDENTITY", "MHZ_TO_HZ", "US_TO_S"]

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
