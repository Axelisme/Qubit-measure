"""Pure projection of historical device Info snapshots to scalar values."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceScalar:
    """A device's finite native value and nonempty unit, without conversion."""

    value: float
    unit: str


def snapshot_scalar_values(
    snapshot: Mapping[str, Mapping[str, object]], names: Iterable[str]
) -> dict[str, DeviceScalar]:
    """Read requested historical Info value/value_unit fields without I/O.

    Missing names are omitted, not replaced by zero. Present names must contain
    a finite int/float value (not bool); otherwise ValueError names the device.
    Missing value_unit means native. Invalid/empty units also raise ValueError.
    Returned values are detached and no input or current driver is changed.
    """
    result: dict[str, DeviceScalar] = {}
    for name in names:
        if name not in snapshot:
            continue
        info = snapshot[name]
        value = info.get("value")
        unit = info.get("value_unit", "native")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Device {name!r} needs a finite numeric scalar value")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"Device {name!r} has an invalid value_unit")
        result[name] = DeviceScalar(float(value), unit)
    return result
