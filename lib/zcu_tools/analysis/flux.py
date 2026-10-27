"""Linear multi-device flux coordinates, independent of records and components."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType

from zcu_tools.cfg_model import validate_device_name


@dataclass(frozen=True)
class DevicePeriod:
    """A signed nonzero span per Phi0 in the named native instrument unit.

    period must be a finite real number, not a boolean. unit is a nonblank
    string (for example A, V or native); no unit conversion is performed.
    Invalid fields raise ValueError at construction.
    """

    period: float
    unit: str = "native"

    def __post_init__(self) -> None:
        if _finite_value(self.period, field="period") == 0:
            raise ValueError("period must be nonzero (signed device units per Phi0)")
        _require_unit(self.unit)


@dataclass(frozen=True)
class FluxFrame:
    """Zero-output flux offset in Phi0 and each device's signed native period.

    offset must be finite and real, not boolean, and is not reduced modulo 1.
    devices maps nonblank instrument names without ASCII dots to DevicePeriod.
    An empty mapping describes a constant flux. Construction snapshots it;
    valid names, including surrounding spaces, are preserved unchanged.
    Invalid fields raise ValueError. Missing device values contribute zero.
    """

    offset: float
    devices: Mapping[str, DevicePeriod]

    def __post_init__(self) -> None:
        _finite_value(self.offset, field="offset")
        for name, device in self.devices.items():
            validate_device_name(name)
            _require_device_period(device, name)
        object.__setattr__(self, "devices", MappingProxyType(dict(self.devices)))


@dataclass(frozen=True)
class ComponentFluxContext:
    """Caller-projected component calibration, default device and native units.

    frame is None when the component offset is uncalibrated, never a zero-offset
    substitute. default_device is the lab's optional preferred instrument; it
    need not have calibration. units maps device names to nonempty units, with
    exactly the frame.devices keys when a frame exists. Construction detaches
    units and rejects invalid names/units or inconsistent keys with ValueError.
    """

    frame: FluxFrame | None
    default_device: str | None
    units: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.default_device is not None:
            validate_device_name(self.default_device)
        for name, unit in self.units.items():
            validate_device_name(name)
            _require_unit(unit)
        if self.frame is not None and self.units.keys() != self.frame.devices.keys():
            raise ValueError("context units must have exactly the frame device keys")
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))


def flux_from_device_values(frame: FluxFrame, values: Mapping[str, float]) -> float:
    """Return offset + sum(value / period) in dimensionless Phi0.

    values uses each device's declared native unit. Missing devices contribute
    zero and extra devices are ignored. Used boolean, non-real or nonfinite
    values raise ValueError naming the device. Inputs remain unchanged.
    """
    return frame.offset + sum(
        _finite_value(values[name], field=f"device {name!r} value") / device.period
        for name, device in frame.devices.items()
        if name in values
    )


def device_value_at_flux(
    frame: FluxFrame,
    values: Mapping[str, float],
    *,
    device_name: str,
    flux: float,
) -> float:
    """Project a finite target flux (Phi0) onto one device's native value.

    Other devices stay at values, with absent devices contributing zero. The
    selected device's current value and extra devices are ignored. No units
    are converted and no inputs change. ValueError names an invalid/unknown
    device, an invalid used value, or a nonfinite/non-real/boolean target flux.
    """
    validate_device_name(device_name)
    if device_name not in frame.devices:
        raise ValueError(f"unknown frame device {device_name!r}")
    target = _finite_value(flux, field="flux")
    other_flux = flux_from_device_values(frame, {**values, device_name: 0.0})
    return frame.devices[device_name].period * (target - other_flux)


def offset_for_flux(
    devices: Mapping[str, DevicePeriod],
    values: Mapping[str, float],
    *,
    flux: float,
) -> float:
    """Infer the unwrapped zero-output offset giving target flux at values.

    devices declares signed native periods; values uses those native units.
    Missing devices contribute zero and extra devices are ignored. Return
    target flux minus all contributions without changing inputs. Invalid
    periods/names/used values or a nonfinite/non-real/boolean target raise
    ValueError, with used-value errors naming the device.
    """
    return _finite_value(flux, field="flux") - flux_from_device_values(
        FluxFrame(0.0, devices), values
    )


def require_matching_device_unit(
    device_name: str, actual_unit: str | None, expected_unit: str | None
) -> None:
    """Require exact unit strings for a named device; return None on a match.

    device_name is a valid instrument name. actual_unit is the unit of a value
    the caller will use; expected_unit is its declared frame/context unit.
    Either unit may be None for an unrecorded unit, interpreted as "native".
    Otherwise units must be nonblank strings. No conversion, trimming or case
    folding is performed: "native" only matches "native", and "mA" differs
    from "A". Invalid names/units raise ValueError; mismatches name the device
    and both units. Callers select used values before calling this function.
    """
    validate_device_name(device_name)
    actual = "native" if actual_unit is None else actual_unit
    expected = "native" if expected_unit is None else expected_unit
    _require_unit(actual)
    _require_unit(expected)
    if actual != expected:
        raise ValueError(
            f"Device {device_name!r} unit {actual!r} does not match "
            f"expected unit {expected!r}"
        )


def _require_unit(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("unit must be a nonempty string")


def _require_device_period(value: object, name: str) -> None:
    if not isinstance(value, DevicePeriod):
        raise ValueError(f"device {name!r} must be a DevicePeriod")


def _finite_value(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number, got {value!r}")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite, got {value!r}") from exc
    if not isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result
