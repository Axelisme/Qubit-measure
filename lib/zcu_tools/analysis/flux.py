"""Linear multi-device flux coordinates, independent of records and components."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real
from types import MappingProxyType

from zcu_tools.cfg_model import validate_device_name


@dataclass(frozen=True)
class DeviceFluxFrame:
    """One device's reference calibration in its native instrument unit.

    integer is the finite device value at integer flux with all other devices
    at their integer references. period is the finite nonzero signed value
    span per Phi0: negative means increasing value decreases flux. unit names
    the value unit (for example A, V or native); no conversion is performed.
    Invalid fields raise ValueError at construction; booleans are not numbers.
    """

    integer: float
    period: float
    unit: str = "native"

    def __post_init__(self) -> None:
        _finite_value(self.integer, field="integer")
        if _finite_value(self.period, field="period") == 0:
            raise ValueError("period must be nonzero (signed device units per Phi0)")
        _require_nonempty_name(self.unit, field="unit")


@dataclass(frozen=True)
class FluxFrame:
    """Nonempty mapping of device names to their independent linear frames.

    devices keys are nonblank instrument names without ASCII dots; values are
    DeviceFluxFrame. Valid names are preserved, including surrounding spaces.
    Each device can use its own native unit and signed period. Construction
    snapshots the mapping. Invalid keys/values or an empty frame raise
    ValueError. All contributing devices must be supplied when evaluating flux.
    """

    devices: Mapping[str, DeviceFluxFrame]

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("devices must be nonempty")
        for name, device in self.devices.items():
            validate_device_name(name)
            _require_device_frame(device, name)
        object.__setattr__(self, "devices", MappingProxyType(dict(self.devices)))


def flux_from_device_values(frame: FluxFrame, values: Mapping[str, float]) -> float:
    """Return sum((value - integer) / period) in dimensionless Phi0.

    values maps every frame device to a finite real instrument value in that
    device's declared unit. Extra devices are ignored. No unit conversion or
    component parsing occurs. Missing devices, booleans and nonfinite/non-real
    used values raise ValueError naming the device. Inputs are not modified.
    """
    return sum(
        (_device_value(values, name) - device.integer) / device.period
        for name, device in frame.devices.items()
    )


@dataclass(frozen=True)
class DeviceFluxPoint:
    """An integer/half pair for one device in that device's native unit.

    integer and half are finite real instrument values (not booleans). They
    can be measured at current other-device values, or expressed at the
    frame's integer references; the converting function declares which.
    Either half branch is allowed. Construction rejects invalid numbers with
    ValueError; conversion checks that their separation is abs(period)/2,
    with relative tolerance 1e-9 and no absolute tolerance.
    """

    integer: float
    half: float

    def __post_init__(self) -> None:
        _finite_value(self.integer, field="integer")
        _finite_value(self.half, field="half")


def reference_flux_points(
    fitted: DeviceFluxPoint,
    *,
    device_name: str,
    frame: FluxFrame,
    values: Mapping[str, float],
) -> DeviceFluxPoint:
    """Translate fitted integer/half back to other devices' integer references.

    fitted uses the selected device's unit and was fitted at the supplied
    other-device values. frame must use the newly fitted signed period for
    the selected device, and the reference calibrations for all others.
    Add selected signed period times the other devices'
    flux contribution to both points. values requires every OTHER frame
    device, in its declared unit; the selected device value is not used.
    device_name must be nonblank without ASCII dots; valid names are unchanged.
    Inputs stay unchanged. ValueError names an invalid/unknown selected device,
    missing/invalid used values, or a pair not separated by abs(period)/2.
    """
    device = _selected_frame(frame, device_name)
    validate_flux_spacing(fitted, device, device_name)
    shift = device.period * flux_from_device_values(
        frame, {**values, device_name: device.integer}
    )
    return DeviceFluxPoint(fitted.integer + shift, fitted.half + shift)


def actual_flux_points(
    reference: DeviceFluxPoint,
    *,
    device_name: str,
    frame: FluxFrame,
    values: Mapping[str, float],
) -> DeviceFluxPoint:
    """Translate reference integer/half to the other devices' current values.

    reference is the selected device's pair with all other frame devices at
    their integer references, in its declared unit. Subtract selected signed
    period times the other devices' flux contribution from both points.
    values requires every OTHER frame device in its declared unit; selected
    device values and extra devices are ignored. Inputs are unchanged.
    device_name must be nonblank without ASCII dots; valid names are unchanged.
    ValueError names invalid/unknown device, missing/invalid used values, or a pair
    not separated by abs(period)/2.
    """
    device = _selected_frame(frame, device_name)
    validate_flux_spacing(reference, device, device_name)
    shift = device.period * flux_from_device_values(
        frame, {**values, device_name: device.integer}
    )
    return DeviceFluxPoint(reference.integer - shift, reference.half - shift)


def validate_flux_spacing(
    points: DeviceFluxPoint, device: DeviceFluxFrame, name: str
) -> None:
    """Check a native-unit point pair against a device's signed period.

    points and device must use the same instrument unit. name labels the device
    in errors; callers validate its syntax and membership separately. Return
    None without changing inputs when abs(integer-half) equals abs(period)/2,
    allowing either half branch and period sign. Relative tolerance is 1e-9;
    absolute tolerance is zero, including tiny periods. Otherwise raise
    ValueError naming the device. No component or record data is read.
    """
    if not isclose(
        abs(points.integer - points.half),
        abs(device.period) / 2,
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"device {name!r} half/integer separation must be abs(period)/2"
        )


def _selected_frame(frame: FluxFrame, name: str) -> DeviceFluxFrame:
    validate_device_name(name)
    if name not in frame.devices:
        raise ValueError(f"unknown frame device {name!r}")
    return frame.devices[name]


def _device_value(values: Mapping[str, float], name: str) -> float:
    if name not in values:
        raise ValueError(f"missing device value for {name!r}")
    return _finite_value(values[name], field=f"device {name!r} value")


def _require_nonempty_name(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _require_device_frame(value: object, name: str) -> None:
    if not isinstance(value, DeviceFluxFrame):
        raise ValueError(f"device {name!r} must be a DeviceFluxFrame")


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
