"""Linear multi-device flux coordinates, independent of records and components."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

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


def solve_device_values(
    frames: Mapping[str, FluxFrame],
    targets: Mapping[str, float],
    *,
    device_names: Sequence[str],
    fixed_values: Mapping[str, float],
) -> dict[str, float]:
    """Jointly solve the linear flux equations for the requested devices.

    frames maps component names to FluxFrame calibrations. targets maps the
    same names to fluxes in Phi0. Each target is used as written, not folded
    modulo 1 or moved to a different branch.

    device_names lists one distinct device name per target component. A
    device absent from an individual frame has zero coupling to that component.
    fixed_values maps every other device appearing in a frame to a value in
    its native unit. Entries for solved devices or devices outside all frames
    do not contribute. A device must have exactly the same unit string in
    all frames that include it. All supplied values must be finite real
    numbers; booleans are not numbers here.

    Return only the solved device names and their native-unit values. Combine
    them with the fixed values to evaluate flux_from_device_values. No inputs
    change, and an empty system returns {}.

    Raise ValueError for mismatched component keys or counts, duplicate solve
    devices, missing fixed values, conflicting units, invalid numbers, or
    failed numerical solves. The system is rejected as singular or nearly
    singular if the 2-norm condition number of its native-unit coefficient
    matrix (signed reciprocal periods, with zeros for absent devices) is
    nonfinite or greater than 1e12.
    """
    if len(targets) != len(device_names):
        raise ValueError(
            f"expected one solve device per target component, got "
            f"{len(device_names)} devices and {len(targets)} targets"
        )
    if frames.keys() != targets.keys():
        missing_targets = sorted(frames.keys() - targets.keys())
        missing_frames = sorted(targets.keys() - frames.keys())
        raise ValueError(
            f"frames and targets must name the same components "
            f"(missing targets: {missing_targets}; missing frames: {missing_frames})"
        )
    solve_devices = set(device_names)
    if len(solve_devices) != len(device_names):
        raise ValueError(f"duplicate names in solve devices: {list(device_names)}")
    _require_joint_frame_devices(frames, solve_devices.union(fixed_values))
    fixed = {
        name: _finite_value(value, field=f"device {name!r} fixed value")
        for name, value in fixed_values.items()
    }
    for name in device_names:
        fixed.pop(name, None)
    if not device_names:
        return {}

    # ADR-0082: Each signed reciprocal period is a coupling in Phi0 per native unit.
    coefficients = np.array(
        [
            [
                1.0 / float(frame.devices[name].period)
                if name in frame.devices
                else 0.0
                for name in device_names
            ]
            for frame in frames.values()
        ],
        dtype=np.float64,
    )
    remaining_flux = np.array(
        [
            _finite_value(targets[name], field=f"target flux for {name!r}")
            - flux_from_device_values(frame, fixed)
            for name, frame in frames.items()
        ],
        dtype=np.float64,
    )
    values = _solve_flux_equations(coefficients, remaining_flux)
    return {
        name: _finite_value(value, field=f"device {name!r} solution")
        for name, value in zip(device_names, values, strict=True)
    }


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


def _require_joint_frame_devices(
    frames: Mapping[str, FluxFrame], covered_devices: set[str]
) -> None:
    """Require a value source and consistent native unit for each frame device."""
    units: dict[str, str] = {}
    for component, frame in frames.items():
        for name, device in frame.devices.items():
            if name not in covered_devices:
                raise ValueError(
                    f"component {component!r} device {name!r} needs a fixed value"
                )
            if name in units:
                require_matching_device_unit(name, device.unit, units[name])
            else:
                units[name] = device.unit


def _solve_flux_equations(
    coefficients: NDArray[np.float64], remaining_flux: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solve a finite, well-conditioned square system or raise ValueError."""
    if not np.isfinite(coefficients).all():
        raise ValueError("flux coefficients must be finite; check device periods")
    if not np.isfinite(remaining_flux).all():
        raise ValueError(
            "remaining flux after applying offsets and fixed values must be finite"
        )
    try:
        condition = float(np.linalg.cond(coefficients))
        if not isfinite(condition) or condition > 1e12:
            raise ValueError(
                f"flux equations are singular or nearly singular (condition {condition:g})"
            )
        return np.asarray(
            np.linalg.solve(coefficients, remaining_flux), dtype=np.float64
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"could not solve flux equations: {exc}") from exc


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
