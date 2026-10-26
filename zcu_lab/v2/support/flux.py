"""Qubit flux blocks and linear device coordinates, without GUI or device I/O."""

from collections.abc import Collection, Mapping
from math import copysign

from zcu_tools.analysis.flux import (
    DeviceFluxFrame,
    DeviceFluxPoint,
    FluxFrame,
    actual_flux_points,
    reference_flux_points,
)
from zcu_tools.cfg_model import validate_device_name

from zcu_lab.components.qubit import QubitFluxCalibration


def qubit_flux_frame(
    flux: Mapping[str, QubitFluxCalibration],
) -> dict[str, dict[str, float]]:
    """Copy a qubit's flux block into a detached record-ready mapping.

    flux contains device-keyed component calibrations in native device units.
    Return half/integer/period for every entry, preserving signed periods and
    valid names unchanged. An empty block returns {}. Invalid device names or
    abs(integer-half) != abs(period)/2 raise ValueError naming the device.
    This pure copy does not read point files, check a setup catalog or obtain
    device values; the caller chooses which point snapshot to copy and when.
    """
    result: dict[str, dict[str, float]] = {}
    for name, calibration in flux.items():
        validate_device_name(name)
        calibration.validate_spacing(name)
        result[name] = {
            "half": calibration.half,
            "integer": calibration.integer,
            "period": calibration.period,
        }
    return result


def device_flux_points(
    flux: Mapping[str, QubitFluxCalibration],
    device_name: str,
    values: Mapping[str, float],
) -> DeviceFluxPoint:
    """Return a device's actual integer/half at other devices' current biases.

    flux is the qubit's reference block; values supplies each OTHER contributing
    device's finite bias in its native unit. The selected value and extra values
    are ignored. No units are converted, inputs changed or hardware read.
    Invalid/unknown device names, inconsistent calibration spacing and missing
    or invalid contributing values raise ValueError naming the device. Empty
    flux is not a usable calibration and raises ValueError.
    """
    frame = _math_frame(flux)
    validate_device_name(device_name)
    if device_name not in flux:
        raise ValueError(f"unknown frame device {device_name!r}")
    calibration = flux[device_name]
    return actual_flux_points(
        DeviceFluxPoint(integer=calibration.integer, half=calibration.half),
        device_name=device_name,
        frame=frame,
        values=values,
    )


def reference_device_flux_points(
    flux: Mapping[str, QubitFluxCalibration],
    device_name: str,
    fitted: QubitFluxCalibration,
    values: Mapping[str, float],
    *,
    swept_devices: Collection[str] = (),
) -> QubitFluxCalibration:
    """Translate raw fitted points back to fixed contributors' integer references.

    flux is the qubit's reference block. fitted contains the raw half/integer
    and fitted period magnitude in the scanned device's native unit. Use that
    NEW magnitude with the stored direction, or positive direction for a first
    calibration. values supplies OTHER contributors' finite native biases from
    the Run snapshot. Return only the selected device's reference calibration;
    the caller writes it without replacing other entries. Inputs stay unchanged.

    swept_devices names all devices swept in this Run. A different swept device
    in this qubit's frame raises ValueError naming it: its snapshot cannot stand
    in for a fixed bias. Invalid device names, inconsistent spacing,
    and missing/invalid used values also raise ValueError naming the device.
    Empty stored flux supports first calibration; no device I/O occurs.
    """
    validate_device_name(device_name)
    qubit_flux_frame(flux)
    for name in swept_devices:
        validate_device_name(name)
        if name != device_name and name in flux:
            raise ValueError(
                f"device {name!r} contributes to this qubit and is swept in the same "
                "Run; reference conversion requires fixed other devices"
            )
    previous = flux.get(device_name)
    period = copysign(abs(fitted.period), 1 if previous is None else previous.period)
    selected = QubitFluxCalibration(
        half=fitted.half, integer=fitted.integer, period=period
    )
    # Replace only the selected math frame: the old magnitude must not set the shift.
    frame = _math_frame({**flux, device_name: selected})
    reference = reference_flux_points(
        DeviceFluxPoint(integer=fitted.integer, half=fitted.half),
        device_name=device_name,
        frame=frame,
        values=values,
    )
    return QubitFluxCalibration(
        half=reference.half, integer=reference.integer, period=period
    )


def _math_frame(flux: Mapping[str, QubitFluxCalibration]) -> FluxFrame:
    return FluxFrame(
        {
            name: DeviceFluxFrame(
                integer=calibration["integer"], period=calibration["period"]
            )
            for name, calibration in qubit_flux_frame(flux).items()
        }
    )
