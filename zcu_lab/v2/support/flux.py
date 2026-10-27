"""Qubit flux blocks and linear device coordinates, without GUI or device I/O."""

from collections.abc import Collection, Iterable, Mapping
from math import copysign, isfinite

from zcu_tools.analysis.flux import (
    DeviceFluxFrame,
    DeviceFluxPoint,
    FluxFrame,
    actual_flux_points,
    reference_flux_points,
)
from zcu_tools.cfg_model import validate_device_name
from zcu_tools.datafile import JsonObject

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


def snapshot_numeric_values(
    snapshot: Mapping[str, JsonObject] | None, names: Iterable[str]
) -> dict[str, float]:
    """Project selected Run snapshot value fields without reading devices.

    names selects the required device settings; extra settings are ignored.
    Return a new mapping of numeric, non-bool JSON values. Missing/non-numeric
    values are omitted; nonfinite numbers remain for the using math/record
    contract to reject with device context. None acts as an absent snapshot.
    Inputs stay unchanged. This mechanical projection does not validate frame
    membership, choose contributors or decide whether records may be written.
    """
    values: dict[str, float] = {}
    for name in names:
        info = None if snapshot is None else snapshot.get(name)
        value = None if info is None else info.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[name] = value
    return values


def record_flux_frame(
    flux: Mapping[str, QubitFluxCalibration],
    snapshot: Mapping[str, JsonObject] | None,
) -> dict[str, dict[str, float]]:
    """Copy Run-captured calibrations paired with that Run's device values.

    flux is the qubit block captured before measurement; snapshot is the same
    Run's complete device settings, in native units. Return a detached frame
    with signed periods. For a present snapshot, every frame device must have
    a finite, non-bool numeric value or ValueError names it at record proposal
    time. Empty flux is legal. None skips membership validation so callers keep
    their existing no-snapshot policy (omit ordinary GUI records, retain workflow
    null). Fluxdep callers still require a snapshot before conversion.
    Calibration/name failures propagate from qubit_flux_frame. No inputs are
    changed; no setup catalog, live devices, current point or history is read.
    """
    frame = qubit_flux_frame(flux)
    if snapshot is not None:
        values = snapshot_numeric_values(snapshot, frame)
        for name in frame:
            if name not in values or not isfinite(values[name]):
                raise ValueError(
                    f"device {name!r} requires a finite non-bool Run snapshot value"
                )
    return frame


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
