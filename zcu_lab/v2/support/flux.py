"""Qubit flux blocks and linear device coordinates, without GUI or device I/O."""

from collections.abc import Iterable, Mapping
from math import isfinite

from zcu_tools.analysis.flux import DevicePeriod, FluxFrame, device_value_at_flux
from zcu_tools.datafile import JsonObject

from zcu_lab.components.qubit import QubitFluxCalibration, QubitFluxSchema


def qubit_flux_frame(flux: QubitFluxSchema) -> JsonObject:
    """Copy the captured component offset/period block for a record.

    Return a detached JSON mapping, including a null uncalibrated offset and all
    signed periods. No files, device values or current parameters are read.
    """
    return flux.model_dump(mode="json")


def snapshot_numeric_values(
    snapshot: Mapping[str, JsonObject] | None, names: Iterable[str]
) -> dict[str, float]:
    """Read finite native biases from selected Run snapshot entries.

    Missing devices or missing value keys contribute nothing. An explicitly
    supplied nonnumeric, bool or nonfinite value raises ValueError naming its
    device. Extra snapshot entries are ignored. No hardware is read.
    """
    values: dict[str, float] = {}
    for name in names:
        info = None if snapshot is None else snapshot.get(name)
        if info is None or "value" not in info:
            continue
        value = info["value"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError(
                f"device {name!r} requires a finite non-bool Run snapshot value"
            )
        values[name] = value
    return values


def record_flux_frame(
    flux: QubitFluxSchema,
    snapshot: Mapping[str, JsonObject] | None,
) -> JsonObject:
    """Copy the Run-captured offset/period block with its original calibration.

    Missing snapshot values contribute zero and do not remove stored periods.
    Present invalid values raise ValueError naming the device. None keeps the
    caller's no-snapshot policy. No live values or current point are read.
    """
    snapshot_numeric_values(snapshot, flux.period)
    return qubit_flux_frame(flux)


def device_flux_points(
    flux: QubitFluxSchema,
    device_name: str,
    values: Mapping[str, float],
) -> QubitFluxCalibration:
    """Project flux 0 and 0.5 onto one device at other devices' captured biases.

    Missing other biases contribute zero. offset must be calibrated and the
    selected device must have a period. The math owner rejects invalid names
    and used values with device context. Return native half/integer coordinates
    and the stored signed period; no inputs change and no devices are read.
    """
    if flux.offset is None:
        raise ValueError("flux.offset is not calibrated")
    frame = FluxFrame(
        offset=flux.offset,
        devices={name: DevicePeriod(period) for name, period in flux.period.items()},
    )
    return QubitFluxCalibration(
        integer=device_value_at_flux(frame, values, device_name=device_name, flux=0.0),
        half=device_value_at_flux(frame, values, device_name=device_name, flux=0.5),
        period=flux.period[device_name],
    )
