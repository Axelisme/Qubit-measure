"""Versioned HDF5 persistence for processed flux-dependence spectra."""

from __future__ import annotations

import math
import os
from numbers import Real

import h5py as h5

from zcu_tools.analysis.flux import DevicePeriod, FluxFrame
from zcu_tools.analysis.fluxdep.models import DeviceSnapshotValue, SpectrumResult


def _number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Expected finite scalar number at {location}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected finite scalar number at {location}")
    return result


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty string at {location}")
    return value


def _group(parent: h5.Group, name: str) -> h5.Group:
    group = parent[name]
    if not isinstance(group, h5.Group):
        raise TypeError(f"Expected Group at {parent.name}/{name}")
    return group


def _dataset(parent: h5.Group, name: str) -> h5.Dataset:
    dataset = parent[name]
    if not isinstance(dataset, h5.Dataset):
        raise TypeError(f"Expected Dataset at {parent.name}/{name}")
    return dataset


def _validate_calibration(result: SpectrumResult) -> None:
    FluxFrame(
        offset=result["offset"],
        devices={result["device"]: DevicePeriod(result["period"], result["unit"])},
    )
    for name, snapshot in result.get("device_snapshot", {}).items():
        _number(snapshot["value"], f"device_snapshot/{name}")
        _text(snapshot["unit"], f"device_snapshot/{name}/unit")
        if not name or "/" in name:
            raise ValueError(f"Invalid snapshot device name: {name!r}")


def dump_spectrums(
    path: str, spectrums: dict[str, SpectrumResult], mode: str = "x"
) -> None:
    """Write v2 spectra, preserving array values and dtypes.

    Each mapping key becomes one HDF5 group. Calibration uses group attributes;
    optional snapshots use scalar datasets with unit attributes. Invalid finite
    calibration or snapshot values raise ValueError before opening the file.
    mode follows h5py (default x refuses existing paths); parent directories are
    created. For a/r+, existing files must have the v2 format/version header;
    otherwise ValueError names the upgrade command and leaves bytes unchanged.
    Mode w explicitly overwrites existing files. Other I/O failures propagate and
    may leave a partially written file.
    """
    for spectrum in spectrums.values():
        _validate_calibration(spectrum)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if mode in ("a", "r+") and os.path.exists(path):
        with h5.File(path, "r") as existing:
            if (
                existing.attrs.get("format") != "zcu.fluxdep-spectrums"
                or existing.attrs.get("version") != 2
            ):
                raise ValueError(
                    f"{path}: expected zcu.fluxdep-spectrums version 2; "
                    f"upgrade with: python scripts/upgrade_spectrums.py {path!r} --apply"
                )
    with h5.File(path, mode) as handle:
        handle.attrs["format"] = "zcu.fluxdep-spectrums"
        handle.attrs["version"] = 2
        for name, spectrum in spectrums.items():
            group = handle.create_group(name)
            for field in ("offset", "period", "device", "unit"):
                group.attrs[field] = spectrum[field]
            if "type" in spectrum:
                group.attrs["type"] = spectrum["type"]
            if "device_snapshot" in spectrum:
                snapshots = group.create_group("device_snapshot")
                for device, snapshot in spectrum["device_snapshot"].items():
                    dataset = snapshots.create_dataset(device, data=snapshot["value"])
                    dataset.attrs["unit"] = snapshot["unit"]
            spect = group.create_group("spectrum")
            for field, value in spectrum["spectrum"].items():
                spect.create_dataset(field, data=value)
            points = group.create_group("points")
            for field, value in spectrum["points"].items():
                points.create_dataset(field, data=value)


def load_spectrums(path: str) -> dict[str, SpectrumResult]:
    """Read only v2 spectra without changing stored arrays or dtypes.

    Unsupported headers raise ValueError with an explicit upgrade command.
    Missing fields raise KeyError. Wrong group/dataset kinds raise TypeError
    naming the HDF5 path; invalid calibration/snapshot scalars raise ValueError.
    Optional type and device_snapshot fields are returned only when present.
    """
    results: dict[str, SpectrumResult] = {}
    with h5.File(path, "r") as handle:
        if (
            handle.attrs.get("format") != "zcu.fluxdep-spectrums"
            or handle.attrs.get("version") != 2
        ):
            raise ValueError(
                f"{path}: expected zcu.fluxdep-spectrums version 2; "
                f"upgrade with: python scripts/upgrade_spectrums.py {path!r} --apply"
            )
        for name in list(handle.keys()):
            group = _group(handle, name)
            spect = _group(group, "spectrum")
            points = _group(group, "points")
            result = SpectrumResult(
                offset=_number(group.attrs["offset"], f"{group.name}/offset"),
                period=_number(group.attrs["period"], f"{group.name}/period"),
                device=_text(group.attrs["device"], f"{group.name}/device"),
                unit=_text(group.attrs["unit"], f"{group.name}/unit"),
                spectrum={
                    "dev_values": _dataset(spect, "dev_values")[()],
                    "fluxs": _dataset(spect, "fluxs")[()],
                    "freqs": _dataset(spect, "freqs")[()],
                    "signals": _dataset(spect, "signals")[()],
                },
                points={
                    "dev_values": _dataset(points, "dev_values")[()],
                    "fluxs": _dataset(points, "fluxs")[()],
                    "freqs": _dataset(points, "freqs")[()],
                },
            )
            if "type" in group.attrs:
                result["type"] = _text(group.attrs["type"], f"{group.name}/type")
            if "device_snapshot" in group:
                snapshots = _group(group, "device_snapshot")
                values: dict[str, DeviceSnapshotValue] = {}
                for device in list(snapshots.keys()):
                    dataset = _dataset(snapshots, device)
                    values[device] = DeviceSnapshotValue(
                        value=_number(dataset[()], f"{snapshots.name}/{device}"),
                        unit=_text(dataset.attrs["unit"], f"{dataset.name}/unit"),
                    )
                result["device_snapshot"] = values
            _validate_calibration(result)
            results[name] = result
    return results
