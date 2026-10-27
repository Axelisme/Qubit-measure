"""Explicit, read-only-by-default upgrade of legacy processed spectra."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import h5py
from pydantic import TypeAdapter
from zcu_tools.analysis.fluxdep.io import load_spectrums


def _legacy_calibrations(handle: h5py.File) -> dict[str, tuple[float, float]]:
    if "format" in handle.attrs or "version" in handle.attrs:
        raise ValueError("unsupported spectrum format/version")
    calibrations: dict[str, tuple[float, float]] = {}
    for name in list(handle.keys()):
        group = handle[name]
        if not isinstance(group, h5py.Group):
            raise TypeError(f"Expected spectrum group at /{name}")
        values: dict[str, float] = {}
        for field in ("flux_half", "flux_int", "flux_period"):
            dataset = group[field]
            if not isinstance(dataset, h5py.Dataset) or dataset.shape != ():
                raise ValueError(f"Expected scalar dataset at /{name}/{field}")
            value = TypeAdapter(float).validate_python(dataset[()], strict=True)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite calibration at /{name}/{field}")
            values[field] = value
        period = values["flux_period"]
        if period == 0:
            raise ValueError(f"Zero period at /{name}/flux_period")
        for block, fields in (
            ("spectrum", ("dev_values", "fluxs", "freqs", "signals")),
            ("points", ("dev_values", "fluxs", "freqs")),
        ):
            for field in fields:
                if not isinstance(group[f"{block}/{field}"], h5py.Dataset):
                    raise TypeError(f"Expected dataset at /{name}/{block}/{field}")
        calibrations[name] = (0.5 - values["flux_half"] / period, period)
    return calibrations


def _upgrade(path: Path, *, apply: bool) -> None:
    with h5py.File(path, "r") as handle:
        if (
            handle.attrs.get("format") == "zcu.fluxdep-spectrums"
            and handle.attrs.get("version") == 2
        ):
            load_spectrums(str(path))
            print(f"{path}: skipped (already v2)")
            return
        calibrations = _legacy_calibrations(handle)
    for name, (offset, period) in calibrations.items():
        print(f"  /{name}/offset: <missing> -> {offset}")
        print(f"  /{name}/period: flux_period -> {period}")
        print(f"  /{name}/device: <missing> -> unnamed_device")
        print(f"  /{name}/unit: <missing> -> native")
        print(f"  /{name}/flux_half, flux_int, flux_period: removed")
    if not apply:
        print(f"{path}: dry-run")
        return
    # Work on a sibling copy so failed validation never replaces the source.
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    candidate = Path(temporary)
    try:
        shutil.copy2(path, candidate)
        with h5py.File(candidate, "r+") as handle:
            handle.attrs.update(format="zcu.fluxdep-spectrums", version=2)
            for name, (offset, period) in calibrations.items():
                group = handle[name]
                if not isinstance(group, h5py.Group):
                    raise TypeError(f"Expected spectrum group at /{name}")
                group.attrs.update(
                    offset=offset, period=period, device="unnamed_device", unit="native"
                )
                for field in ("flux_half", "flux_int", "flux_period"):
                    del group[field]
        load_spectrums(str(candidate))
        backup = path.with_name(f"{path.name}.{uuid4().hex}.bak")
        with path.open("rb") as source, backup.open("xb") as target:
            shutil.copyfileobj(source, target)
        print(f"{path}: backup: {backup}")
        os.replace(candidate, path)
    finally:
        candidate.unlink(missing_ok=True)
    print(f"{path}: applied")


def main(argv: Sequence[str] | None = None) -> int:
    """Preview exact legacy HDF5 paths; --apply backs up and replaces with v2.

    None reads process arguments. Print each changed HDF5 location and status.
    Dry-run creates no files. Apply validates a sibling candidate, then saves
    an exclusive original-bytes .bak before atomic replacement. Existing v2 files
    are validated and skipped. Rejected files remain unchanged; processing
    continues with other paths and returns 1 on any rejection, otherwise 0.
    Invalid arguments raise argparse SystemExit(2). No device identity is guessed:
    legacy spectra receive unnamed_device/native and no snapshot.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true")
    options = parser.parse_args(argv)
    paths = TypeAdapter(list[Path]).validate_python(options.paths)
    apply = TypeAdapter(bool).validate_python(options.apply)
    failed = False
    for path in paths:
        try:
            _upgrade(path, apply=apply)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"{path}: rejected\n  failed checks: {error}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
