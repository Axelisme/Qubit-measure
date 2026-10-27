"""Explicit spectrum source reads and caller-selected native coordinates."""

from dataclasses import dataclass
from pathlib import Path

import h5py

from zcu_tools.datafile import Container, LabberData, load, load_labber_data


@dataclass(frozen=True)
class NativeSpectrumSelection:
    """Explicit Container variable, its device-axis name and instrument identity.

    variable names a 2-D complex variable. device_axis names its native bias
    axis. device is the caller-selected instrument, or the explicit
    unnamed_device isolation choice. No cfg-derived identity is substituted.
    """

    variable: str
    device_axis: str
    device: str = "unnamed_device"


def read_spectrum_source(filepath: str) -> Container | LabberData:
    """Read an exact native Container or Labber log, using its declared header.

    Native content and metadata stay together in one detached Container. A
    declared container never falls back to a Labber parser after validation
    failure. Unknown headers use the Labber reader; I/O errors propagate.
    """
    with h5py.File(filepath, "r") as handle:
        native = handle.attrs.get("format") == "zcu.container"
    return load(Path(filepath)) if native else load_labber_data(filepath)
