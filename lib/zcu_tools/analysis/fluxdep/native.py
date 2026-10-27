"""Explicit projection of generic native containers to a device-major spectrum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.datafile import Container


@dataclass(frozen=True)
class NativeSpectrum:
    """Selected device-major complex signal grid with explicit axis units.

    dev_values uses unit unchanged from the selected device axis. freqs_mhz is
    the other axis converted to MHz. signals has shape (N_device, N_frequency).
    Returned arrays are detached from the input container.
    """

    dev_values: NDArray[np.float64]
    freqs_mhz: NDArray[np.float64]
    signals: NDArray[np.complex128]
    unit: str


def project_native_spectrum(
    container: Container, *, variable: str, device_axis: str
) -> NativeSpectrum:
    """Project one explicitly named 2-D complex variable and its device axis.

    The remaining axis must have unit exactly Hz/kHz/MHz/GHz; convert it to MHz.
    Keep real device values and unit unchanged, arranging signals device-major.
    Both axes must have real numeric dtype, be 1-D and match the selected data
    dimensions. Complex axes, including zero-imaginary values, are rejected.
    Invalid variable, dimensionality, dtype, axis or frequency unit raises
    ValueError listing every variable and its axis names/units. Never infer
    selection from cfg, experiment definitions or the number of candidates.
    """
    candidates = (
        "; ".join(
            f"{name}: " + ", ".join(f"{axis.name} [{axis.unit}]" for axis in item.axes)
            for name, item in container.vars.items()
        )
        or "<no variables>"
    )
    diagnostic = f"; available variables/axes: {candidates}"
    if variable not in container.vars:
        raise ValueError(f"Unknown variable {variable!r}{diagnostic}")
    selected = container.vars[variable]
    if selected.values.ndim != 2 or not np.iscomplexobj(selected.values):
        raise ValueError(f"Variable {variable!r} must be 2-D complex{diagnostic}")
    if len(selected.axes) != 2 or any(
        axis.values.ndim != 1
        or axis.values.shape[0] != selected.values.shape[index]
        or axis.values.dtype.kind not in "iuf"
        for index, axis in enumerate(selected.axes)
    ):
        raise ValueError(
            f"Variable {variable!r} requires two matching real 1-D axes{diagnostic}"
        )
    names = [axis.name for axis in selected.axes]
    if device_axis not in names:
        raise ValueError(f"Unknown device axis {device_axis!r}{diagnostic}")
    device_index = names.index(device_axis)
    frequency_index = 1 - device_index
    frequency = selected.axes[frequency_index]
    scales = {"Hz": 1e-6, "kHz": 1e-3, "MHz": 1.0, "GHz": 1e3}
    if frequency.unit not in scales:
        raise ValueError(f"Invalid frequency unit {frequency.unit!r}{diagnostic}")
    device = selected.axes[device_index]
    return NativeSpectrum(
        dev_values=np.array(device.values, dtype=np.float64, copy=True),
        freqs_mhz=np.array(frequency.values, dtype=np.float64, copy=True)
        * scales[frequency.unit],
        signals=np.array(
            selected.values if device_index == 0 else selected.values.T,
            dtype=np.complex128,
            copy=True,
        ),
        unit=device.unit,
    )
