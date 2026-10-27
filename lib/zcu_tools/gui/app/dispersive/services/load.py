"""Load one-tone data using the selected predictor's recorded flux frame.

Native arrays share analysis.fluxdep's explicit projection. Snapshot metadata
comes from the same Container and scalar values from the device owner. Labber
and historical native inputs without snapshots require exactly one predictor
device, retaining unnamed_device as source identity. Frequencies use GHz.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np

from zcu_tools.analysis.flux import FluxFrame, flux_from_device_values
from zcu_tools.analysis.fluxdep import project_native_spectrum
from zcu_tools.analysis.spectrum import SpectrumData, format_rawdata
from zcu_tools.datafile import Container
from zcu_tools.device import snapshot_scalar_values
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.gui.app.analysis_project.native import (
    NativeSpectrumSelection,
    read_spectrum_source,
)
from zcu_tools.gui.app.dispersive.state import DispersiveState, OnetoneEntry


def transpose_spectrum_data(
    signals2d: np.ndarray, dev_values: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Swap Labber device/frequency axes, returning signals.T, freqs, dev_values."""
    return signals2d.T, freqs, dev_values


def _other_device_values(
    frame: FluxFrame, snapshot: Mapping[str, Mapping[str, object]], device: str
) -> dict[str, float]:
    if device not in frame.devices:
        raise ValueError(f"Source device {device!r} is not in predictor flux frame")
    scalars = snapshot_scalar_values(
        snapshot, [name for name in frame.devices if name != device]
    )
    values: dict[str, float] = {}
    for name, scalar in scalars.items():
        if scalar.unit != frame.devices[name].unit:
            raise ValueError(
                f"Device {name!r} snapshot unit {scalar.unit!r} does not match "
                f"predictor unit {frame.devices[name].unit!r}"
            )
        values[name] = scalar.value
    return values


class LoadService:
    """Read one spectrum and publish only after successful projection."""

    def __init__(self, state: DispersiveState) -> None:
        self._state = state

    def load_onetone(
        self,
        filepath: str,
        *,
        transpose_axes: bool = False,
        selection: NativeSpectrumSelection | None = None,
    ) -> str:
        """Read exact path, project device values to Phi0, and return its basename.

        Requires loaded predictor inputs. Native selection explicitly names the
        variable/device axis/instrument; transpose applies only to Labber. Native
        metadata is decoded from the same Container. Only other predictor devices
        use snapshot scalars; missing names contribute zero. No snapshot requires
        exactly one predictor device and retains unnamed_device source identity.
        Invalid selection, metadata or used scalars raise ValueError without
        changing state. I/O errors propagate. No component or live device is read.
        """
        inputs = self._state.fit_inputs
        if inputs is None:
            raise RuntimeError(
                "load the fluxonium fit inputs before a one-tone spectrum"
            )
        frame = inputs.flux.to_frame()
        source = read_spectrum_source(filepath)
        snapshot = None
        unit = "native"
        device = "unnamed_device"
        if isinstance(source, Container):
            if selection is None:
                raise ValueError(
                    "Native input requires variable, device_axis and device"
                )
            projected = project_native_spectrum(
                source, variable=selection.variable, device_axis=selection.device_axis
            )
            dev_values = projected.dev_values
            freqs = projected.freqs_mhz * 1e6
            signals2d = projected.signals
            unit = projected.unit
            device = selection.device
            if device not in frame.devices and device != "unnamed_device":
                raise ValueError(
                    f"Source device {device!r} is not in predictor flux frame"
                )
            if "metadata" in source.docs:
                snapshot = metadata_from_container(source).device_snapshot
        else:
            if len(source.axes) < 2:
                raise ValueError(
                    f"{filepath!r} has no frequency axis (not a 2D spectrum)"
                )
            dev_values = np.asarray(source.axes[0].values)
            freqs = np.asarray(source.axes[1].values)
            # Labber stores (N_frequency, N_device), unlike the analysis grid.
            signals2d = np.asarray(source.z).T
            if transpose_axes:
                signals2d, dev_values, freqs = transpose_spectrum_data(
                    signals2d, dev_values, freqs
                )

        values: dict[str, float] = {}
        if snapshot is None:
            if len(frame.devices) != 1:
                raise ValueError(
                    "No device snapshot: predictor must have exactly one device; "
                    f"found {list(frame.devices)}"
                )
            scanned_device = next(iter(frame.devices))
            device = "unnamed_device"
        else:
            scanned_device = device
            values = _other_device_values(frame, snapshot, scanned_device)

        if isinstance(source, Container) and unit != frame.devices[scanned_device].unit:
            raise ValueError(
                f"Device {scanned_device!r} axis unit {unit!r} does not match "
                f"predictor unit {frame.devices[scanned_device].unit!r}"
            )
        dev_values, freqs, signals2d = format_rawdata(dev_values, freqs, signals2d)
        fluxs = np.array(
            [
                flux_from_device_values(frame, {**values, scanned_device: float(value)})
                for value in dev_values
            ]
        )
        raw = SpectrumData(
            dev_values=dev_values.astype(np.float64),
            fluxs=fluxs,
            freqs=freqs.astype(np.float64),
            signals=signals2d.astype(np.complex128),
        )
        name = os.path.basename(filepath)
        self._state.set_onetone(
            OnetoneEntry(name=name, raw=raw, device=device, unit=unit)
        )
        return name
