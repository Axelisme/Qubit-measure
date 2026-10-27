"""LoadService — read a raw spectrum hdf5 into a SpectrumEntry.

Native Container arrays use the shared explicit spectrum projection; metadata
uses experiment.native_persistence and scalar values use the device owner.
Labber files retain their low-level reader and unnamed_device/native identity.
Both paths use format_rawdata for GHz and monotonic ordering. No experiment.v2
or lab definitions are imported. OneTone/TwoTone only select downstream tools.

This module is the synchronous core. Wrapping it in a worker thread + OperationGate
(so a slow load degrades to an operation handle, like measure's run/device setup)
is layered on top later; keeping the core sync makes it directly testable.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from zcu_tools.analysis.flux import (
    ComponentFluxContext,
    DevicePeriod,
    FluxFrame,
    device_value_at_flux,
)
from zcu_tools.analysis.fluxdep import project_native_spectrum
from zcu_tools.analysis.fluxdep.io import load_spectrums
from zcu_tools.analysis.fluxdep.models import DeviceSnapshotValue, PointsData
from zcu_tools.analysis.spectrum import SpectrumData, format_rawdata
from zcu_tools.datafile import Container, LabberData
from zcu_tools.device import snapshot_scalar_values
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.gui.app.analysis_project.flux import (
    ComponentFluxProvider,
    component_flux_context,
)
from zcu_tools.gui.app.analysis_project.native import (
    NativeSpectrumSelection,
    read_spectrum_source,
)
from zcu_tools.gui.app.fluxdep.state import (
    FluxDepState,
    SpectrumEntry,
    SpecType,
)
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.simulate import value2flux

logger = logging.getLogger(__name__)


def transpose_spectrum_data(
    signals2d: np.ndarray, dev_values: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Swap the device-value and frequency axes of a raw spectrum.

    Some legacy files store the axes transposed (x=frequency, y=flux,
    z=(freq, flux)) versus what the loader expects (x=flux, y=freq,
    z=(flux, freq)). This swaps ``dev_values`` ↔ ``freqs`` and transposes the
    signal grid accordingly, returning ``(signals.T, freqs, dev_values)``.
    """
    return signals2d.T, freqs, dev_values


def _empty_points() -> PointsData:
    """A points block with no selected points yet (filled by PointsService)."""
    empty = np.empty(0, dtype=np.float64)
    return PointsData(dev_values=empty.copy(), fluxs=empty.copy(), freqs=empty.copy())


class LoadService:
    """Loads raw spectrum hdf5 files into the State's spectrum collection."""

    def __init__(
        self,
        state: FluxDepState,
        *,
        flux_context_provider: ComponentFluxProvider | None = None,
    ) -> None:
        self._state = state
        self._flux_context_provider = flux_context_provider

    def flux_context(self) -> ComponentFluxContext | None:
        """Return the target's injected context, or None without an opened target.

        Missing/rejecting providers raise ValueError naming component and kind.
        """
        if self._state.project is None:
            return None
        return component_flux_context(self._state.project, self._flux_context_provider)

    def load_spectrum(
        self,
        filepath: str,
        spec_type: SpecType,
        inherit_from: str | None = None,
        *,
        transpose_axes: bool = False,
        selection: NativeSpectrumSelection | None = None,
    ) -> str:
        """Load ``filepath`` as a new spectrum and write it into State.

        The spectrum name is the file's basename. ``inherit_from`` (an existing
        spectrum name with committed or seeded alignment) seeds this spectrum's
        flux alignment as an initial guess;
        omitted, the alignment starts at the identity (flux_half=0, period=1) and
        is refined later by the line-picker. The loaded ``fluxs`` are derived from
        whatever alignment is in effect at load time (re-derived on alignment).

        selection is required for native Container inputs and explicitly names
        variable, device axis and instrument. Native axis units are retained;
        metadata comes from the same Container, and unrecorded snapshots stay
        None. Only injected component context supplies legal named devices.
        Missing/rejecting providers permit only the explicit unnamed_device
        isolation choice, whose provider error the UI presents before submission.
        ``transpose_axes=True`` swaps only legacy Labber coordinates; it never
        overrides explicit native axes.

        Returns the spectrum name (basename of ``filepath``). InvalidInputError
        reports unknown_spectrum for absent inheritance or spectrum_not_2d
        for a missing frequency axis, before State publication.
        FailedPreconditionError reports alignment_not_seeded when the source has
        neither committed nor seeded alignment. Native I/O and
        parsing failures propagate without caller-correctable classification.
        """
        source = read_spectrum_source(filepath)
        snapshots: dict[str, DeviceSnapshotValue] | None = None
        unit = "native"
        device = "unnamed_device"
        if isinstance(source, Container):
            if selection is None:
                raise InvalidInputError(
                    "Native input requires variable and device_axis",
                    reason_code="native_selection_required",
                )
            projected = project_native_spectrum(
                source, variable=selection.variable, device_axis=selection.device_axis
            )
            dev_values = projected.dev_values
            freqs = projected.freqs_mhz * 1e6
            signals2d = projected.signals
            unit = projected.unit
            device = selection.device
            try:
                context = self.flux_context()
            except ValueError as error:
                if device != "unnamed_device":
                    raise
                logger.warning("%s; loading as explicit unnamed_device", error)
                context = None
            choices = (
                tuple(context.frame.devices)
                if context is not None and context.frame is not None
                else ("unnamed_device",)
            )
            if device not in choices:
                raise InvalidInputError(
                    f"Unknown native device {device!r}; choose from {list(choices)}",
                    reason_code="unknown_device",
                )
            if "metadata" in source.docs:
                snapshot = metadata_from_container(source).device_snapshot
                if snapshot is not None:
                    scalars = snapshot_scalar_values(
                        snapshot, context.units if context is not None else [device]
                    )
                    snapshots = {
                        name: DeviceSnapshotValue(value=value.value, unit=value.unit)
                        for name, value in scalars.items()
                    }
        else:
            device = "unnamed_device"
            dev_values, freqs, signals2d = self._labber_arrays(source, filepath)

        if transpose_axes and not isinstance(source, Container):
            signals2d, dev_values, freqs = transpose_spectrum_data(
                signals2d, dev_values, freqs
            )

        dev_values, freqs, signals2d = format_rawdata(dev_values, freqs, signals2d)

        flux_half, flux_int, flux_period = self._initial_alignment(inherit_from)
        fluxs = value2flux(dev_values, flux_half, flux_period)

        raw = SpectrumData(
            dev_values=dev_values.astype(np.float64),
            fluxs=np.asarray(fluxs, dtype=np.float64),
            freqs=freqs.astype(np.float64),
            signals=signals2d.astype(np.complex128),
        )

        name = os.path.basename(filepath)
        entry = SpectrumEntry(
            name=name,
            spec_type=spec_type,
            raw=raw,
            points=_empty_points(),
            flux_half=flux_half,
            flux_int=flux_int,
            flux_period=flux_period,
            device=device,
            unit=unit,
            device_snapshot=snapshots,
            alignment_seeded=inherit_from is not None,
            inherited_from=inherit_from,
        )
        self._state.put_spectrum(entry)
        logger.debug("load_spectrum: %r as %s", name, spec_type)
        return name

    @staticmethod
    def _labber_arrays(
        ld: LabberData, filepath: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(ld.axes) < 2:
            raise InvalidInputError(
                f"{filepath!r} has no frequency axis (not a 2D spectrum)",
                reason_code="spectrum_not_2d",
            )
        dev_values = np.asarray(ld.axes[0].values)
        freqs = np.asarray(ld.axes[1].values)
        # native load_labber_data returns z as (Ny, Nx) = (N_freq, N_dev); the
        # downstream pipeline (format_rawdata, SpectrumData) expects device-major
        # (N_dev, N_freq), so transpose the inner two axes back.
        signals2d = np.asarray(ld.z).T

        return dev_values, freqs, signals2d

    def _initial_alignment(
        self, inherit_from: str | None
    ) -> tuple[float, float, float]:
        """Seed alignment from an existing spectrum, or the identity default."""
        if inherit_from is None:
            return 0.0, 0.0, 1.0
        src = self._state.get_spectrum(inherit_from)
        if not src.can_seed_alignment:
            raise FailedPreconditionError(
                f"Spectrum {inherit_from!r} has no alignment to inherit",
                reason_code="alignment_not_seeded",
            )
        return src.flux_half, src.flux_int, src.flux_period

    def load_processed_spectrums(self, filepath: str) -> list[str]:
        """Restore a processed ``spectrums.hdf5`` (alignment + selected points).

        Each restored spectrum is aligned and picking-completed, including zero
        points, so it shows in ResultPreview. Return loaded names. Read the
        persisted spectrum type; legacy missing type defaults to "TwoTone".
        No workflow metadata is required beyond the processed spectrum payload.
        Reject the entire restore with InvalidInputError if any name already
        exists, listing all conflicts before publishing any spectrum.
        """
        spectrums = load_spectrums(filepath)
        conflicts = [name for name in spectrums if name in self._state.spectrums]
        if conflicts:
            raise InvalidInputError(
                "Processed spectrum names already exist: " + ", ".join(conflicts),
                reason_code="spectrum_name_conflict",
            )
        names: list[str] = []
        for name, result in spectrums.items():
            spec_type: SpecType = (
                "OneTone" if result.get("type") == "OneTone" else "TwoTone"
            )
            frame = FluxFrame(
                offset=result["offset"],
                devices={
                    result["device"]: DevicePeriod(result["period"], result["unit"])
                },
            )
            entry = SpectrumEntry(
                name=name,
                spec_type=spec_type,
                raw=result["spectrum"],
                points=result["points"],
                flux_half=device_value_at_flux(
                    frame, {}, device_name=result["device"], flux=0.5
                ),
                flux_int=device_value_at_flux(
                    frame, {}, device_name=result["device"], flux=0.0
                ),
                flux_period=result["period"],
                device=result["device"],
                unit=result["unit"],
                device_snapshot=result.get("device_snapshot"),
                aligned=True,
                points_completed=True,
                alignment_seeded=True,
            )
            self._state.put_spectrum(entry)
            names.append(name)
        logger.debug("load_processed_spectrums: %d from %r", len(names), filepath)
        return names
