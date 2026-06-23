"""LoadService — read a raw spectrum hdf5 into a SpectrumEntry.

Depends only on the low-level ``load_labber_data`` (pure hdf5 IO) and
``format_rawdata`` (Hz→GHz + monotonic ordering); it does NOT import
``experiment.v2`` (that would pull the whole measure experiment layer into
fluxdep). OneTone and TwoTone load identically — ``spec_type`` is metadata
recorded on the entry and only branches the point-selection tool downstream.

This module is the synchronous core. Wrapping it in a worker thread + OperationGate
(so a slow load degrades to an operation handle, like measure's run/device setup)
is layered on top later; keeping the core sync makes it directly testable.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from zcu_tools.gui.app.fluxdep.state import (
    FluxDepState,
    SpectrumEntry,
    SpecType,
)
from zcu_tools.notebook.persistance import (
    PointsData,
    SpectrumData,
    format_rawdata,
    load_spectrums,
)
from zcu_tools.simulate import value2flux
from zcu_tools.utils.labber_io import load_labber_data

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

    def __init__(self, state: FluxDepState) -> None:
        self._state = state

    def load_spectrum(
        self,
        filepath: str,
        spec_type: SpecType,
        inherit_from: str | None = None,
        transpose_axes: bool = False,
    ) -> str:
        """Load ``filepath`` as a new spectrum and write it into State.

        The spectrum name is the file's basename. ``inherit_from`` (an existing
        spectrum name) seeds this spectrum's flux alignment as an initial guess;
        omitted, the alignment starts at the identity (flux_half=0, period=1) and
        is refined later by the line-picker. The loaded ``fluxs`` are derived from
        whatever alignment is in effect at load time (re-derived on alignment).

        ``transpose_axes=True`` swaps the device-value and frequency axes at load
        time — for legacy files that store x=frequency / y=flux (the transpose of
        the expected x=flux / y=frequency).

        Returns the spectrum name (basename of ``filepath``).
        """
        ld = load_labber_data(filepath)
        dev_values = np.asarray(ld.axes[0].values)
        if len(ld.axes) < 2:
            raise ValueError(f"{filepath!r} has no frequency axis (not a 2D spectrum)")
        freqs = np.asarray(ld.axes[1].values)
        # native load_labber_data returns z as (Ny, Nx) = (N_freq, N_dev); the
        # downstream pipeline (format_rawdata, SpectrumData) expects device-major
        # (N_dev, N_freq), so transpose the inner two axes back.
        signals2d = np.asarray(ld.z).T

        if transpose_axes:
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
            # inherited alignment is a meaningful seed for the line-picker; a
            # fresh load (identity default) is not.
            alignment_seeded=inherit_from is not None,
        )
        self._state.put_spectrum(entry)
        logger.debug(
            "load_spectrum: %r as %s (inherit_from=%r)", name, spec_type, inherit_from
        )
        return name

    def _initial_alignment(
        self, inherit_from: str | None
    ) -> tuple[float, float, float]:
        """Seed alignment from an existing spectrum, or the identity default."""
        if inherit_from is None:
            return 0.0, 0.0, 1.0
        src = self._state.spectrums.get(inherit_from)
        if src is None:
            raise KeyError(f"inherit_from spectrum {inherit_from!r} not loaded")
        return src.flux_half, src.flux_int, src.flux_period

    def load_processed_spectrums(self, filepath: str) -> list[str]:
        """Restore a processed ``spectrums.hdf5`` (alignment + selected points).

        Each restored spectrum lands fully advanced — aligned and points-selected
        — so it shows in the result-preview stage. Returns the loaded names. NOTE:
        ``dump_spectrums`` does not persist ``spec_type``; a missing type defaults
        to ``"TwoTone"`` (the user can re-select points to change tooling).
        """
        spectrums = load_spectrums(filepath)
        names: list[str] = []
        for name, result in spectrums.items():
            spec_type: SpecType = (
                "OneTone" if result.get("type") == "OneTone" else "TwoTone"
            )
            entry = SpectrumEntry(
                name=name,
                spec_type=spec_type,
                raw=result["spectrum"],
                points=result["points"],
                flux_half=result["flux_half"],
                flux_int=result["flux_int"],
                flux_period=result["flux_period"],
                aligned=True,
                points_selected=result["points"]["freqs"].size > 0,
                alignment_seeded=True,
            )
            self._state.put_spectrum(entry)
            names.append(name)
        logger.debug("load_processed_spectrums: %d from %r", len(names), filepath)
        return names
