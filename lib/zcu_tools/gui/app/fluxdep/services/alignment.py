"""AlignmentService + PointsService — record line-picking / point-selection
results into State, deriving the flux coordinate from the device-value axis.

Both are thin: the interactive picking happens in the UI widgets; these services
take the resulting scalars/arrays, compute the flux mapping (``value2flux``), and
write the spectrum entry on the Qt main thread.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.models import PointsData
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.simulate import value2flux

logger = logging.getLogger(__name__)


class AlignmentService:
    """Writes a spectrum's flux alignment (from the line-picker)."""

    def __init__(self, state: FluxDepState) -> None:
        self._state = state

    def set_alignment(self, name: str, flux_half: float, flux_int: float) -> None:
        """Record half/integer flux positions; derive period and re-map fluxs.

        ``flux_period = 2 * |flux_int - flux_half|`` (the device-value span of one
        flux quantum). Re-map both raw and selected-point fluxs from native
        device values in one State commit, preserving points and completion.
        Unknown name raises KeyError; a zero period or nonfinite mapped arrays
        raise ValueError without mutation. Must run on the State owner thread.
        """
        entry = self._state.spectrums[name]
        flux_period = 2.0 * abs(flux_int - flux_half)
        if flux_period == 0.0:
            raise ValueError("flux_int must differ from flux_half (period is zero)")

        # Compute the re-mapped flux axis here (this service owns the value2flux
        # mapping), then hand it to State, which writes the axis + alignment
        # scalars together under one version bump (no bypassing the boundary).
        new_fluxs = np.asarray(
            value2flux(entry.raw["dev_values"], flux_half, flux_period),
            dtype=np.float64,
        )
        new_point_fluxs = np.asarray(
            value2flux(entry.points["dev_values"], flux_half, flux_period),
            dtype=np.float64,
        )
        self._state.set_alignment(
            name, flux_half, flux_int, flux_period, new_fluxs, new_point_fluxs
        )
        logger.debug(
            "set_alignment: %r half=%g int=%g period=%g",
            name,
            flux_half,
            flux_int,
            flux_period,
        )

    def reset_alignment(self, name: str) -> None:
        """Reopen named spectrum's alignment, keeping points/completion; KeyError if absent."""
        self._state.reset_alignment(name)


class PointsService:
    """Writes a spectrum's selected points (from the point-selection widget)."""

    def __init__(self, state: FluxDepState) -> None:
        self._state = state

    def reset_points(self, name: str) -> None:
        """Clear named spectrum's points/completion; KeyError if absent, ValueError if unaligned."""
        self._state.reset_points(name)

    def set_points(
        self, name: str, dev_values: NDArray[np.float64], freqs: NDArray[np.float64]
    ) -> None:
        """Record selected (dev_value, freq) points, sorted by device value.

        Complete picking even for zero points. Derive flux coordinates from the
        spectrum's current alignment, so points share its flux mapping.
        """
        entry = self._state.spectrums[name]
        order = np.argsort(dev_values)
        sorted_devs = np.asarray(dev_values, dtype=np.float64)[order]
        sorted_freqs = np.asarray(freqs, dtype=np.float64)[order]
        fluxs = value2flux(sorted_devs, entry.flux_half, entry.flux_period)

        points = PointsData(
            dev_values=sorted_devs,
            fluxs=np.asarray(fluxs, dtype=np.float64),
            freqs=sorted_freqs,
        )
        self._state.set_points(name, points)
        logger.debug("set_points: %r n=%d", name, sorted_devs.size)
