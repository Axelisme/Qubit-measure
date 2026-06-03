"""SpectrumStore + SelectionService — collection queries and cross-spectrum
point-cloud selection.

SpectrumStore is a thin query/CRUD facade over State's spectrum collection.
SelectionService assembles the joint point cloud from every spectrum's selected
points and records the cross-spectrum selection mask.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from zcu_tools.fluxdep_gui.state import FluxDepState, SpectrumEntry

logger = logging.getLogger(__name__)


class SpectrumStore:
    """Query / CRUD facade over the spectrum collection."""

    def __init__(self, state: FluxDepState) -> None:
        self._state = state

    def list_spectrums(self) -> list[str]:
        return list(self._state.spectrums.keys())

    def get_spectrum(self, name: str) -> SpectrumEntry:
        return self._state.spectrums[name]

    def remove_spectrum(self, name: str) -> None:
        self._state.remove_spectrum(name)

    def set_active(self, name: Optional[str]) -> None:
        self._state.set_active(name)


class SelectionService:
    """Cross-spectrum joint point cloud + selection mask."""

    def __init__(self, state: FluxDepState) -> None:
        self._state = state

    def derive_pointcloud(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Concatenate every spectrum's selected (flux, freq) points.

        A derived value (not stored): the joint cloud the cross-spectrum selector
        operates on. Spectra are concatenated in insertion order; a spectrum with
        no selected points contributes nothing.
        """
        flux_parts: list[NDArray[np.float64]] = []
        freq_parts: list[NDArray[np.float64]] = []
        for entry in self._state.spectrums.values():
            flux_parts.append(np.asarray(entry.points["fluxs"], dtype=np.float64))
            freq_parts.append(np.asarray(entry.points["freqs"], dtype=np.float64))
        if not flux_parts:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty.copy()
        return np.concatenate(flux_parts), np.concatenate(freq_parts)

    def set_selection(self, selected: NDArray[np.bool_]) -> None:
        """Record the cross-spectrum selection mask over the joint point cloud.

        Fast-fails if the mask length disagrees with the current joint cloud size
        (a stale mask from before a spectrum was added/removed).
        """
        fluxs, _ = self.derive_pointcloud()
        if selected.shape[0] != fluxs.shape[0]:
            raise ValueError(
                f"selection mask length {selected.shape[0]} != joint point cloud "
                f"size {fluxs.shape[0]}"
            )
        self._state.set_selection(np.asarray(selected, dtype=np.bool_))
        logger.debug(
            "set_selection: n_selected=%d/%d", int(selected.sum()), selected.size
        )
