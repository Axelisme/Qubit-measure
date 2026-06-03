"""ExportService — write the spectrum collection to spectrums.hdf5.

Assembles each SpectrumEntry into a persistance ``SpectrumResult`` and delegates
to ``dump_spectrums``. The default path follows the notebook layout
(``result_dir/data/fluxdep/spectrums.hdf5``); the directory is created only here,
at the command boundary, not on a pure query.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from zcu_tools.fluxdep_gui.state import FluxDepState
from zcu_tools.notebook.persistance import SpectrumResult, dump_spectrums

logger = logging.getLogger(__name__)

DEFAULT_CHIP = "unknown_chip"
DEFAULT_QUBIT = "unknown_qubit"


def default_export_path(chip_name: str, qub_name: str) -> str:
    """The notebook-layout default export path for a chip/qubit.

    ``result/<chip>/<qubit>/data/fluxdep/spectrums.hdf5``. Empty names fall back
    to ``unknown_chip`` / ``unknown_qubit`` so the path is always well-formed.
    """
    chip = chip_name or DEFAULT_CHIP
    qub = qub_name or DEFAULT_QUBIT
    return os.path.join("result", chip, qub, "data", "fluxdep", "spectrums.hdf5")


class ExportService:
    """Exports the spectrum collection to a spectrums.hdf5 file."""

    def __init__(self, state: FluxDepState) -> None:
        self._state = state

    def default_path(self) -> str:
        """Default export path from the project's chip/qubit names."""
        return default_export_path(
            self._state.project.chip_name, self._state.project.qub_name
        )

    def export_spectrums(self, filepath: Optional[str] = None, mode: str = "x") -> str:
        """Write every loaded spectrum to ``filepath`` (or the default path).

        Fast-fails if the collection is empty. ``mode`` is the h5py file mode
        (default ``"x"`` = create, fail if exists; pass ``"w"`` to overwrite).
        Returns the resolved path.
        """
        if not self._state.spectrums:
            raise ValueError("no spectra to export")

        path = filepath if filepath is not None else self.default_path()
        spectrums: dict[str, SpectrumResult] = {
            name: SpectrumResult(
                type=entry.spec_type,
                flux_half=entry.flux_half,
                flux_int=entry.flux_int,
                flux_period=entry.flux_period,
                spectrum=entry.raw,
                points=entry.points,
            )
            for name, entry in self._state.spectrums.items()
        }
        dump_spectrums(path, spectrums, mode=mode)
        logger.debug("export_spectrums: %d spectra -> %r", len(spectrums), path)
        return path
