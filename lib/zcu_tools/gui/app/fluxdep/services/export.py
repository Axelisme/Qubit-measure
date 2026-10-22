"""ExportService — write the spectrum collection to spectrums.hdf5.

Assembles each SpectrumEntry into an analysis.fluxdep.models ``SpectrumResult`` and delegates
to ``dump_spectrums``. The default path uses the entry layout
(``<entry>/fluxdep/spectrums.hdf5``); the directory is created only here,
at the command boundary, not on a pure query.
"""

from __future__ import annotations

import logging
import os

from zcu_tools.analysis.fluxdep.io import dump_spectrums
from zcu_tools.analysis.fluxdep.models import SpectrumResult
from zcu_tools.gui.app.analysis_project.entry import require_target
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.expected_error import FailedPreconditionError

logger = logging.getLogger(__name__)

__all__ = [
    "default_export_path",
    "ExportService",
]


def default_export_path(result_path: str) -> str:
    """Return <entry result_path>/fluxdep/spectrums.hdf5 without creating it."""
    return os.path.join(result_path, "fluxdep", "spectrums.hdf5")


class ExportService:
    """Exports the spectrum collection to a spectrums.hdf5 file."""

    def __init__(self, state: FluxDepState) -> None:
        self._state = state

    def default_path(self) -> str:
        """Return the opened entry intermediate path; reject no_entry."""
        return default_export_path(
            str(require_target(self._state.project).resource.result_path)
        )

    def export_spectrums(self, filepath: str | None = None, mode: str = "x") -> str:
        """Write every loaded spectrum to ``filepath`` (or the default path).

        Fast-fails if the collection is empty. ``mode`` is the h5py file mode
        (default ``"x"`` = create, fail if exists; pass ``"w"`` to overwrite).
        Returns the resolved path. An empty collection raises
        FailedPreconditionError (no_spectrums) before creating any file.
        Native I/O failures propagate without classification.
        """
        if not self._state.spectrums:
            raise FailedPreconditionError(
                "no spectra to export", reason_code="no_spectrums"
            )

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
