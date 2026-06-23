"""LoadService — read a raw one-tone hdf5 into the State's OnetoneEntry.

Depends only on the low-level ``load_labber_data`` (pure hdf5 IO) and
``format_rawdata`` (Hz→GHz + monotonic ordering); it does NOT import
``experiment.v2`` (whose ``FluxDepExp.load`` returns a ``FluxDepResult`` dataclass
in MHz and would pull the whole measure experiment layer in). Frequencies are
stored in GHz.

The flux axis is derived from the project's fluxonium fit alignment
(``flux_half`` / ``flux_period`` from the ``fluxdep_fit`` section), so a one-tone
can only be loaded after the fit inputs are present.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from zcu_tools.gui.app.dispersive.state import DispersiveState, OnetoneEntry
from zcu_tools.notebook.persistance import SpectrumData, format_rawdata
from zcu_tools.simulate import value2flux
from zcu_tools.utils.labber_io import load_labber_data

logger = logging.getLogger(__name__)


def transpose_spectrum_data(
    signals2d: np.ndarray, dev_values: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Swap the device-value and frequency axes of a raw spectrum.

    Some legacy files store the axes transposed (x=frequency, y=flux) versus what
    the loader expects (x=flux, y=frequency). This swaps ``dev_values`` ↔ ``freqs``
    and transposes the signal grid, returning ``(signals.T, freqs, dev_values)``.
    """
    return signals2d.T, freqs, dev_values


class LoadService:
    """Loads a raw one-tone hdf5 into the State as the single OnetoneEntry."""

    def __init__(self, state: DispersiveState) -> None:
        self._state = state

    def load_onetone(self, filepath: str, transpose_axes: bool = False) -> str:
        """Load ``filepath`` as the one-tone spectrum and write it into State.

        The flux axis is derived via ``value2flux`` from the fit's ``flux_half`` /
        ``flux_period`` — so this fast-fails when the fluxonium inputs have not yet
        been loaded (run ``load_fit_inputs`` first). ``transpose_axes=True`` swaps
        the device-value and frequency axes for legacy x=frequency / y=flux files.

        Returns the spectrum name (basename of ``filepath``).
        """
        inputs = self._state.fit_inputs
        if inputs is None:
            raise RuntimeError(
                "load the fluxonium fit inputs (params.json) before a one-tone "
                "spectrum — the flux axis is derived from the fit's alignment"
            )

        ld = load_labber_data(filepath)
        if len(ld.axes) < 2:
            raise ValueError(f"{filepath!r} has no frequency axis (not a 2D spectrum)")
        dev_values = np.asarray(ld.axes[0].values)  # inner axis (x)
        freqs = np.asarray(ld.axes[1].values)  # outer axis (y)
        # ``load_labber_data`` returns z in (Ny, Nx) = (Nfreq, Ndev) order and does
        # NOT flip the inner two axes; downstream (``format_rawdata`` and the rest
        # of this loader) expects dev-major (Ndev, Nfreq), so transpose to restore
        # the orientation the deleted dict ``load_data`` used to provide.
        signals2d = np.asarray(ld.z).T

        if transpose_axes:
            signals2d, dev_values, freqs = transpose_spectrum_data(
                signals2d, dev_values, freqs
            )

        dev_values, freqs, signals2d = format_rawdata(dev_values, freqs, signals2d)
        fluxs = value2flux(dev_values, inputs.flux_half, inputs.flux_period)

        raw = SpectrumData(
            dev_values=dev_values.astype(np.float64),
            fluxs=np.asarray(fluxs, dtype=np.float64),
            freqs=freqs.astype(np.float64),
            signals=signals2d.astype(np.complex128),
        )
        name = os.path.basename(filepath)
        self._state.set_onetone(OnetoneEntry(name=name, raw=raw))
        logger.debug("load_onetone: %r (transpose=%s)", name, transpose_axes)
        return name
