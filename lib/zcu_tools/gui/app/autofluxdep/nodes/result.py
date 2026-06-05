"""Result — a Node's sweep-lived, flux-aware domain output (see CONTEXT.md).

A Result is distinct from a Patch. The **Patch** is the provides scalars other
Nodes consume; the **Result** is *this Node's own complete output* — the raw 2D
signals, per-point frequency axes, fit curves — used by the Plotter and (later)
for saving. Both come from the same per-point fit, so they cannot disagree.

A Result is **sweep-lived and flux-aware**: the flux axis is always the first
dimension, pre-allocated nan-filled at Run start (``Builder.make_init_result``)
and filled in place one flux row at a time by ``Node.produce``. Unlike a runner
Task it is NOT list-and-merged — the Node knows the workflow sweeps flux and
carries the flux axis directly.

The trailing dimensions and the drawing differ by Node type — which is why each
Result is a per-NodeType domain dataclass (not a generic base) built by that
Builder's ``make_init_result`` and drawn by that Builder's Plotter. There is no
abstract Result base: the orchestrator never touches a Result (it is curried
into the Node by ``build_node`` and read only by the Node + the Plotter), so it
needs no common interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class QubitFreqResult:
    """qubit_freq's sweep Result: a (n_flux, n_detune) signal map + per-row fit.

    Pre-allocated nan-filled at Run start; each flux row filled in place by the
    qubit_freq Node's ``produce``:

    - ``flux`` — the (n_flux,) flux axis, filled per row by ``produce`` (each
      point knows its own flux value); nan where not yet measured.
    - ``detune`` — the (n_detune,) detune axis in MHz (param-only, same every
      row; the *absolute* freq axis shifts per row with ``predict_freq``, but
      the detune extent is fixed by the ``detune_sweep`` param).
    - ``signal`` — (n_flux, n_detune) the rotated-to-real signal, overwritten
      each acquire round for the current row (grows clearer round by round).
    - ``fit_curve`` — (n_flux, n_detune) the fitted Lorentzian per row, filled
      only after acquire+fit completes (stays nan during acquire).
    - ``fit_freq`` — (n_flux,) the fitted absolute qubit frequency per row
      (nan until that row's fit succeeds).
    - ``predict_freq`` — (n_flux,) the predicted frequency the detune axis was
      centred on (so the absolute freq of column j at row i is
      ``predict_freq[i] + detune[j]``).

    A row stays nan where the sweep stopped or a Node was skipped — an honest
    "not measured", never truncated. The Plotter must tolerate a mid-acquire row
    that has ``signal`` but nan ``fit_curve`` / ``fit_freq``.
    """

    flux: NDArray[np.float64]
    detune: NDArray[np.float64]
    signal: NDArray[np.float64]
    fit_curve: NDArray[np.float64]
    fit_freq: NDArray[np.float64]
    predict_freq: NDArray[np.float64]

    @classmethod
    def allocate(cls, n_flux: int, detune: NDArray[np.float64]) -> "QubitFreqResult":
        """Pre-allocate a nan-filled Result for ``n_flux × n_detune``.

        The flux axis is nan up front (each ``produce`` writes its own row's
        flux value); the detune axis is param-only and identical every row.
        """
        det = np.asarray(detune, dtype=np.float64)
        n_detune = det.shape[0]
        return cls(
            flux=np.full(n_flux, np.nan, dtype=np.float64),
            detune=det,
            signal=np.full((n_flux, n_detune), np.nan, dtype=np.float64),
            fit_curve=np.full((n_flux, n_detune), np.nan, dtype=np.float64),
            fit_freq=np.full(n_flux, np.nan, dtype=np.float64),
            predict_freq=np.full(n_flux, np.nan, dtype=np.float64),
        )

    @property
    def n_flux(self) -> int:
        return self.flux.shape[0]

    @property
    def n_detune(self) -> int:
        return self.detune.shape[0]
