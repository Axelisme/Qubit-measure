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
    def allocate(
        cls, flux: NDArray[np.float64], detune: NDArray[np.float64]
    ) -> QubitFreqResult:
        """Pre-allocate a Result over the full ``flux`` axis × ``detune``.

        The flux axis is filled up front (known at Run start, used as the
        Plotter's x); the trailing signal / fit fields stay nan until each
        ``produce`` fills its row. The detune axis is param-only.
        """
        fx = np.asarray(flux, dtype=np.float64)
        det = np.asarray(detune, dtype=np.float64)
        n_flux, n_detune = fx.shape[0], det.shape[0]
        return cls(
            flux=fx,
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


@dataclass
class Sweep1DResult:
    """A generic (n_flux, n_x) sweep Result for the 1D experiments.

    Shared by t1 (x=relax time), lenrabi (x=pulse length), t2ramsey / t2echo
    (x=delay time), and mist (x=gain) — all sweep one trailing axis per flux
    point and (except mist) fit a single scalar from the row.

    - ``flux`` — (n_flux,) flux axis, filled per row by ``produce``.
    - ``x`` — (n_x,) the trailing sweep axis (param-only, same every row); its
      meaning (``x_label``) and unit are Node-type domain knowledge.
    - ``signal`` — (n_flux, n_x) the rotated-to-real signal, filled per row.
    - ``fit_curve`` — (n_flux, n_x) the fitted curve per row (nan until fit;
      all-nan for mist, which has no fit).
    - ``fit_value`` — (n_flux,) the primary fitted scalar per row (t1 / pi_length
      / t2 …; nan for mist or a failed fit). What the colormap overlay tracks.
    - ``x_label`` — the trailing-axis label (for the Plotter's x axis).
    """

    flux: NDArray[np.float64]
    x: NDArray[np.float64]
    signal: NDArray[np.float64]
    fit_curve: NDArray[np.float64]
    fit_value: NDArray[np.float64]
    x_label: str = "x"

    @classmethod
    def allocate(
        cls, flux: NDArray[np.float64], x: NDArray[np.float64], x_label: str = "x"
    ) -> Sweep1DResult:
        fx = np.asarray(flux, dtype=np.float64)
        xs = np.asarray(x, dtype=np.float64)
        n_flux, n_x = fx.shape[0], xs.shape[0]
        return cls(
            flux=fx,
            x=xs,
            signal=np.full((n_flux, n_x), np.nan, dtype=np.float64),
            fit_curve=np.full((n_flux, n_x), np.nan, dtype=np.float64),
            fit_value=np.full(n_flux, np.nan, dtype=np.float64),
            x_label=x_label,
        )

    @property
    def n_flux(self) -> int:
        return self.flux.shape[0]

    @property
    def n_x(self) -> int:
        return self.x.shape[0]


@dataclass
class Sweep2DResult:
    """ro_optimize's (n_flux, n_freq, n_gain) Result: a per-flux 2D landscape.

    Each flux point sweeps freq × gain and takes the argmax (no fit). Unlike the
    accumulating 1D maps, the Plotter shows only the *current* flux row's
    freq×gain landscape with the peak marked (overwrite, not accumulate).

    - ``flux`` — (n_flux,) flux axis, filled per row.
    - ``freq`` / ``gain`` — the two trailing axes (param-only).
    - ``signal`` — (n_flux, n_freq, n_gain) the magnitude landscape per row.
    - ``best_freq`` / ``best_gain`` — (n_flux,) the argmax point per row.
    """

    flux: NDArray[np.float64]
    freq: NDArray[np.float64]
    gain: NDArray[np.float64]
    signal: NDArray[np.float64]
    best_freq: NDArray[np.float64]
    best_gain: NDArray[np.float64]

    @classmethod
    def allocate(
        cls,
        flux: NDArray[np.float64],
        freq: NDArray[np.float64],
        gain: NDArray[np.float64],
    ) -> Sweep2DResult:
        fx = np.asarray(flux, dtype=np.float64)
        f = np.asarray(freq, dtype=np.float64)
        g = np.asarray(gain, dtype=np.float64)
        n_flux = fx.shape[0]
        return cls(
            flux=fx,
            freq=f,
            gain=g,
            signal=np.full((n_flux, f.shape[0], g.shape[0]), np.nan, dtype=np.float64),
            best_freq=np.full(n_flux, np.nan, dtype=np.float64),
            best_gain=np.full(n_flux, np.nan, dtype=np.float64),
        )

    @property
    def n_flux(self) -> int:
        return self.flux.shape[0]
