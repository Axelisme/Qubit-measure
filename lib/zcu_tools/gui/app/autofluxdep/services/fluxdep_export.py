"""Fluxdep-compatible export sidecar derived from QubitFreqResult."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from zcu_tools.gui.app.autofluxdep.experiments._support.result import QubitFreqResult
from zcu_tools.utils.datasaver import save_labber_data


def export_qubit_freq_fluxdep_spectrum(
    result: QubitFreqResult,
    filepath: str | Path,
    *,
    flux_unit: str = "",
    committed_mask: NDArray[np.bool_] | None = None,
) -> str:
    """Write a fluxdep raw Labber spectrum from a qubit_freq Result.

    ``QubitFreqResult`` stores detune-relative columns whose absolute frequency
    is row-local: ``predict_freq[row] + detune[col]``. Fluxdep raw loader accepts
    one common absolute frequency axis, so each committed row is interpolated onto
    a common MHz grid before writing. Values outside a row's measured span remain
    NaN.
    """
    committed = _committed_mask(result, committed_mask)
    common_freq_mhz = _common_frequency_grid(result, committed)
    exported = np.full(
        (result.n_flux, common_freq_mhz.shape[0]), np.nan, dtype=np.complex128
    )
    for row_idx in range(result.n_flux):
        if not committed[row_idx]:
            continue
        predict = result.predict_freq[row_idx]
        row = result.signal[row_idx]
        if not np.isfinite(predict) or np.isnan(row).all():
            continue
        absolute = np.asarray(predict + result.detune, dtype=np.float64)
        order = np.argsort(absolute)
        xs = absolute[order]
        ys = np.asarray(row, dtype=np.complex128)[order]
        valid = np.isfinite(xs) & np.isfinite(ys.real) & np.isfinite(ys.imag)
        if valid.sum() < 2:
            continue
        exported[row_idx] = _interp_complex(common_freq_mhz, xs[valid], ys[valid])

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return save_labber_data(
        str(path),
        z=("Signal", "a.u.", exported.T),
        axes=[
            ("Flux device value", flux_unit, result.flux),
            ("Frequency", "Hz", common_freq_mhz * 1e6),
        ],
    )


def _committed_mask(
    result: QubitFreqResult, committed_mask: NDArray[np.bool_] | None
) -> NDArray[np.bool_]:
    if committed_mask is None:
        return np.ones(result.n_flux, dtype=np.bool_)
    mask = np.asarray(committed_mask, dtype=np.bool_)
    if mask.shape != (result.n_flux,):
        raise ValueError(
            f"committed_mask shape {mask.shape} must match n_flux {result.n_flux}"
        )
    return mask


def _common_frequency_grid(
    result: QubitFreqResult,
    committed_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    finite_predict = result.predict_freq[
        committed_mask & np.isfinite(result.predict_freq)
    ]
    if finite_predict.size == 0:
        raise ValueError("qubit_freq export needs at least one finite predict_freq")
    detune = np.asarray(result.detune, dtype=np.float64)
    if detune.ndim != 1 or detune.size == 0:
        raise ValueError("qubit_freq export needs a non-empty detune axis")
    sorted_detune = np.sort(detune)
    if sorted_detune.size == 1:
        step = 1.0
    else:
        diffs = np.diff(sorted_detune)
        finite_diffs = np.abs(diffs[np.isfinite(diffs) & (diffs != 0.0)])
        if finite_diffs.size == 0:
            raise ValueError("qubit_freq export cannot derive frequency grid step")
        step = float(np.median(finite_diffs))
    row_min = finite_predict + float(np.nanmin(detune))
    row_max = finite_predict + float(np.nanmax(detune))
    start = float(np.nanmin(row_min))
    stop = float(np.nanmax(row_max))
    npts = int(round((stop - start) / step)) + 1
    if npts <= 1:
        return np.array([start], dtype=np.float64)
    return start + step * np.arange(npts, dtype=np.float64)


def _interp_complex(
    grid: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    real = np.interp(grid, xs, ys.real, left=np.nan, right=np.nan)
    imag = np.interp(grid, xs, ys.imag, left=np.nan, right=np.nan)
    return real + 1j * imag


__all__ = ["export_qubit_freq_fluxdep_spectrum"]
