"""Per-original-tone T1 fits on the common requested axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from zcu_tools.analysis.fitting import compute_fit_quality, fit_decay, fit_dual_decay
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2.twotone.time_domain.t1.core import (
    T1Analysis,
    T1AnalyzeOptions,
    t1_signal2real,
)

if TYPE_CHECKING:
    from .core import MuxT1Result


@dataclass(frozen=True)
class MuxT1Fit:
    """Original full-tone index and existing single-qubit T1 numerical analysis.

    fit holds t1/t1_err in us, optional second decay component in us and the
    existing fit quality mapping under 'fit'. It owns no figures or source cfg.
    """

    index: int
    fit: T1Analysis


def analyze_t1(
    result: MuxT1Result, options: T1AnalyzeOptions, *, plots: Plots
) -> tuple[MuxT1Fit | MuxFitFailure, ...]:
    """Fit each IQ row against shared requested us after skip/finite filtering.

    Require float64 nonnegative strictly increasing times (P,), int64 unique
    nonnegative original indices (K,) and complex128 signals (K,P), with K,P>0.
    NaN IQ is allowed for partial acquisition. Invalid sources and negative or
    noninteger skip raise ValueError before fitting. Existing single/dual decay
    fitters and quality consume only acquired points. Numeric failures return
    indexed MuxFitFailure reasons. plots owns fit_i, independent of duplicate
    readout labels; caller finishes/releases figures. No cfg or hardware is read.
    """
    times, indices, signals = result.times, result.tone_indices, result.signals
    if (
        times.ndim != 1
        or times.dtype != np.dtype(np.float64)
        or not times.size
        or not np.all(np.isfinite(times))
        or np.any(times < 0)
        or np.any(np.diff(times) <= 0)
        or indices.ndim != 1
        or indices.dtype != np.dtype(np.int64)
        or not indices.size
        or np.any(indices < 0)
        or len(np.unique(indices)) != len(indices)
        or signals.dtype != np.dtype(np.complex128)
        or signals.shape != (len(indices), len(times))
    ):
        raise ValueError("invalid MuxT1Result axes or IQ shape/dtype")
    if type(options.skip) is not int or options.skip < 0:
        raise ValueError("skip must be a nonnegative integer")
    outcomes: list[MuxT1Fit | MuxFitFailure] = []
    for row, raw_index in enumerate(indices):
        index = int(raw_index)
        signal = signals[row, options.skip :]
        finite = np.isfinite(signal)
        xs = times[options.skip :][finite]
        signal = signal[finite]
        fig, ax = plots.subplots(f"fit_{index}", figsize=(8, 5))
        ax.set(xlabel="Requested delay (us)", ylabel="Signal (a.u.)")
        fig.suptitle(f"tone {index}")
        try:
            minimum = 6 if options.dual_exp else 4
            if len(xs) < minimum:
                raise ValueError(f"T1 fit requires at least {minimum} acquired points")
            real = t1_signal2real(signal)
            if options.dual_exp:
                t1, err, t1b, errb, fitted, (params, covariance) = fit_dual_decay(
                    xs, real
                )
                names = ("y0", "yscale1", "decay_time1", "yscale2", "decay_time2")
                estimates = (t1, err, t1b, errb)
            else:
                t1, err, fitted, (params, covariance) = fit_decay(xs, real)
                names = ("y0", "yscale", "decay_time")
                t1b, errb = None, None
                estimates = (t1, err)
            if not np.all(np.isfinite(estimates)):
                raise ValueError("T1 fit returned nonfinite estimates")
            quality = compute_fit_quality(
                real,
                fitted,
                {name: float(value) for name, value in zip(names, params, strict=True)},
                covariance,
            )
            fit = T1Analysis(float(t1), float(err), t1b, errb, {"fit": quality})
            outcomes.append(MuxT1Fit(index, fit))
            ax.plot(xs, real, "o", label="measured")
            ax.plot(xs, fitted, label=f"T1={t1:.4g} ± {err:.2g} us")
            if options.dual_exp:
                ax.set_title(f"Second decay: {t1b:.4g} ± {errb:.2g} us")
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            np.linalg.LinAlgError,
        ) as exc:
            outcomes.append(MuxFitFailure(index, str(exc)))
            ax.plot(xs, np.abs(signal), "o", label="acquired magnitude")
            ax.text(0.02, 0.98, str(exc), transform=ax.transAxes, va="top", wrap=True)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
    return tuple(outcomes)
