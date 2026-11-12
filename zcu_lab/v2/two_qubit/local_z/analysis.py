"""Saved-source local-Z quadratures and thermal ensemble phase diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.analysis.fitting.phase_line import PhaseLineFit


@dataclass(frozen=True)
class LocalZFit:
    """One target's phase-line summary, in degrees and degrees per gate.

    correction_deg is minus slope_deg without wrapping. intercept_deg is the
    preparation/projection offset, slope_error_deg is its count-derived 1σ
    uncertainty. Unavailable fits keep None and reason. fit_quality contains
    fixed-size residual and parameter-error diagnostics, not a quality gate.
    """

    intercept_deg: float | None
    slope_deg: float | None
    correction_deg: float | None
    slope_error_deg: float | None
    fit_quality: FitQuality | None
    reason: str | None


@dataclass(frozen=True)
class LocalZAnalysis:
    """Fixed F1/F2 phase-line summaries in target order; no scan arrays."""

    fits: tuple[LocalZFit, ...]


def summarize_fit(fit: PhaseLineFit) -> LocalZFit:
    """Keep calibration values and fit diagnostics; leave curves for plotting."""
    return LocalZFit(
        fit.intercept_deg,
        fit.slope_deg,
        None if fit.slope_deg is None else -fit.slope_deg,
        fit.slope_error_deg,
        fit.fit_quality,
        fit.reason,
    )
