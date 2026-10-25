"""Record spectroscopy decision inputs without changing scientific acceptance."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from zcu_tools.experiment.v2.utils.snr import estimate_snr

from zcu_lab.v2.twotone.freq.core import FreqResult, qubfreq_signal2real
from zcu_lab.workflows.common.calibration_analysis import FrequencyFit
from zcu_lab.workflows.common.calibration_results import (
    FrequencyDiagnostics,
    FrequencyScan,
)
from zcu_lab.workflows.common.gates import Gate, gate

if TYPE_CHECKING:
    from zcu_lab.workflows.common.calibration import CalibrationTunables

# Shared with accept_frequency_fit; this is not a new acceptance threshold.
MAX_FREQUENCY_MEAN_RESIDUAL_RATIO = 0.2


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def scan_diagnostics(result: FreqResult, candidate_count: int) -> FrequencyDiagnostics:
    """Record full-trace SNR and candidate presence before any local fitting."""
    return FrequencyDiagnostics(
        trace_snr=_finite(estimate_snr(qubfreq_signal2real(result.signals))),
        gates=(gate("candidate_count", ">=", 1, candidate_count),),
    )


def selected_peak_diagnostics(
    scan: FrequencyScan, tun: CalibrationTunables
) -> FrequencyDiagnostics:
    """Record the detected target's SNR and two-step truncation decision.

    Peak detection has already filtered candidates using its captured tunables.
    single mode does not reject truncated targets. Preserve full-trace metrics.
    """
    peak = scan.selected_peak
    gates = (
        gate("candidate_present", "==", 1, float(peak is not None)),
        gate(
            "selected_peak_snr",
            ">=",
            tun.peak_min_snr,
            None if peak is None else peak.snr,
        ),
        gate(
            "selected_peak_not_truncated",
            "==",
            1,
            None if peak is None else float(not peak.truncated),
            applied=scan.stage != "single",
        ),
    )
    return FrequencyDiagnostics(
        trace_snr=scan.diagnostics.trace_snr,
        selected_peak_center_mhz=None if peak is None else _finite(peak.center),
        peak_snr=None if peak is None else _finite(peak.snr),
        gates=gates,
    )


def fit_diagnostics(
    fit: FrequencyFit, scan: FrequencyScan, tun: CalibrationTunables
) -> FrequencyDiagnostics:
    """Record fitted MHz, residual metrics and each existing frequency fit gate.

    Acceptance gates mirror accept_frequency_fit, but are diagnostic-only for
    coarse scans. Coarse fits instead require a center within the isolated fit
    interval. Neither gate results nor serialized finite values drive acceptance.
    """
    span = float(np.ptp(fit.fitted))
    mae_ratio = (
        float(np.mean(np.abs(fit.observed - fit.fitted))) / span if span > 0 else None
    )
    relative_error = fit.error_mhz / fit.width_mhz if fit.width_mhz != 0 else None
    margin = tun.frequency_edge_margin * float(np.ptp(fit.source.freqs))
    applied = scan.stage != "coarse"
    gates = scan.diagnostics.gates + (
        gate("positive_center_mhz", ">", 0, fit.frequency_mhz, applied=applied),
        gate(
            "center_window_lower_mhz",
            ">=",
            float(np.min(fit.source.freqs)) + margin,
            fit.frequency_mhz,
            applied=applied,
        ),
        gate(
            "center_window_upper_mhz",
            "<=",
            float(np.max(fit.source.freqs)) - margin,
            fit.frequency_mhz,
            applied=applied,
        ),
        gate("positive_fwhm_mhz", ">", 0, fit.width_mhz, applied=applied),
        gate("nonnegative_center_error_mhz", ">=", 0, fit.error_mhz, applied=applied),
        gate(
            "center_error_over_fwhm",
            "<",
            tun.max_frequency_rel_err,
            relative_error,
            applied=applied,
        ),
        gate(
            "normalized_rms",
            "<",
            tun.max_fit_residual,
            fit.quality.normalized_residual_rms,
            applied=applied,
        ),
        gate(
            "mae_over_fit_ptp",
            "<",
            MAX_FREQUENCY_MEAN_RESIDUAL_RATIO,
            mae_ratio,
            applied=applied,
        ),
    )
    if scan.stage == "coarse":
        gates += (
            gate(
                "coarse_interval_lower_mhz",
                ">=",
                float(np.min(fit.freqs)),
                fit.frequency_mhz,
            ),
            gate(
                "coarse_interval_upper_mhz",
                "<=",
                float(np.max(fit.freqs)),
                fit.frequency_mhz,
            ),
        )
    return scan.diagnostics.model_copy(
        update={
            "fitted_center_mhz": _finite(fit.frequency_mhz),
            "center_error_mhz": _finite(fit.error_mhz),
            "fwhm_mhz": _finite(fit.width_mhz),
            "normalized_rms": _finite(fit.quality.normalized_residual_rms),
            "mae_over_fit_ptp": _finite(mae_ratio),
            "gates": gates,
        }
    )


def confirmation_gates(
    coarse_fit: FrequencyFit, scan: FrequencyScan
) -> tuple[Gate, ...]:
    """Record fine-peak presence and displacement versus the coarse peak width.

    The strongest fine peak is used, as in confirm_frequency_candidate. No
    candidate is selected or accepted here. Width and displacement use MHz.
    """
    peak = coarse_fit.selected_peak
    if peak is None:
        raise ValueError("Fine confirmation requires a coarse candidate")
    strongest = next(iter(scan.peaks), None)
    return (
        gate("fine_candidate_present", "==", 1, float(strongest is not None)),
        gate(
            "fine_peak_distance_mhz",
            "<=",
            peak.width,
            None
            if strongest is None
            else abs(strongest.center - coarse_fit.frequency_mhz),
        ),
    )
