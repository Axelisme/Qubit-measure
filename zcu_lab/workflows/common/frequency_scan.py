"""Per-attempt spectroscopy acquisition, fit gates and linewidth observation."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from zcu_tools.analysis.fitting import detect_peaks
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed

from zcu_lab.v2.twotone.freq.analysis import FreqAnalyzeOptions
from zcu_lab.v2.twotone.freq.core import (
    FreqCfg,
    FreqExp,
    FreqResult,
    qubfreq_signal2real,
)
from zcu_lab.workflows.common.calibration_analysis import (
    FrequencyFit,
    analyze_qubit_freq,
)
from zcu_lab.workflows.common.calibration_results import (
    FrequencyCandidateAttempt,
    FrequencyScan,
)
from zcu_lab.workflows.common.frequency_tracking import FrequencyTracker, trusted_curve
from zcu_lab.workflows.common.saving import save_native

if TYPE_CHECKING:
    from zcu_lab.workflows.common.calibration import (
        CalibrationInputs,
        CalibrationTunables,
    )


def accept_frequency_fit(
    fit: FrequencyFit, result: FreqResult, tun: CalibrationTunables
) -> bool:
    """Return whether a fitted center passes this invocation\'s frequency gates.

    result is the full acquired MHz window. fit contains local same-coordinate
    observations/model values and MHz estimates/errors. tun supplies edge margin,
    relative center error and residual bounds. Do not mutate fit or result.
    """
    margin = tun.frequency_edge_margin * float(np.ptp(result.freqs))
    residual = fit.quality.normalized_residual_rms
    return (
        math.isfinite(fit.frequency_mhz)
        and fit.frequency_mhz > 0
        and float(np.min(result.freqs)) + margin
        <= fit.frequency_mhz
        <= float(np.max(result.freqs)) - margin
        and math.isfinite(fit.width_mhz)
        and fit.width_mhz > 0
        and math.isfinite(fit.error_mhz)
        and fit.error_mhz >= 0
        and fit.error_mhz / fit.width_mhz < tun.max_frequency_rel_err
        and residual is not None
        and residual < tun.max_fit_residual
        and trusted_curve(fit.observed, fit.fitted, 0.2)
    )


def observe_frequency_linewidth(
    tracker: FrequencyTracker,
    fit: FrequencyFit,
    result: FreqResult,
    gain: float,
    index: int,
    tun: CalibrationTunables,
) -> None:
    """Observe trusted FWHM/gain at this point index; otherwise leave feedback alone.

    result supplies the full scan span in MHz. tun sets the stricter linewidth
    residual bound. A trusted observation requires positive gain; the tracker
    raises ValueError otherwise. Frequency acceptance remains the caller\'s decision.
    """
    if 0 < fit.width_mhz <= float(np.ptp(result.freqs)) and trusted_curve(
        fit.observed, fit.fitted, tun.linewidth_max_mean_residual_ratio
    ):
        tracker.observe_linewidth(index, fit.width_mhz, gain)


def frequency_probe_gains(
    inputs: CalibrationInputs,
    tracker: FrequencyTracker,
) -> tuple[float, float, float]:
    """Return coarse, fine and legacy single gains for the captured tunables.

    Coarse defaults to the pulse gain. Fine defaults to half the adaptive gain,
    capped at half coarse gain. Explicit coarse/fine values override defaults.
    """
    tun = inputs.tun
    initial = float(inputs.probe_pulse.gain)
    adaptive = tracker.probe_gain(initial, tun)
    coarse = tun.coarse_gain if tun.coarse_gain is not None else initial
    fine = tun.fine_gain if tun.fine_gain is not None else 0.5 * min(adaptive, coarse)
    return coarse, fine, adaptive


async def acquire_frequency_scan(
    scope: Scope,
    cfg: FreqCfg,
    *,
    scan: FrequencyScan,
    inputs: CalibrationInputs,
) -> tuple[FrequencyScan, FreqResult | Failed]:
    """Save one Completed scan and detect candidates without selecting or fitting.

    scan.file supplies the intended relative filename; clear it on acquisition
    failure. Partial acquisition returns Failed("stopped") without detection.
    Preserve raw data before analysis. Do not change history or feedback.
    """
    outcome = await scope.run(FreqExp().run, cfg, name="frequency")
    if isinstance(outcome, Failed):
        return replace(scan, file=None), outcome
    if scan.file is None:
        raise ValueError("A scan needs its planned native filename")
    save_native(
        outcome,
        inputs.directory / scan.file,
        spec=FreqExp.AXES_SPEC,
        snapshot=inputs.snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return scan, Failed("stopped")
    result = outcome.value.result
    if not np.all(np.isfinite(result.signals)):
        return scan, Failed("Nonfinite measured IQ")
    observed = qubfreq_signal2real(result.signals)
    tun = inputs.tun
    peaks = tuple(
        detect_peaks(
            result.freqs,
            observed,
            min_prominence=tun.peak_min_prominence,
            min_distance=tun.peak_min_distance_mhz,
            min_snr=tun.peak_min_snr,
            min_width_samples=tun.peak_min_width_samples,
        )
    )
    return replace(scan, peaks=peaks), result


def fit_frequency_scan(
    result: FreqResult | Failed,
    scan: FrequencyScan,
    tun: CalibrationTunables,
    expected_center_mhz: float | None,
) -> tuple[FrequencyScan, FrequencyFit | Failed]:
    """Fit the nearest expected or strongest candidate using this scan's peaks.

    None selects the strongest candidate. two_step rejects truncated targets
    and coarse fits outside their isolated interval. single fits the full window.
    No acquisition, saving or feedback occurs; retain candidates on failure.
    """
    if isinstance(result, Failed):
        return scan, result
    peaks = scan.peaks
    selected = (
        min(peaks, key=lambda peak: abs(peak.center - expected_center_mhz))
        if peaks and expected_center_mhz is not None
        else next(iter(peaks), None)
    )
    scan = replace(scan, peaks=peaks, selected_peak=selected)
    if scan.stage != "single" and selected is not None and selected.truncated:
        return scan, Failed("Selected peak is truncated")
    fit = analyze_qubit_freq(
        result,
        FreqAnalyzeOptions(
            expected_center_mhz=expected_center_mhz,
            min_prominence=tun.peak_min_prominence,
            min_distance_mhz=tun.peak_min_distance_mhz,
            min_snr=tun.peak_min_snr,
            min_width_samples=tun.peak_min_width_samples,
        ),
        peaks=peaks,
        isolate_peaks=scan.stage != "single",
    )
    if not isinstance(fit, Failed):
        scan = replace(scan, fitted_center_mhz=fit.frequency_mhz)
        if scan.stage == "coarse" and not float(
            np.min(fit.freqs)
        ) <= fit.frequency_mhz <= float(np.max(fit.freqs)):
            return scan, Failed(
                "Coarse fitted center is outside its candidate interval"
            )
    return scan, fit


def confirm_frequency_candidate(
    coarse_fit: FrequencyFit,
    coarse_file: str | None,
    scan: FrequencyScan,
    fit: FrequencyFit | Failed,
    tun: CalibrationTunables,
    acquisition: FreqResult | Failed,
) -> FrequencyCandidateAttempt:
    """Record fine confirmation against the coarse target's center and width.

    The strongest fine peak must lie within one coarse candidate width of its
    fitted center, then pass the existing frequency gates. Failed acquisitions
    and partials retain their failure reason; no fine file is invented.
    """
    peak = coarse_fit.selected_peak
    if peak is None:
        raise ValueError("Fine confirmation requires a coarse candidate")
    strongest = next(iter(scan.peaks), None)
    if isinstance(acquisition, Failed):
        reason = acquisition.reason
    elif strongest is None:
        reason = "False coarse peak: no fine candidates"
    elif abs(strongest.center - coarse_fit.frequency_mhz) > peak.width:
        reason = "False coarse peak: fine peak outside coarse width"
    elif isinstance(fit, Failed):
        reason = fit.reason
    elif not accept_frequency_fit(fit, fit.source, tun):
        reason = "Fine frequency fit failed quality gates"
    else:
        reason = None
    return FrequencyCandidateAttempt(
        coarse_file, peak, coarse_fit.frequency_mhz, scan.file, reason
    )


def fit_coarse_candidates(
    result: FreqResult | Failed,
    scan: FrequencyScan,
    tun: CalibrationTunables,
    limit: int,
) -> tuple[
    FrequencyScan, tuple[FrequencyFit, ...], tuple[FrequencyCandidateAttempt, ...]
]:
    """Locally fit up to limit targets, nearest prediction first, then prominence.

    Reuse scan.peaks without acquiring or redetecting. Return usable fits and
    per-candidate rejection diagnostics. The scan keeps its first target's fit.
    """
    if isinstance(result, Failed) or not scan.peaks or limit <= 0:
        return scan, (), ()
    expected = scan.center_mhz - tun.detune_center_mhz
    nearest = min(scan.peaks, key=lambda peak: abs(peak.center - expected))
    candidates = [nearest, *(peak for peak in scan.peaks if peak is not nearest)]
    fits: list[FrequencyFit] = []
    rejected: list[FrequencyCandidateAttempt] = []
    for index, peak in enumerate(candidates[:limit]):
        candidate_scan, fit = fit_frequency_scan(result, scan, tun, peak.center)
        if index == 0:
            scan = candidate_scan
        if isinstance(fit, Failed):
            rejected.append(
                FrequencyCandidateAttempt(
                    scan.file, peak, candidate_scan.fitted_center_mhz, None, fit.reason
                )
            )
        else:
            fits.append(fit)
    return scan, tuple(fits), tuple(rejected)
