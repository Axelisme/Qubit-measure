"""Acquire, retain and analyze one spectroscopy attempt, without feedback."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from zcu_tools.analysis.fitting import detect_peaks
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed

from zcu_lab.v2.twotone.freq.analysis import FreqAnalyzeOptions
from zcu_lab.v2.twotone.freq.core import FreqCfg, FreqExp, qubfreq_signal2real
from zcu_lab.workflows.common.calibration_analysis import (
    FrequencyFit,
    analyze_qubit_freq,
)
from zcu_lab.workflows.common.calibration_results import FrequencyScan
from zcu_lab.workflows.common.frequency_tracking import FrequencyTracker
from zcu_lab.workflows.common.saving import save_native

if TYPE_CHECKING:
    from zcu_lab.workflows.common.calibration import CalibrationInputs


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


async def measure_frequency_scan(
    scope: Scope,
    cfg: FreqCfg,
    scan: FrequencyScan,
    inputs: CalibrationInputs,
) -> tuple[FrequencyScan, FrequencyFit | Failed]:
    """Save one Completed scan before detecting candidates and fitting the target.

    scan.file supplies the intended relative filename; it is cleared on
    acquisition failure. Partial acquisition returns Failed("stopped") without
    detection. A truncated target returns Failed("Selected peak is truncated")
    in two-step mode, leaving its candidates in the scan for widening decisions.
    No history, predictor or linewidth feedback changes here.
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
        )
    )
    expected = scan.center_mhz - (
        0.0 if scan.stage == "fine" else tun.detune_center_mhz
    )
    selected = (
        min(peaks, key=lambda peak: abs(peak.center - expected)) if peaks else None
    )
    scan = replace(scan, peaks=peaks, selected_peak=selected)
    if scan.stage != "single" and selected is not None and selected.truncated:
        return scan, Failed("Selected peak is truncated")
    fit = analyze_qubit_freq(
        result,
        FreqAnalyzeOptions(
            expected_center_mhz=expected,
            min_prominence=tun.peak_min_prominence,
            min_distance_mhz=tun.peak_min_distance_mhz,
        ),
        peaks=peaks,
        isolate_peaks=scan.stage != "single",
    )
    if not isinstance(fit, Failed):
        scan = replace(scan, fitted_center_mhz=fit.frequency_mhz)
    return scan, fit
