"""Per-point calibration outcomes and serializable feedback diagnostics."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict
from zcu_tools.analysis.fitting import PeakCandidate
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.simulate.fluxonium.physical_fit import FluxoniumModelSnapshot

from zcu_lab.workflows.common.frequency_tracking import RecoveryAttempt
from zcu_lab.workflows.common.gates import Gate

type StageStatus = Literal["accepted", "rejected", "failed", "skipped"]


class FrequencyDiagnostics(BaseModel):
    """Frequency decision inputs, retained even when a fit is rejected.

    Center/error/FWHM use MHz. peak_snr is the selected candidate's height SNR;
    trace_snr uses the full projected scan's Savitzky-Golay SNR estimator.
    normalized_rms uses the fit interval's observed peak-to-peak; MAE uses its
    fitted peak-to-peak. Missing/nonfinite metrics are null, never accepted
    substitutes. gates records evaluated decisions; fit_failure names a failed
    analysis or acquisition. selected_peak_center_mhz links coarse candidates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    selected_peak_center_mhz: float | None = None
    fitted_center_mhz: float | None = None
    center_error_mhz: float | None = None
    fwhm_mhz: float | None = None
    peak_snr: float | None = None
    trace_snr: float | None = None
    normalized_rms: float | None = None
    mae_over_fit_ptp: float | None = None
    gates: tuple[Gate, ...] = ()
    fit_failure: str | None = None


@dataclass(frozen=True)
class FrequencyScan:
    """One coarse/fine/single attempt, including acquisition and fit failures.

    file is the relative native filename after Completed, otherwise None.
    gain/span_mhz/center_mhz describe the actual scan. peaks contains every
    detected candidate, empty for failed/partial acquisition. selected_peak is
    the sampled target; fitted_center_mhz is None when fitting did not finish.
    diagnostics retains its decision metrics. coarse_candidate_diagnostics
    records every locally tried coarse fit, not just the first selected target.
    """

    stage: Literal["coarse", "fine", "single"]
    file: str | None
    gain: float
    span_mhz: float
    center_mhz: float
    peaks: tuple[PeakCandidate, ...] = ()
    selected_peak: PeakCandidate | None = None
    fitted_center_mhz: float | None = None
    diagnostics: FrequencyDiagnostics = field(default_factory=FrequencyDiagnostics)
    coarse_candidate_diagnostics: tuple[FrequencyDiagnostics, ...] = ()


@dataclass(frozen=True)
class FrequencyCandidateAttempt:
    """One coarse candidate and the result of its fine confirmation.

    coarse_file identifies the saved survey; peak is its sampled candidate.
    coarse_center_mhz is the local fitted center, None on coarse fit failure.
    fine_file is the saved confirmation, None if no fine acquisition completed.
    rejection_reason is None only when the fine fit passed all frequency gates.
    """

    coarse_file: str | None
    peak: PeakCandidate
    coarse_center_mhz: float | None
    fine_file: str | None
    rejection_reason: str | None


@dataclass(frozen=True)
class FrequencyResult:
    """One point's spectroscopy outcome and immutable relative native references.

    status is accepted, rejected (fit gates or stopped partial), or failed
    (Schedule or expected fitting failure). Partial reason is stopped.
    reason is None on acceptance, otherwise the diagnostic. frequency_mhz is
    this point's positive fitted MHz only on acceptance, never a prior seed.
    files lists every Completed coarse/fine/single attempt in acquisition order, including
    partial, rejected and failed fits; Schedule failures have no filename.
    References do not guarantee later existence. Unexpected writer/adapter/fit/
    plot exceptions propagate instead. probe_gains records actual spectroscopy
    gain for every acquisition attempt, including failed/partial attempts.
    scan_center_mhz is the physical-plus-IDW center before window offset.
    predictor_model captures the model used to choose this point's center.
    recovery is this point's physical refit attempt, or None; an accepted candidate
    first affects the next point. scans records stage, filename, gain, span and
    all candidates for every coarse/fine/single attempt, including failures.
    candidate_attempts records each tried coarse peak, its local fitted center,
    fine filename and rejection reason, including coarse fit failures. None
    rejection_reason means its fine confirmation passed all frequency gates.
    These diagnostics contain no live predictor.
    """

    status: StageStatus
    reason: str | None
    frequency_mhz: float | None
    files: tuple[str, ...]
    probe_gains: tuple[float, ...] = ()
    scan_center_mhz: float | None = None
    predictor_model: FluxoniumModelSnapshot | None = None
    recovery: RecoveryAttempt | None = None
    scans: tuple[FrequencyScan, ...] = ()
    candidate_attempts: tuple[FrequencyCandidateAttempt, ...] = ()


@dataclass(frozen=True)
class AmpCalibrationResult:
    """Fixed-length gain calibration, including rejected fit diagnostics.

    status/reason follow StageStatus. length_us is the configured fixed pulse length.
    pi_gain/error retain fitted diagnostics even when rejected. pi2_gain is
    the trusted pi/2 gain on acceptance, otherwise None.
    file is amp-rabi.h5 only after Completed, including stopped partial data.
    Missing fits remain None. No diagnostic here certifies a usable pulse.
    """

    status: StageStatus
    reason: str | None
    length_us: float
    file: str | None = None
    pi_gain: float | None = None
    pi_gain_error: float | None = None
    pi2_gain: float | None = None


@dataclass(frozen=True)
class RabiResult:
    """One point's selected Rabi-mode outcome, not a substitute for prior calibration.

    status is accepted/rejected/failed/skipped; reason is None only on acceptance.
    frequency_mhz is the accepted frequency used at this point (MHz), or None
    when frequency was not accepted and Rabi is skipped. It never carries a
    previous point's frequency. Accepted implies a current accepted frequency.
    pi_length_us is the accepted fitted duration. pi2_length_us is the trusted
    phase-aware Len duration or fixed Amp duration >= .03 us, or None when
    pi2 is untrusted. drive_gain is Len's acquisition gain, even on rejection,
    or Amp's accepted fitted pi gain. It is None for skipped or nonaccepted Amp.
    file is the relative native
    filename after Completed, including partial and failed/rejected fits,
    otherwise None. Partial returns rejected/stopped without fitting; expected
    fitting failures return failed/reason. Unexpected errors propagate.
    too_fast_pi_length_us records a quality-trusted rejected fit below .03 us,
    never a usable pulse. Auto mode uses it only to propose a lower gain.
    pi2_drive_gain is Amp fitted pi/2 gain; None retains drive_gain for Len.
    amp stores gain-stage diagnostics/native reference, None when not run.
    len_status records the first-stage status when Amp follows Len. file keeps
    the Len native reference in combined mode, otherwise the only Rabi file.
    too_slow_pi_length_us records a quality-trusted rejected fit above 90% of
    the measured maximum duration; auto mode uses it only to raise gain.
    target_pi_length_us is the resolved setpoint in us. requested_sweep and
    planned_sweep are the original and clock-planned length grids in us.
    Planned is None if acquisition failed before returning a native record;
    all three are None on dependency skip.
    """

    status: StageStatus
    reason: str | None
    frequency_mhz: float | None
    pi_length_us: float | None
    pi2_length_us: float | None
    file: str | None
    drive_gain: float | None = None
    too_fast_pi_length_us: float | None = None
    pi2_drive_gain: float | None = None
    amp: AmpCalibrationResult | None = None
    len_status: StageStatus | None = None
    too_slow_pi_length_us: float | None = None
    target_pi_length_us: float | None = None
    requested_sweep: SweepCfg | None = None
    planned_sweep: SweepCfg | None = None


@dataclass(frozen=True)
class ReadoutResult:
    """Readout scan outcome and immutable relative native reference.

    status is accepted/rejected/failed/skipped (dependency or flux cadence).
    reason is None only on acceptance. frequency_mhz/gain are this point's
    accepted readout MHz/native gain only; nonaccepted results have no values.
    file names the Completed landscape, including rejection, otherwise None.
    Nonacceptance leaves the legal seed/prior readout carry unchanged.
    Partial returns rejected/stopped after saving, without SNR selection.
    Unexpected analysis/configuration/writer errors propagate.
    """

    status: StageStatus
    reason: str | None
    frequency_mhz: float | None
    gain: float | None
    file: str | None
