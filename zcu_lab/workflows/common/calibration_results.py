"""Per-point calibration outcomes and serializable feedback diagnostics."""

from dataclasses import dataclass
from typing import Literal

from zcu_tools.analysis.fitting import PeakCandidate
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.simulate.fluxonium.physical_fit import FluxoniumModelSnapshot

from zcu_lab.workflows.common.frequency_tracking import RecoveryAttempt

type StageStatus = Literal["accepted", "rejected", "failed", "skipped"]


@dataclass(frozen=True)
class FrequencyScan:
    """One coarse/fine/single attempt, including acquisition and fit failures.

    file is the relative native filename after Completed, otherwise None.
    gain/span_mhz/center_mhz describe the actual scan. peaks contains every
    detected candidate, empty for failed/partial acquisition. selected_peak is
    the sampled target; fitted_center_mhz is None when fitting did not finish.
    """

    stage: Literal["coarse", "fine", "single"]
    file: str | None
    gain: float
    span_mhz: float
    center_mhz: float
    peaks: tuple[PeakCandidate, ...] = ()
    selected_peak: PeakCandidate | None = None
    fitted_center_mhz: float | None = None


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


@dataclass(frozen=True)
class RabiResult:
    """One point's Length-Rabi outcome, not a substitute for prior calibration.

    status is accepted/rejected/failed/skipped; reason is None only on acceptance.
    frequency_mhz is the accepted frequency used at this point (MHz), or None
    when frequency was not accepted and Rabi is skipped. It never carries a
    previous point's frequency. Accepted implies a current accepted frequency.
    pi_length_us is the accepted fitted duration. pi2_length_us is the trusted
    phase-aware duration >= .03 us, or None when pi2 is untrusted. drive_gain
    records the actual acquisition gain even on failure/rejection, or None on
    dependency skip. file is the relative native
    filename after Completed, including partial and failed/rejected fits,
    otherwise None. Partial returns rejected/stopped without fitting; expected
    fitting failures return failed/reason. Unexpected errors propagate.
    too_fast_pi_length_us records a quality-trusted rejected fit below .03 us,
    never a usable pulse. Auto mode uses it only to propose a lower gain.
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
