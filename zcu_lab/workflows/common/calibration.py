"""common calibration helpers for lab workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg, ReadoutCfg

from zcu_lab.v2.twotone.freq.core import FreqCfg, FreqModuleCfg, FreqSweepCfg
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiCfg,
    LenRabiExp,
    LenRabiModuleCfg,
    LenRabiSweepCfg,
)
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import (
    FreqGainCfg,
    FreqGainExp,
    FreqGainModuleCfg,
    FreqGainSweepCfg,
)
from zcu_lab.workflows.common.acquire import run_qubit_freq
from zcu_lab.workflows.common.analysis import (
    analyze_len_rabi,
    analyze_qubit_freq,
)
from zcu_lab.workflows.common.saving import save_frequency, save_rabi, save_readout


class CalibrationTunables(BaseModel):
    """Generation and acceptance knobs captured once per flux-point invocation.

    qf_widen specifies the finite sequence of spectroscopy spans, not retries
    after adapter errors. Error and residual upper bounds are strict; readout
    SNR uses an inclusive lower bound. ro_every counts flux points.
    Acceptance/cadence/range defaults are provisional simulator settings, not
    hardware-validated physics defaults; users confirm them at CP-B/CP-C.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    qf_reps: int = Field(default=100, ge=1, description="Spectroscopy averages")
    qf_rounds: int = Field(default=1, ge=1, description="Spectroscopy rounds")
    rabi_reps: int = Field(default=100, ge=1, description="Rabi averages")
    rabi_rounds: int = Field(default=1, ge=1, description="Rabi rounds")
    ro_reps: int = Field(default=100, ge=1, description="Readout averages")
    ro_rounds: int = Field(default=1, ge=1, description="Readout rounds")
    span_mhz: float = Field(default=30.0, gt=0, description="Base spectroscopy span")
    qf_widen: tuple[Annotated[float, Field(gt=0)], ...] = Field(
        default=(1.0, 2.0, 4.0),
        min_length=1,
        description="Ordered spectroscopy span multipliers",
    )
    max_frequency_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict center-error/FWHM upper bound"
    )
    max_rabi_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict pi-error/pi-duration upper bound"
    )
    max_fit_residual: float = Field(
        default=0.2,
        gt=0,
        le=1,
        description="Strict normalized residual RMS upper bound",
    )
    min_readout_snr: float = Field(
        default=5.0, gt=0, description="Inclusive minimum readout SNR"
    )
    ro_every: int = Field(
        default=5, ge=1, description="Flux-point readout scan cadence"
    )
    rabi_start_factor: float = Field(
        default=0.1, gt=0, description="Sweep start divided by prior pi duration"
    )
    rabi_stop_factor: float = Field(
        default=3.0, gt=0, description="Sweep stop divided by prior pi duration"
    )

    @model_validator(mode="after")
    def _validate_rabi_range(self) -> CalibrationTunables:
        if self.rabi_stop_factor <= self.rabi_start_factor:
            raise ValueError("Rabi stop factor must exceed start factor")
        return self


@dataclass(frozen=True)
class Calibrated:
    """Accepted scalar calibration, also usable as deeply read-only host seed.

    f01_mhz/readout_freq_mhz are positive finite frequencies in MHz.
    pi_length_us is a positive finite pi-pulse duration at the template gain.
    readout_gain is a finite nonnegative native readout gain.
    Invalid values raise ValueError at construction.
    """

    f01_mhz: float
    pi_length_us: float
    readout_freq_mhz: float
    readout_gain: float

    def __post_init__(self) -> None:
        for name, value in (
            ("f01_mhz", self.f01_mhz),
            ("pi_length_us", self.pi_length_us),
            ("readout_freq_mhz", self.readout_freq_mhz),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.readout_gain) or self.readout_gain < 0:
            raise ValueError("readout_gain must be nonnegative and finite")


@dataclass(frozen=True)
class CalibratedPi2(Calibrated):
    """Successful calibration with a fitted pi/2 duration for Echo and Ramsey.

    The original four Calibrated scalars retain their meanings and validation.
    pi2_length_us is a positive finite, phase-aware Length-Rabi fit duration in
    microseconds at the selected template's gain and waveform. It is neither
    half pi_length_us nor an initial seed; callers obtain it from accepted Rabi.
    Invalid original scalars or pi2 duration raise ValueError at construction.
    """

    pi2_length_us: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.pi2_length_us) or self.pi2_length_us <= 0:
            raise ValueError("pi2_length_us must be positive and finite")


@dataclass
class CalCarry:
    """Coroutine-local calibration carry retained between committed points.

    seed is the legal host scalar seed; last contains the latest accepted
    scalar parts or None initially. Frequency, Rabi and readout update only
    their own fields. f01_by_flux contains accepted (device setpoint, MHz)
    frequency history even when a later stage fails. Pause preserves this carry;
    only explicit point commits persist reports. There are no stage checkpoints.
    """

    seed: Calibrated
    last: Calibrated | None = None
    f01_by_flux: list[tuple[float, float]] = field(default_factory=list)


type StageStatus = Literal["accepted", "rejected", "failed", "skipped"]


@dataclass(frozen=True)
class CalibrationInputs:
    """Fixed native routing and captured policy shared by the three experiments.

    qub_pulse/readout are resolved shared drive/pulse-backed readout modules.
    tun is this invocation's ordinary generation/acceptance/cadence policy.
    The wrapper is frozen; nested modules are copied before applying scalars.
    snapshot is the fixed historical entry/point/role state used by all three
    native saves, not a live source. directory is the allocated commit directory.
    Coherence and MIST reuse fixed modules/snapshot/directory across stages,
    replacing tun from tun.current.cal before each stage. A stage does not
    recapture sources or refresh its own policy during acquisition.
    """

    qub_pulse: PulseCfg
    readout: PulseReadoutCfg
    tun: CalibrationTunables
    snapshot: RunSnapshot
    directory: Path
    point_index: int


@dataclass(frozen=True)
class FrequencyResult:
    """One point's spectroscopy outcome and immutable relative native references.

    status is accepted, rejected (fit gates or stopped partial), or failed
    (Schedule or expected fitting failure). Partial reason is stopped.
    reason is None on acceptance, otherwise the diagnostic. frequency_mhz is
    this point's positive fitted MHz only on acceptance, never a prior seed.
    files lists every Completed widening attempt in acquisition order, including
    partial, rejected and failed fits; Schedule failures have no filename.
    References do not guarantee later existence. Unexpected writer/adapter/fit/
    plot exceptions propagate instead.
    """

    status: StageStatus
    reason: str | None
    frequency_mhz: float | None
    files: tuple[str, ...]


@dataclass(frozen=True)
class RabiResult:
    """One point's Length-Rabi outcome, not a substitute for prior calibration.

    status is accepted/rejected/failed/skipped; reason is None only on acceptance.
    frequency_mhz is the accepted frequency used at this point (MHz), or None
    when frequency was not accepted and Rabi is skipped. It never carries a
    previous point's frequency. Accepted implies a current accepted frequency.
    pi_length_us/pi2_length_us are positive fitted durations only on acceptance;
    pi2 comes from the phase-aware fit, not pi/2. file is the relative native
    filename after Completed, including partial and failed/rejected fits,
    otherwise None. Partial returns rejected/stopped without fitting; expected
    fitting failures return failed/reason. Unexpected errors propagate.
    """

    status: StageStatus
    reason: str | None
    frequency_mhz: float | None
    pi_length_us: float | None
    pi2_length_us: float | None
    file: str | None


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


def apply_readout_calibration(readout: ReadoutCfg, cal: Calibrated) -> None:
    """Apply accepted readout scalars to a caller-owned native cfg in place.

    Updates the shared generator/ADC frequency in MHz and native readout gain.
    Callers must copy any template they need to retain before calling.
    Native set_param errors propagate. No acquisition or saving occurs.
    """
    readout.set_param("freq", cal.readout_freq_mhz)
    readout.set_param("gain", cal.readout_gain)


def _validate_calibration_inputs(flux: float, t1_estimate_us: float) -> None:
    """Reject nonfinite flux and nonpositive/nonfinite T1 estimates in µs."""
    if not math.isfinite(t1_estimate_us) or t1_estimate_us <= 0:
        raise ValueError("T1 estimate must be positive and finite")
    if not math.isfinite(flux):
        raise ValueError("Calibration flux must be finite")


async def calibrate_frequency(
    scope: Scope,
    *,
    plots: Plots,
    inputs: CalibrationInputs,
    flux: float,
    carry: CalCarry,
) -> FrequencyResult:
    """Locate the qubit resonance so this flux point can use a resonant drive.

    Await in a workflow. flux is a finite device setpoint. carry is
    coroutine-local; only frequency/history change. inputs supplies the
    resolved modules, this invocation's policy and the fixed historical snapshot
    used by the native saver after each Completed attempt,
    before fitting. Widening uses inputs.tun.qf_widen; the center follows an
    EWMA(alpha=.5) and relax timing is 1 us. Acceptance uses the captured gates.
    Partial attempts save then return rejected/stopped without fitting. Widening
    continues as usual. Expected fit failures return failed/reason, with all
    Completed refs retained; neither failure nor partial updates carry/history.
    Invalid flux raises before acquisition; all unexpected errors propagate.
    """
    # Check the setpoint before acquiring and project the frequency history.
    if not math.isfinite(flux):
        raise ValueError("Calibration flux must be finite")
    prior = carry.seed if carry.last is None else carry.last
    hint = prior.f01_mhz
    for _, frequency in carry.f01_by_flux:
        hint = 0.5 * frequency + 0.5 * hint
    filenames: list[str] = []
    status: StageStatus = "rejected"
    reason = "No accepted frequency fit"
    for attempt, widen in enumerate(inputs.tun.qf_widen, start=1):
        if scope.cancelled:
            return FrequencyResult("skipped", "Point cancelled", None, tuple(filenames))
        # Rebuild the spectroscopy plot with expected and previous frequencies.
        name = "calibration/qubit_freq"
        if name not in plots:
            plots.subplots(name)

        def rebuild(figure: Figure) -> None:
            figure.clear()
            ax = figure.subplots()
            ax.axvline(hint, linestyle="--", label="expected")
            ax.axvline(prior.f01_mhz, linestyle=":", label="previous")

        plots.draw(name, rebuild)
        plot = plots.liveplot_1d(
            name, "Frequency (MHz)", "Signal", num_lines=2, axes=plots[name].axes[0]
        )

        # Sweep a widened span around the history estimate using retained readout.
        drive = inputs.qub_pulse.model_copy(deep=True)
        ro = inputs.readout.model_copy(deep=True)
        apply_readout_calibration(ro, prior)
        span = inputs.tun.span_mhz * widen
        cfg = FreqCfg(
            reps=inputs.tun.qf_reps,
            rounds=inputs.tun.qf_rounds,
            relax_delay=1.0,
            modules=FreqModuleCfg(qub_pulse=drive, readout=ro),
            sweep=FreqSweepCfg(
                freq=SweepCfg(
                    start=hint - span / 2,
                    stop=hint + span / 2,
                    expts=101,
                    step=span / 100,
                )
            ),
        )
        outcome = await scope.run(run_qubit_freq, cfg, name="frequency")
        if isinstance(outcome, Failed):
            status, reason = "failed", outcome.reason
            continue
        # Save every completed attempt, then fit and accept its resonance.
        filename = f"frequency-{attempt:02d}.h5"
        save_frequency(
            outcome,
            inputs.directory / filename,
            snapshot=inputs.snapshot,
            identity=scope.identity,
        )
        filenames.append(filename)
        if outcome.partial_result:
            status, reason = "rejected", "stopped"
            continue
        fit = analyze_qubit_freq(outcome.value.result)
        if isinstance(fit, Failed):
            status, reason = "failed", fit.reason
            continue
        residual = fit.quality.normalized_residual_rms
        if (
            math.isfinite(fit.frequency_mhz)
            and fit.frequency_mhz > 0
            and math.isfinite(fit.width_mhz)
            and fit.width_mhz > 0
            and math.isfinite(fit.error_mhz)
            and fit.error_mhz >= 0
            and fit.error_mhz / fit.width_mhz < inputs.tun.max_frequency_rel_err
            and residual is not None
            and residual < inputs.tun.max_fit_residual
        ):
            plot.update(
                outcome.value.result.freqs, np.stack((fit.observed, fit.fitted))
            )
            carry.last = replace(prior, f01_mhz=fit.frequency_mhz)
            carry.f01_by_flux.append((flux, fit.frequency_mhz))
            return FrequencyResult(
                "accepted", None, fit.frequency_mhz, tuple(filenames)
            )
        status, reason = "rejected", "No accepted frequency fit"
    return FrequencyResult(status, reason, None, tuple(filenames))


async def calibrate_rabi(
    scope: Scope,
    *,
    plots: Plots,
    inputs: CalibrationInputs,
    frequency: FrequencyResult,
    carry: CalCarry,
    t1_estimate_us: float,
) -> RabiResult:
    """Measure Rabi oscillations to calibrate pi and pi/2 pulse durations here.

    frequency must be the current point's result, never a prior frequency.
    Missing acceptance returns a named skipped result. t1_estimate_us is positive
    finite us; relax_delay remains five times it. inputs supplies captured
    modules/policy. carry retains readout and frequency; only fitted pi/pi2 change
    on acceptance. inputs.snapshot is the fixed historical entry/point state
    used to save rabi.h5 after Completed, including rejected fits. point_index
    must be a nonnegative integer (not bool), the zero-based flux index; invalid
    indexes raise ValueError at entry. The title shows its one-based number. Lengths
    stay in microseconds and the projected signal has arbitrary units.
    Partial saves then returns rejected/stopped without fitting or carry updates.
    Schedule and expected fit failures are named failed; unexpected errors
    propagate. Invalid timing raises before any acquisition. Await in a workflow.
    """
    # Require a valid plot index and this point's accepted frequency.
    if type(inputs.point_index) is not int or inputs.point_index < 0:
        raise ValueError("Rabi point index must be a nonnegative integer")
    _validate_calibration_inputs(0.0, t1_estimate_us)
    if frequency.status != "accepted":
        return RabiResult(
            "skipped",
            "Requires this point's accepted frequency",
            None,
            None,
            None,
            None,
        )
    if frequency.frequency_mhz is None:
        raise ValueError("Accepted frequency result requires MHz")
    f01 = frequency.frequency_mhz
    prior = carry.seed if carry.last is None else carry.last
    # Sweep duration at the accepted frequency, retaining legal readout.
    drive = inputs.qub_pulse.model_copy(deep=True)
    drive.set_param("freq", f01)
    ro = inputs.readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
    start = inputs.tun.rabi_start_factor * prior.pi_length_us
    stop = inputs.tun.rabi_stop_factor * prior.pi_length_us
    cfg = LenRabiCfg(
        reps=inputs.tun.rabi_reps,
        rounds=inputs.tun.rabi_rounds,
        relax_delay=5 * t1_estimate_us,
        modules=LenRabiModuleCfg(qub_pulse=drive, readout=ro),
        sweep=LenRabiSweepCfg(
            length=SweepCfg(start=start, stop=stop, expts=51, step=(stop - start) / 50)
        ),
    )
    outcome = await scope.run(LenRabiExp().run, cfg, name="Rabi")
    if isinstance(outcome, Failed):
        return RabiResult("failed", outcome.reason, f01, None, None, None)
    # Save before fitting, so rejected fits retain their native trace.
    save_rabi(
        outcome,
        inputs.directory / "rabi.h5",
        snapshot=inputs.snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return RabiResult("rejected", "stopped", f01, None, None, "rabi.h5")
    fit = analyze_len_rabi(outcome.value.result)
    if isinstance(fit, Failed):
        return RabiResult("failed", fit.reason, f01, None, None, "rabi.h5")
    residual = fit.quality.normalized_residual_rms
    if not (
        math.isfinite(fit.pi_length_us)
        and fit.pi_length_us > 0
        and math.isfinite(fit.error_us)
        and fit.error_us >= 0
        and fit.error_us / fit.pi_length_us < inputs.tun.max_rabi_rel_err
        and residual is not None
        and residual < inputs.tun.max_fit_residual
    ):
        return RabiResult("rejected", "Rabi fit rejected", f01, None, None, "rabi.h5")
    # Update only accepted pulse durations and show the fitted oscillation.
    carry.last = CalibratedPi2(
        prior.f01_mhz,
        fit.pi_length_us,
        prior.readout_freq_mhz,
        prior.readout_gain,
        fit.pi2_length_us,
    )
    name = "calibration/len_rabi"
    if name not in plots:
        plots.subplots(name)

    def draw(figure: Figure) -> None:
        figure.clear()
        ax = figure.subplots()
        ax.plot(outcome.value.result.lengths, fit.observed, label="observed")
        ax.plot(outcome.value.result.lengths, fit.fitted, label="fit")
        ax.set(
            xlabel="Pulse length (µs)",
            ylabel="Signal (a.u.)",
            title=f"Length Rabi — flux point {inputs.point_index + 1}",
        )

    plots.draw(name, draw)
    return RabiResult(
        "accepted", None, f01, fit.pi_length_us, fit.pi2_length_us, "rabi.h5"
    )


def readout_due(point_index: int, tun: CalibrationTunables) -> bool:
    """Return whether this zero-based flux point scans readout.

    The rule is point_index % tun.ro_every == 0; index zero naturally scans.
    Failed/rejected prior scans do not change the cadence or cause extra scans.
    The caller retains legal seed/prior readout on cadence skips and checks this
    point's Rabi dependency separately. Negative indexes raise ValueError.
    """
    if point_index < 0:
        raise ValueError("Readout point index must be nonnegative")
    return point_index % tun.ro_every == 0


async def calibrate_readout(
    scope: Scope,
    *,
    inputs: CalibrationInputs,
    rabi: RabiResult,
    carry: CalCarry,
    t1_estimate_us: float,
) -> ReadoutResult:
    """Scan readout frequency and gain to maximize discrimination after Rabi.

    An accepted rabi implies a current accepted frequency; its frequency_mhz
    is the MHz used here, never a prior frequency. Nonaccepted Rabi returns a
    named dependency skip. Caller uses readout_due to decide flux cadence;
    an accepted Rabi causes a scan whenever this stage is called. Timing is
    positive finite us, with relax_delay five times it. carry's readout changes
    only on accepted finite SNR. Failure/rejection retains legal prior
    readout, without returning fabricated accepted values. inputs.snapshot is
    the fixed historical entry/point state used to save readout.h5 after Completed,
    including rejection. Partial saves then returns rejected/stopped without
    selecting a best point or updating carry. Invalid input/unexpected errors raise.
    Await in a workflow; cancellation remains Engine-owned.
    """
    # Require accepted frequency and pulse durations at this point.
    _validate_calibration_inputs(0.0, t1_estimate_us)
    if rabi.status != "accepted":
        return ReadoutResult(
            "skipped",
            "Requires this point's accepted frequency and Rabi",
            None,
            None,
            None,
        )
    if rabi.frequency_mhz is None or rabi.pi_length_us is None:
        raise ValueError("Accepted Rabi requires current frequency and pi duration")
    prior = carry.seed if carry.last is None else carry.last
    # Sweep readout gain and frequency with the calibrated pi drive.
    drive = inputs.qub_pulse.model_copy(deep=True)
    drive.set_param("freq", rabi.frequency_mhz)
    drive.set_param("length", rabi.pi_length_us)
    ro = inputs.readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
    cfg = FreqGainCfg(
        reps=inputs.tun.ro_reps,
        rounds=inputs.tun.ro_rounds,
        relax_delay=5 * t1_estimate_us,
        modules=FreqGainModuleCfg(qub_pulse=drive, readout=ro),
        sweep=FreqGainSweepCfg(
            freq=SweepCfg(
                start=prior.readout_freq_mhz - 5.0,
                stop=prior.readout_freq_mhz + 5.0,
                expts=51,
                step=10.0 / 50,
            ),
            gain=SweepCfg(start=0.0, stop=0.2, expts=31, step=0.2 / 30),
        ),
    )
    outcome = await scope.run(FreqGainExp().run, cfg, name="readout")
    if isinstance(outcome, Failed):
        return ReadoutResult("failed", outcome.reason, None, None, None)
    # Save the landscape and choose its best point meeting the SNR threshold.
    save_readout(
        outcome,
        inputs.directory / "readout.h5",
        snapshot=inputs.snapshot,
        identity=scope.identity,
    )
    if outcome.partial_result:
        return ReadoutResult("rejected", "stopped", None, None, "readout.h5")
    signals = outcome.value.result.signals
    accepted = np.isfinite(signals) & (signals >= inputs.tun.min_readout_snr)
    if not np.any(accepted):
        return ReadoutResult(
            "rejected", "No readout SNR meets threshold", None, None, "readout.h5"
        )
    row, column = np.unravel_index(
        np.argmax(np.where(accepted, signals, -np.inf)), signals.shape
    )
    frequency = float(outcome.value.result.freqs[row])
    gain = float(outcome.value.result.gains[column])
    carry.last = replace(prior, readout_freq_mhz=frequency, readout_gain=gain)
    return ReadoutResult("accepted", None, frequency, gain, "readout.h5")


def current_point_calibration(
    rabi: RabiResult, carry: CalCarry
) -> CalibratedPi2 | None:
    """Combine current accepted Rabi with retained legal readout for acquisition.

    Return None for failed/rejected/skipped Rabi, never borrowing prior pi or
    frequency. Accepted Rabi must contain positive fitted pi/pi2 and current
    MHz; invalid accepted records raise. Readout comes from accepted carry or
    the legal host seed. No I/O or carry mutation occurs.
    """
    if rabi.status != "accepted":
        return None
    if (
        rabi.frequency_mhz is None
        or rabi.pi_length_us is None
        or rabi.pi2_length_us is None
    ):
        raise ValueError("Accepted Rabi requires current frequency, pi and pi2")
    prior = carry.seed if carry.last is None else carry.last
    return CalibratedPi2(
        rabi.frequency_mhz,
        rabi.pi_length_us,
        prior.readout_freq_mhz,
        prior.readout_gain,
        rabi.pi2_length_us,
    )
