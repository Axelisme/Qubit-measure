"""Three-stage native calibration, with acceptance owned by the workflow.

The Engine owns execution, cancellation and publication. Callers supply the
three exact-path savers explicitly; this module never selects a storage format.
Templates are copied before applying calibration and generation tunables.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment.workflows import (
    Completed,
    Effect,
    Failed,
    Live1D,
    WorkflowEnv,
)
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg, ReadoutCfg

from zcu_lab.v2.twotone.freq.core import FreqCfg, FreqModuleCfg, FreqSweepCfg
from zcu_lab.v2.twotone.rabi.len_rabi.core import (
    LenRabiCfg,
    LenRabiModuleCfg,
    LenRabiSweepCfg,
)
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import (
    FreqGainCfg,
    FreqGainModuleCfg,
    FreqGainSweepCfg,
)
from zcu_lab.workflows.acquire import (
    FrequencyAcquisition,
    RabiAcquisition,
    ReadoutAcquisition,
    run_len_rabi,
    run_qubit_freq,
    run_ro_landscape,
)
from zcu_lab.workflows.analysis import analyze_len_rabi, analyze_qubit_freq


class CalibrationTunables(BaseModel):
    """Generation and acceptance knobs captured once per calibration call.

    qf_widen specifies the finite sequence of spectroscopy spans, not retries
    after adapter errors. Error and residual upper bounds are strict; readout
    SNR uses an inclusive lower bound. ro_every counts successful calibrations.
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
        default=5, ge=1, description="Successful readout scan cadence"
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
    readout_gain is a finite nonnegative native pulse gain, applied only to
    PulseReadoutCfg. DirectReadoutCfg has no pulse gain to update.
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
class CalibrationFailed:
    """Rejected calibration stage and human-readable reason; no partial values.

    stage is frequency, rabi or readout. reason describes the Schedule failure
    or acceptance rejection. Exceptions are not converted into this value.
    """

    stage: Literal["frequency", "rabi", "readout"]
    reason: str


@dataclass
class CalCarry:
    """Committed calibration history, changed only after all stages succeed.

    seed is the initial scalar calibration. last is the latest accepted value
    or None. f01_by_flux holds ordered (native flux setpoint, MHz) pairs.
    n_calibrated counts successes for readout cadence. The Engine copies this
    state per step; cancellation discards the in-flight copy.
    """

    seed: Calibrated
    last: Calibrated | None = None
    f01_by_flux: list[tuple[float, float]] = field(default_factory=list)
    n_calibrated: int = 0


type FrequencySaver = Callable[[Completed[FreqCfg, FrequencyAcquisition], Path], None]
type RabiSaver = Callable[[Completed[LenRabiCfg, RabiAcquisition], Path], None]
type ReadoutSaver = Callable[[Completed[FreqGainCfg, ReadoutAcquisition], Path], None]


@dataclass(frozen=True)
class CalibrationSavers:
    """Three exact-path writers, grouped to keep calibration's inputs named.

    frequency/rabi/readout each write and close the path for their concrete
    Completed cfg/result pair. The writers choose their own format and retain
    acquisition evidence. No dispatch, fallback or storage state is added.
    """

    frequency: FrequencySaver
    rabi: RabiSaver
    readout: ReadoutSaver


def apply_readout_calibration(readout: ReadoutCfg, cal: Calibrated) -> None:
    """Apply accepted readout scalars to a caller-owned native cfg in place.

    PulseReadoutCfg updates pulse/ADC frequency in MHz and native pulse gain.
    DirectReadoutCfg updates only ADC frequency; it has no pulse gain.
    Callers must copy any template they need to retain before calling.
    Native set_param errors propagate. No acquisition or saving occurs.
    """
    if isinstance(readout, PulseReadoutCfg):
        readout.set_param("freq", cal.readout_freq_mhz)
        readout.set_param("gain", cal.readout_gain)
    else:
        readout.set_param("ro_freq", cal.readout_freq_mhz)


@dataclass(frozen=True)
class _CalibrationDefaults:
    """Captured native acquisition defaults supplied by the adapter owner.

    initial_delay/relax_delay use microseconds. dev is the adapter-owned native
    device mapping or None, not a flux override. No fallback values are stored.
    """

    initial_delay: float
    relax_delay: float
    dev: Mapping[str, DeviceInfo] | None


def _calibration_defaults(
    stage: Literal["frequency", "rabi", "readout"],
) -> _CalibrationDefaults:
    """Use one acquisition-default seam; await accepted role-path adapters."""
    raise NotImplementedError(f"Captured {stage} adapter defaults are not available")


def _frequency_cfg(
    qub_pulse: PulseCfg,
    readout: PulseReadoutCfg,
    tun: CalibrationTunables,
    prior: Calibrated,
    widen: float,
    hint_mhz: float,
) -> FreqCfg:
    defaults = _calibration_defaults("frequency")
    drive = qub_pulse.model_copy(deep=True)
    ro = readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
    span = tun.span_mhz * widen
    return FreqCfg(
        reps=tun.qf_reps,
        rounds=tun.qf_rounds,
        initial_delay=defaults.initial_delay,
        relax_delay=defaults.relax_delay,
        dev=defaults.dev,
        modules=FreqModuleCfg(qub_pulse=drive, readout=ro),
        sweep=FreqSweepCfg(
            freq=SweepCfg(
                start=hint_mhz - span / 2,
                stop=hint_mhz + span / 2,
                expts=101,
                step=span / 100,
            )
        ),
    )


def _rabi_cfg(
    qub_pulse: PulseCfg,
    readout: PulseReadoutCfg,
    tun: CalibrationTunables,
    prior: Calibrated,
    f01_mhz: float,
) -> LenRabiCfg:
    defaults = _calibration_defaults("rabi")
    drive = qub_pulse.model_copy(deep=True)
    drive.set_param("freq", f01_mhz)
    ro = readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
    start = tun.rabi_start_factor * prior.pi_length_us
    stop = tun.rabi_stop_factor * prior.pi_length_us
    return LenRabiCfg(
        reps=tun.rabi_reps,
        rounds=tun.rabi_rounds,
        initial_delay=defaults.initial_delay,
        relax_delay=defaults.relax_delay,
        dev=defaults.dev,
        modules=LenRabiModuleCfg(qub_pulse=drive, readout=ro),
        sweep=LenRabiSweepCfg(
            length=SweepCfg(
                start=start,
                stop=stop,
                expts=51,
                step=(stop - start) / 50,
            )
        ),
    )


def _readout_cfg(
    qub_pulse: PulseCfg,
    readout: PulseReadoutCfg,
    tun: CalibrationTunables,
    prior: Calibrated,
    f01_mhz: float,
    pi_length_us: float,
) -> FreqGainCfg:
    defaults = _calibration_defaults("readout")
    drive = qub_pulse.model_copy(deep=True)
    drive.set_param("freq", f01_mhz)
    drive.set_param("length", pi_length_us)
    ro = readout.model_copy(deep=True)
    apply_readout_calibration(ro, prior)
    return FreqGainCfg(
        reps=tun.ro_reps,
        rounds=tun.ro_rounds,
        initial_delay=defaults.initial_delay,
        relax_delay=defaults.relax_delay,
        dev=defaults.dev,
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


def calibrate(
    env: WorkflowEnv[Calibrated],
    qub_pulse: PulseCfg,
    readout: PulseReadoutCfg,
    tun: CalibrationTunables,
    flux: float,
    carry: CalCarry,
    *,
    savers: CalibrationSavers,
) -> Generator[Effect, None, Calibrated | CalibrationFailed]:
    """Acquire native calibration stages and update carry only on full success.

    Use yield from inside a workflow step. env supplies SoC through Engine
    ports. qub_pulse is the shared host-resolved drive; readout is the shared
    pulse-backed module. Routing/waveform/gain come from these fixed selections.
    tun controls grids, averaging and acceptance. flux must be finite and labels accepted frequency history.
    carry is the current step's mutable copy, seeded with a valid Calibrated.
    savers groups three explicit writers, avoiding callable context/registry;
    each writer
    writes/closes its exact Engine temporary path before Completed is returned.

    Each stage builds its native cfg from the two selected modules, captured
    defaults and this call's tunables/history. Frequency/Rabi resolutions are
    fixed at 101/51; readout scans a fixed 10 MHz/51-point frequency window
    around the current accepted center. Calibration defaults are not available
    until the three Measure adapters' role-path source is received; that
    missing acquisition path raises NotImplementedError rather than using
    fabricated timing/device values.

    Frequency uses EWMA(alpha=0.5) history and the declared widening sequence.
    Reject nonpositive/nonfinite center/FWHM, negative/nonfinite error, excessive
    error/FWHM or normalized residual RMS. Rabi scales prior pi length using
    rabi_start_factor/rabi_stop_factor and uses the same residual gate, with
    error/pi replacing error/FWHM. Readout scans
    on first success and then at n_calibrated % ro_every == 0; its finite maximum
    SNR must meet min_readout_snr. Other calls retain prior readout values.
    Schedule failures and rejection return CalibrationFailed without modifying
    carry. Adapter, fitting, saver and plot errors propagate. Cancel never
    returns a value; Engine discards the generator and its step state.
    Completed acquisition files remain even if a later stage fails.
    """
    if not math.isfinite(flux):
        raise ValueError("Calibration flux must be finite")
    prior = carry.seed if carry.last is None else carry.last
    hint = prior.f01_mhz
    for _, frequency in carry.f01_by_flux:
        hint = 0.5 * frequency + 0.5 * hint
    stages = env.pbar("calibration", total=3)
    stages.set_progress(0)
    f01 = None
    for widen in tun.qf_widen:
        ax = env.axes("calibration/qubit_freq")
        ax.cla()
        ax.axvline(hint, linestyle="--", label="expected")
        ax.axvline(prior.f01_mhz, linestyle=":", label="previous")
        (line,) = ax.plot([], [])
        outcome = yield from env.run(
            run_qubit_freq,
            _frequency_cfg(qub_pulse, readout, tun, prior, widen, hint),
            save=savers.frequency,
            live=Live1D(line, y=lambda signals: np.abs(signals)),
        )
        if isinstance(outcome, Failed):
            continue
        fit = analyze_qubit_freq(outcome.result)
        residual = fit.quality.normalized_residual_rms
        if (
            math.isfinite(fit.frequency_mhz)
            and fit.frequency_mhz > 0
            and math.isfinite(fit.width_mhz)
            and fit.width_mhz > 0
            and math.isfinite(fit.error_mhz)
            and fit.error_mhz >= 0
            and fit.error_mhz / fit.width_mhz < tun.max_frequency_rel_err
            and residual is not None
            and residual < tun.max_fit_residual
        ):
            f01 = fit.frequency_mhz
            ax.plot(outcome.result.freqs, fit.fitted, label="fit")
            break
    if f01 is None:
        return CalibrationFailed("frequency", "No accepted frequency fit")

    stages.set_progress(1)
    rabi = yield from env.run(
        run_len_rabi,
        _rabi_cfg(qub_pulse, readout, tun, prior, f01),
        save=savers.rabi,
    )
    if isinstance(rabi, Failed):
        return CalibrationFailed("rabi", rabi.reason)
    rabi_fit = analyze_len_rabi(rabi.result)
    residual = rabi_fit.quality.normalized_residual_rms
    if not (
        math.isfinite(rabi_fit.pi_length_us)
        and rabi_fit.pi_length_us > 0
        and math.isfinite(rabi_fit.error_us)
        and rabi_fit.error_us >= 0
        and rabi_fit.error_us / rabi_fit.pi_length_us < tun.max_rabi_rel_err
        and residual is not None
        and residual < tun.max_fit_residual
    ):
        return CalibrationFailed("rabi", "Rabi fit rejected")
    ax = env.axes("calibration/len_rabi")
    ax.cla()
    ax.plot(rabi.result.lengths, rabi_fit.observed, label="observed")
    ax.plot(rabi.result.lengths, rabi_fit.fitted, label="fit")

    stages.set_progress(2)
    ro_freq, ro_gain = prior.readout_freq_mhz, prior.readout_gain
    if carry.last is None or carry.n_calibrated % tun.ro_every == 0:
        ro_result = yield from env.run(
            run_ro_landscape,
            _readout_cfg(qub_pulse, readout, tun, prior, f01, rabi_fit.pi_length_us),
            save=savers.readout,
        )
        if isinstance(ro_result, Failed):
            return CalibrationFailed("readout", ro_result.reason)
        signals = ro_result.result.signals
        accepted = np.isfinite(signals) & (signals >= tun.min_readout_snr)
        if not np.any(accepted):
            return CalibrationFailed("readout", "No readout SNR meets threshold")
        # Invalid cells cannot win; this is selection policy, not repaired data.
        row, column = np.unravel_index(
            np.argmax(np.where(accepted, signals, -np.inf)), signals.shape
        )
        ro_freq = float(ro_result.result.freqs[row])
        ro_gain = float(ro_result.result.gains[column])

    cal = Calibrated(f01, rabi_fit.pi_length_us, ro_freq, ro_gain)
    carry.last = cal
    carry.f01_by_flux.append((flux, f01))
    carry.n_calibrated += 1
    stages.set_progress(3)
    return cal
