"""Native Echo/Ramsey cfg generation and scalar conversion carriers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.cfg_model import DeviceName
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyCfg
from zcu_lab.workflows.coherence_fluxdep.calibration import (
    CalibrationFailed,
    CalibrationFiles,
)
from zcu_lab.workflows.coherence_fluxdep.t1 import T1Plan
from zcu_lab.workflows.common.calibration import (
    CalibratedPi2,
    apply_readout_calibration,
)
from zcu_lab.workflows.common.gain import GainClamp, clamp_gain


class T2Plan(BaseModel):
    """Fixed setpoints, native calibration modules and historical snapshot.

    flux_dev names the connected driver; flux_unit is explicit A/V/native with
    no conversion. fluxes are ordered absolute setpoints. points is the number
    of uniform delay samples. qub_pulse supplies shared spectroscopy/Rabi/pi
    waveform and gain; readout supplies pulse-backed routing. snapshot is the
    Start capture for native files, never a live source or destination chooser.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    flux_dev: DeviceName = Field(min_length=1)
    flux_unit: Literal["A", "V", "native"]
    fluxes: tuple[float, ...]
    points: int = Field(default=51, ge=5)
    qub_pulse: PulseCfg
    readout: PulseReadoutCfg
    snapshot: RunSnapshot


class T2EchoTunables(BaseModel):
    """Lifetime generation knobs captured per T2 invocation.

    reps/rounds control T2 averaging; shared calibration is owned by Coherence. guess_us
    seeds the delay-history EWMA in microseconds; sweep_factor multiplies it
    into maximum delay, capped at max_length_us. Auto relax uses relax_factor
    times the caller's T1 estimate, floored at relax_min_us. Fixed modes use
    sweep_start_us/sweep_stop_us and relax_delay_us without auto clamps.
    detune_ratio adds fringes per delay step; zero selects pure Echo decay.
    min_lifetime_points defaults to 5 fitted delays in [0, lifetime], including
    endpoints. Invalid quality diagnostics and undersampling reject before the
    twice-window and mean-error rules. Ramsey inherits the same gates.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    reps: int = Field(default=1000, ge=1)
    rounds: int = Field(default=10, ge=1)
    acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    earlystop_snr: float | None = Field(
        default=20.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    guess_us: float = Field(default=20.0, gt=0)
    sweep_factor: float = Field(default=5.0, gt=0)
    relax_factor: float = Field(default=3.0, gt=0)
    sweep_mode: Literal["auto", "fixed"] = Field(
        default="auto", description="History-based or fixed delay window"
    )
    sweep_start_us: float = Field(
        default=0.0, ge=0, description="First delay in us, in either mode"
    )
    sweep_stop_us: float = Field(
        default=12.5, gt=0, description="Fixed last delay in us"
    )
    max_length_us: float = Field(
        default=12.5, gt=0, description="Auto sweep stop upper bound in us"
    )
    relax_mode: Literal["auto", "fixed"] = Field(
        default="auto", description="T1-based or fixed relaxation"
    )
    relax_delay_us: float = Field(
        default=30.0, ge=0, description="Fixed relaxation delay in us"
    )
    relax_min_us: float = Field(
        default=1.0, ge=0, description="Auto relaxation lower bound in us"
    )
    max_model_window_factor: float = Field(
        default=2.0,
        gt=0,
        description="Diagnostic lifetime / maximum fitted delay upper bound; no feedback change",
    )
    min_lifetime_points: int = Field(
        default=5,
        ge=1,
        description="Minimum fitted delay points with 0 <= delay <= lifetime",
    )
    detune_ratio: float = Field(
        default=0.1,
        description="Artificial fringes per delay step; zero selects pure decay",
    )

    @model_validator(mode="after")
    def validate_window(self) -> T2EchoTunables:
        """Reject fixed windows or auto caps that cannot exceed the start."""
        stop = self.sweep_stop_us if self.sweep_mode == "fixed" else self.max_length_us
        if stop <= self.sweep_start_us:
            raise ValueError("T2 stop/cap must exceed start")
        return self


class T2RamseyTunables(T2EchoTunables):
    """Ramsey generation adds finite detune_ratio, in fringes per step.

    Native detune in MHz equals detune_ratio divided by the requested delay
    step in microseconds. The finite ratio may be zero or signed; there is no
    guessed detune. The native default is .05; nonfinite inputs are rejected.
    All inherited fields retain their Echo timing meanings.
    """

    detune_ratio: float = 0.05


@dataclass(frozen=True)
class T2Point:
    """Scalar committed T2 outcome; raw trace remains in native acquisition.

    index is zero-based in plan.fluxes; flux is the device-unit setpoint.
    calibration is fitted CalibratedPi2 or a stage/reason failure. lifetime_us
    and error_us are fitted microsecond values, both None after rejection or
    native failure. reason is None on acceptance, otherwise a specific diagnosis.
    calibration_files names completed stages relative to this iteration on
    calibration success, or is None on calibration failure. decay_file names the
    final native trace, or is None on failed calibration/acquisition. Rejected
    fits still retain their saved raw trace.
    """

    index: int
    flux: float
    calibration: CalibratedPi2 | CalibrationFailed
    lifetime_us: float | None
    error_us: float | None
    reason: str | None
    calibration_files: CalibrationFiles | None
    decay_file: str | None


def t2echo_cfg(
    plan: T1Plan | T2Plan,
    tun: T2EchoTunables,
    cal: CalibratedPi2,
    guess_us: float,
    *,
    t1_estimate_us: float,
    gain_clamped: list[GainClamp] | None = None,
) -> T2EchoCfg:
    """Build detached Echo cfg with the fitted pi/pi2 and optional fringes.

    guess_us is the positive finite delay-grid estimate in microseconds.
    Template waveform is shared. Len uses separate fitted pi/pi2 durations at
    drive_gain. Amp uses one length and independent pi/pi2 gains. A missing
    pi2_drive_gain uses drive_gain; missing drive_gain retains template gain.
    Readout scalars use calibration. Auto stop caps sweep_factor times guess_us
    at max_length_us. Auto relax uses the positive finite T1 estimate
    (current accepted T1, historical EWMA, then seed), floored at relax_min_us.
    Fixed modes use explicit timing unchanged. Detune in MHz is ratio/step.
    Invalid grid/T1 estimates raise ValueError; inputs remain unchanged.
    Clip runtime gains to DAC limits; append any changes to gain_clamped.
    """
    sweep = _sweep(plan, tun, guess_us)
    pi = plan.qub_pulse.model_copy(deep=True)
    pi.set_param("freq", cal.f01_mhz)
    pi.set_param("length", cal.pi_length_us)
    if cal.drive_gain is not None:
        pi.set_param("gain", clamp_gain(cal.drive_gain, "pi", gain_clamped))
    pi2 = plan.qub_pulse.model_copy(deep=True)
    pi2.set_param("freq", cal.f01_mhz)
    pi2.set_param("length", cal.pi2_length_us)
    pi2_gain = cal.pi2_drive_gain if cal.pi2_drive_gain is not None else cal.drive_gain
    if pi2_gain is not None:
        pi2.set_param("gain", clamp_gain(pi2_gain, "pi2", gain_clamped))
    ro = plan.readout.model_copy(deep=True)
    apply_readout_calibration(ro, cal, gain_clamped)
    return T2EchoCfg(
        reps=tun.reps,
        rounds=tun.rounds,
        acquire_retry=tun.acquire_retry,
        earlystop_snr=tun.earlystop_snr,
        relax_delay=_relax(tun, t1_estimate_us),
        detune=tun.detune_ratio / sweep.step,
        pi_pulse=pi,
        pi2_pulse=pi2,
        readout=ro,
        length=sweep,
    )


def t2ramsey_cfg(
    plan: T1Plan | T2Plan,
    tun: T2RamseyTunables,
    cal: CalibratedPi2,
    guess_us: float,
    *,
    t1_estimate_us: float,
    gain_clamped: list[GainClamp] | None = None,
) -> T2RamseyCfg:
    """Build detached Ramsey cfg with native detune_ratio/length_step MHz.

    Inputs share t2echo_cfg's fixed modules and positive finite microsecond grid
    contract, including optional gain_clamped before/after diagnostics. Pi/2 uses cal.pi2_length_us and its independent pi2_drive_gain
    when present, otherwise the Len gain. Neither scalar is halved from pi. No hardware, plotting or saving occurs; cfg errors propagate.
    """
    sweep = _sweep(plan, tun, guess_us)
    pi2 = plan.qub_pulse.model_copy(deep=True)
    pi2.set_param("freq", cal.f01_mhz)
    pi2.set_param("length", cal.pi2_length_us)
    pi2_gain = cal.pi2_drive_gain if cal.pi2_drive_gain is not None else cal.drive_gain
    if pi2_gain is not None:
        pi2.set_param("gain", clamp_gain(pi2_gain, "pi2", gain_clamped))
    ro = plan.readout.model_copy(deep=True)
    apply_readout_calibration(ro, cal, gain_clamped)
    return T2RamseyCfg(
        reps=tun.reps,
        rounds=tun.rounds,
        acquire_retry=tun.acquire_retry,
        earlystop_snr=tun.earlystop_snr,
        relax_delay=_relax(tun, t1_estimate_us),
        detune=tun.detune_ratio / sweep.step,
        pi2_pulse=pi2,
        readout=ro,
        length=sweep,
    )


def _sweep(plan: T1Plan | T2Plan, tun: T2EchoTunables, guess: float) -> SweepCfg:
    if not np.isfinite(guess) or guess <= 0:
        raise ValueError("T2 grid guess must be positive and finite")
    stop = (
        min(tun.max_length_us, tun.sweep_factor * guess)
        if tun.sweep_mode == "auto"
        else tun.sweep_stop_us
    )
    if stop <= tun.sweep_start_us:
        raise ValueError("T2 generated stop must exceed start")
    return SweepCfg(
        start=tun.sweep_start_us,
        stop=stop,
        expts=plan.points,
        step=(stop - tun.sweep_start_us) / (plan.points - 1),
    )


def _relax(tun: T2EchoTunables, t1_estimate_us: float) -> float:
    if not np.isfinite(t1_estimate_us) or t1_estimate_us <= 0:
        raise ValueError("T1 relaxation estimate must be positive and finite")
    return (
        max(tun.relax_min_us, tun.relax_factor * t1_estimate_us)
        if tun.relax_mode == "auto"
        else tun.relax_delay_us
    )
