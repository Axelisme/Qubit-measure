"""Native Echo/Ramsey cfg generation and scalar conversion carriers."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.datafile import RunSnapshot
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.v2.twotone.time_domain.t2echo.core import (
    T2EchoCfg,
    T2EchoModuleCfg,
    T2EchoSweepCfg,
)
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import (
    T2RamseyCfg,
    T2RamseyModuleCfg,
    T2RamseySweepCfg,
)

from .calibration import (
    CalibratedPi2,
    CalibrationFailed,
    CalibrationFiles,
    apply_readout_calibration,
)
from .t1 import T1Plan


class T2Plan(BaseModel):
    """Fixed setpoints, native calibration modules and historical snapshot.

    flux_dev names the connected driver; flux_unit is explicit A/V/native with
    no conversion. fluxes are ordered absolute setpoints. points is the number
    of uniform delay samples. qub_pulse supplies shared spectroscopy/Rabi/pi
    waveform and gain; readout supplies pulse-backed routing. snapshot is the
    Start capture for native files, never a live source or destination chooser.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    flux_dev: str = Field(min_length=1)
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
    into maximum delay. relax_factor times guess_us fixes relax timing without
    an additional T1 estimate. Acceptance retains the old two T2 rejection
    rules and is not replaced by T1 relative-error or RMS thresholds.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    reps: int = Field(default=100, ge=1)
    rounds: int = Field(default=10, ge=1)
    guess_us: float = Field(default=20.0, gt=0)
    sweep_factor: float = Field(default=5.0, gt=0)
    relax_factor: float = Field(default=3.0, gt=0)


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
    plan: T1Plan | T2Plan, tun: T2EchoTunables, cal: CalibratedPi2, guess_us: float
) -> T2EchoCfg:
    """Build detached zero-detune Echo cfg from the successful fitted pi/pi2.

    guess_us is the positive finite delay-grid estimate in microseconds.
    Template waveform/gain is identical for pi and pi/2, with separate fitted
    durations. Readout scalars use calibration. Delay ends at sweep_factor times
    guess_us, relax at relax_factor times tun.guess_us. Invalid grid estimates
    raise ValueError; no setup/acquisition occurs or caller models change.
    """
    sweep = _sweep(plan, tun, guess_us)
    pi = plan.qub_pulse.model_copy(deep=True)
    pi.set_param("freq", cal.f01_mhz)
    pi.set_param("length", cal.pi_length_us)
    pi2 = plan.qub_pulse.model_copy(deep=True)
    pi2.set_param("freq", cal.f01_mhz)
    pi2.set_param("length", cal.pi2_length_us)
    ro = plan.readout.model_copy(deep=True)
    apply_readout_calibration(ro, cal)
    return T2EchoCfg(
        reps=tun.reps,
        rounds=tun.rounds,
        relax_delay=tun.relax_factor * tun.guess_us,
        detune=0.0,
        modules=T2EchoModuleCfg(pi_pulse=pi, pi2_pulse=pi2, readout=ro),
        sweep=T2EchoSweepCfg(length=sweep),
    )


def t2ramsey_cfg(
    plan: T1Plan | T2Plan, tun: T2RamseyTunables, cal: CalibratedPi2, guess_us: float
) -> T2RamseyCfg:
    """Build detached Ramsey cfg with native detune_ratio/length_step MHz.

    Inputs share t2echo_cfg's fixed modules and positive finite microsecond grid
    contract. Pi/2 uses fit.pi2_length_us at the same waveform/gain, never half
    the fitted pi. No hardware, plotting or saving occurs; cfg errors propagate.
    """
    sweep = _sweep(plan, tun, guess_us)
    pi2 = plan.qub_pulse.model_copy(deep=True)
    pi2.set_param("freq", cal.f01_mhz)
    pi2.set_param("length", cal.pi2_length_us)
    ro = plan.readout.model_copy(deep=True)
    apply_readout_calibration(ro, cal)
    return T2RamseyCfg(
        reps=tun.reps,
        rounds=tun.rounds,
        relax_delay=tun.relax_factor * tun.guess_us,
        detune=tun.detune_ratio / sweep.step,
        modules=T2RamseyModuleCfg(pi2_pulse=pi2, readout=ro),
        sweep=T2RamseySweepCfg(length=sweep),
    )


def _sweep(plan: T1Plan | T2Plan, tun: T2EchoTunables, guess: float) -> SweepCfg:
    if not np.isfinite(guess) or guess <= 0:
        raise ValueError("T2 grid guess must be positive and finite")
    stop = tun.sweep_factor * guess
    return SweepCfg(
        start=0.0, stop=stop, expts=plan.points, step=stop / (plan.points - 1)
    )
