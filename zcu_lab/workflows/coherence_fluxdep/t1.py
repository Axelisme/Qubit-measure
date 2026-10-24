"""Native T1 flux-dependence plan, state and complete cfg generation.

The workflow owns generation, calibration policy and accepted scalar history.
Engine owns copied inputs, cancellation and record commits. The workflow owns
native filenames and saving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg
from zcu_lab.workflows.coherence_fluxdep.calibration import (
    CalibrationFailed,
    CalibrationFiles,
)
from zcu_lab.workflows.common.calibration import Calibrated, apply_readout_calibration


class T1Plan(BaseModel):
    """Fixed setpoints, two resolved module choices and historical capture.

    flux_dev names the connected device. Required flux_unit is A, V or native
    (Fake); the workflow tool rejects mismatched driver units without conversion or mode
    changes. fluxes retain absolute values in that unit and order; points >= 5 controls each uniform delay grid. qub_pulse/readout
    are the shared host-resolved drive and pulse-backed readout selections.
    Drive routing/waveform/gain are fixed; runtime overwrites frequency/pi
    duration and applies accepted readout frequency/gain on shared routing.
    snapshot retains the captured entry identity, work point and roles for
    each acquisition; it does not choose destinations or read live state.
    No complete native cfg, averaging, timing policy or sweep lives in plan.
    Engine copies the plan before runtime constructs per-acquisition cfgs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    flux_dev: str = Field(min_length=1, description="Connected flux device name")
    flux_unit: Literal["A", "V", "native"] = Field(
        description="Explicit setpoint unit: A, V or native (Fake); no conversion"
    )
    fluxes: tuple[float, ...] = Field(
        description="Ordered absolute setpoints in native device units"
    )
    points: int = Field(default=51, ge=5, description="Uniform T1 delay samples")
    qub_pulse: PulseCfg = Field(
        description="Resolved shared spectroscopy/Rabi/pi drive"
    )
    readout: PulseReadoutCfg = Field(description="Resolved shared pulse-backed readout")
    snapshot: RunSnapshot = Field(description="Host-captured historical entry state")


class T1Tunables(BaseModel):
    """Captured T1 generation and acceptance knobs.

    reps/rounds control T1 averaging; shared calibration is owned by Coherence.
    guess_us seeds lifetime history; sweep_factor/relax_factor multiply the
    current lifetime guess into delay-grid end/relax delay in microseconds.
    max_rel_err and max_fit_residual are strict upper bounds for error/lifetime
    and normalized residual RMS. Defaults are provisional, confirmed at CP-B/C.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    reps: int = Field(default=100, ge=1, description="T1 averages")
    rounds: int = Field(default=10, ge=1, description="T1 averaging rounds")
    guess_us: float = Field(default=20.0, gt=0, description="Initial lifetime in us")
    sweep_factor: float = Field(
        default=5.0, gt=0, description="Sweep end divided by lifetime guess"
    )
    relax_factor: float = Field(
        default=3.0, gt=0, description="Relax delay divided by lifetime guess"
    )
    max_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict T1 error/lifetime upper bound"
    )
    max_fit_residual: float = Field(
        default=0.2, gt=0, le=1, description="Strict normalized residual RMS bound"
    )


def t1_cfg(plan: T1Plan, tun: T1Tunables, cal: Calibrated, lifetime_us: float) -> T1Cfg:
    """Build a detached complete uniform native T1 cfg, without acquisition.

    lifetime_us is the positive finite lifetime guess in microseconds;
    invalid values raise ValueError. Delay ranges from zero through
    tun.sweep_factor * lifetime_us, with plan.points samples. Relax delay is
    tun.relax_factor * lifetime_us. reps/rounds come from tun.
    Pi drive uses plan.qub_pulse waveform and cal's acquisition gain, frequency
    and duration. None drive_gain retains the template gain. Readout uses cal's frequency/gain on plan.readout routing.
    Native initial_delay/dev defaults remain owned by the native cfg model;
    no reset or extra module role is selected. Flux is set by the workflow,
    never patched into a cfg device mapping.
    Inputs are unchanged; native cfg construction/set_param errors propagate.
    """
    if not math.isfinite(lifetime_us) or lifetime_us <= 0:
        raise ValueError("T1 lifetime guess must be positive and finite")
    pi_pulse = plan.qub_pulse.model_copy(deep=True)
    pi_pulse.set_param("freq", cal.f01_mhz)
    pi_pulse.set_param("length", cal.pi_length_us)
    if cal.drive_gain is not None:
        pi_pulse.set_param("gain", cal.drive_gain)
    readout = plan.readout.model_copy(deep=True)
    apply_readout_calibration(readout, cal)
    stop = tun.sweep_factor * lifetime_us
    return T1Cfg(
        reps=tun.reps,
        rounds=tun.rounds,
        relax_delay=tun.relax_factor * lifetime_us,
        pi_pulse=pi_pulse,
        readout=readout,
        length=SweepCfg(
            start=0.0, stop=stop, expts=plan.points, step=stop / (plan.points - 1)
        ),
    )


@dataclass(frozen=True)
class T1Point:
    """Committed outcome of one absolute flux setpoint.

    index is zero-based in plan.fluxes. flux uses the connected device's units.
    calibration is the accepted scalar calibration or its stage/reason failure.
    lifetime_us and error_us are accepted lifetime/error in microseconds, or both
    None when acquisition/fit failed. reason is None on success, otherwise the
    calibration, Schedule or fit-rejection reason. Raw traces live in native files.
    calibration_files names completed stages relative to this iteration on
    calibration success, or is None on calibration failure. decay_file names the
    final native trace, or is None on failed calibration/acquisition. Rejected
    fits still retain their saved raw trace.
    """

    index: int
    flux: float
    calibration: Calibrated | CalibrationFailed
    lifetime_us: float | None
    error_us: float | None
    reason: str | None
    calibration_files: CalibrationFiles | None
    decay_file: str | None
