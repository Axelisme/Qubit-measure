"""Native T1 flux-dependence plan, state and complete cfg generation.

The workflow owns generation and accepted scalar history. Engine owns copied
inputs and execution. Native saver/declaration wiring is added separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.experiment.workflows import InitEnv
from zcu_tools.program.v2 import SweepCfg

from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg
from zcu_lab.workflows.calibration import (
    CalCarry,
    Calibrated,
    CalibrationPlan,
    CalibrationTunables,
    apply_readout_calibration,
)


class T1Plan(BaseModel):
    """Fixed native setpoints, calibration inputs and uniform T1 template.

    flux_dev names the connected device. fluxes are ordered absolute setpoints
    in that device's units. points is the delay-grid sample count.
    calibration provides complete native calibration templates, including the
    Rabi drive whose waveform/gain determines the measured pi duration.
    t1 provides reset, readout routing, optional dev and other native settings;
    its pi drive and length grid are replaced by t1_cfg. uniform must be True.
    The frozen outer plan still contains mutable cfg; Engine deep-copies it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    flux_dev: str = Field(min_length=1, description="Connected flux device name")
    fluxes: tuple[float, ...] = Field(
        description="Ordered absolute setpoints in native device units"
    )
    points: int = Field(default=51, ge=5, description="Uniform T1 delay samples")
    calibration: CalibrationPlan = Field(description="Complete calibration inputs")
    t1: T1Cfg = Field(description="Complete uniform native T1 template")

    @model_validator(mode="after")
    def _validate_uniform(self) -> T1Plan:
        if not self.t1.uniform:
            raise ValueError("T1 flux dependence requires uniform T1 acquisition")
        return self


class T1Tunables(BaseModel):
    """Captured calibration, T1 generation and acceptance knobs.

    cal controls calibration only. reps/rounds control T1 averaging.
    guess_us seeds lifetime history; sweep_factor/relax_factor multiply the
    current lifetime guess into delay-grid end/relax delay in microseconds.
    max_rel_err and max_fit_residual are strict upper bounds for error/lifetime
    and normalized residual RMS. Defaults are provisional, confirmed at CP-B/C.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cal: CalibrationTunables = Field(default_factory=CalibrationTunables)
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


@dataclass
class T1State:
    """Committed data for a T1 flux-dependence step.

    remaining contains unattempted absolute device setpoints in order.
    cal is the scalar calibration seed and all-success carry.
    t1_by_flux contains ordered (native setpoint, accepted lifetime in us) pairs.
    """

    remaining: list[float]
    cal: CalCarry
    t1_by_flux: list[tuple[float, float]] = field(default_factory=list)


def init_t1(env: InitEnv[Calibrated], plan: T1Plan) -> T1State:
    """Create fresh scalar carry/history and copy the plan's remaining setpoints.

    env.context must supply the deeply read-only Calibrated seed; absent
    context raises MissingCapability. No hardware, files or plots are touched.
    """
    return T1State(list(plan.fluxes), CalCarry(env.context))


def t1_cfg(plan: T1Plan, tun: T1Tunables, cal: Calibrated, lifetime_us: float) -> T1Cfg:
    """Build a detached complete uniform native T1 cfg, without acquisition.

    lifetime_us is the positive finite lifetime guess in microseconds;
    invalid values raise ValueError. Delay ranges from zero through
    tun.sweep_factor * lifetime_us, with plan.points samples. Relax delay is
    tun.relax_factor * lifetime_us. reps/rounds come from tun.
    Pi drive uses the calibration Rabi template's waveform/gain with cal's
    accepted frequency/duration. Readout uses cal's frequency/gain on the T1
    template's routing. Reset, optional dev and other native settings survive.
    Inputs are unchanged; native cfg construction/set_param errors propagate.
    """
    if not math.isfinite(lifetime_us) or lifetime_us <= 0:
        raise ValueError("T1 lifetime guess must be positive and finite")
    cfg = plan.t1.model_copy(deep=True)
    cfg.reps, cfg.rounds = tun.reps, tun.rounds
    cfg.relax_delay = tun.relax_factor * lifetime_us
    cfg.modules.pi_pulse = plan.calibration.rabi.modules.qub_pulse.model_copy(deep=True)
    cfg.modules.pi_pulse.set_param("freq", cal.f01_mhz)
    cfg.modules.pi_pulse.set_param("length", cal.pi_length_us)
    apply_readout_calibration(cfg.modules.readout, cal)
    stop = tun.sweep_factor * lifetime_us
    cfg.sweep.length = SweepCfg(
        start=0.0, stop=stop, expts=plan.points, step=stop / (plan.points - 1)
    )
    return cfg
