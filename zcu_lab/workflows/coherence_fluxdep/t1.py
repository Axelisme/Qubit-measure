"""Native T1 flux-dependence plan, settings and saved point data.

The workflow owns generation, calibration policy and accepted scalar history.
Engine owns copied inputs, cancellation and record commits. The workflow owns
native filenames and saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.workflows.coherence_fluxdep.calibration import (
    CalibrationFailed,
    CalibrationFiles,
)
from zcu_lab.workflows.common.calibration import Calibrated


class T1Plan(BaseModel):
    """Fixed setpoints, two resolved module choices and historical capture.

    flux_dev names the connected device. Required flux_unit is A, V or native
    (Fake); the workflow tool rejects mismatched driver units without conversion or mode
    changes. fluxes retain absolute values in that unit and order; points >= 5
    controls the number of delay samples. qub_pulse/readout
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
    points: int = Field(default=51, ge=5, description="T1 delay samples")
    qub_pulse: PulseCfg = Field(description="Resolved Rabi and downstream pi drive")
    readout: PulseReadoutCfg = Field(description="Resolved shared pulse-backed readout")
    snapshot: RunSnapshot = Field(description="Host-captured historical entry state")


class T1Tunables(BaseModel):
    """Captured T1 generation and acceptance knobs.

    reps/rounds control T1 averaging; shared calibration is owned by Coherence.
    guess_us seeds lifetime history; sweep_factor/relax_factor multiply the
    current lifetime guess into delay-grid end/relax delay in microseconds.
    sweep_mode/relax_mode choose history-based auto timing or fixed endpoints/
    relax_delay_us. Auto timing uses the sweep bounds and relax_min_us; fixed
    values bypass those clamps. uniform selects linear or exponential sampling.
    max_rel_err and max_fit_residual are strict upper bounds for error/lifetime
    and normalized residual RMS. min_lifetime_points defaults to 5 fitted delays
    in [0, lifetime], including endpoints. Invalid quality diagnostics reject.
    Defaults are provisional, confirmed at CP-B/C.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    reps: int = Field(default=1000, ge=1, description="T1 averages")
    rounds: int = Field(default=10, ge=1, description="T1 averaging rounds")
    acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    earlystop_snr: float | None = Field(
        default=20.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    guess_us: float = Field(default=20.0, gt=0, description="Initial lifetime in us")
    sweep_factor: float = Field(
        default=5.0, gt=0, description="Sweep end divided by lifetime guess"
    )
    relax_factor: float = Field(
        default=3.0, gt=0, description="Relax delay divided by lifetime guess"
    )
    sweep_mode: Literal["auto", "fixed"] = Field(
        default="auto", description="History-based or fixed delay window"
    )
    sweep_start_us: float = Field(
        default=0.5, ge=0, description="First delay in us, in either mode"
    )
    sweep_stop_us: float = Field(
        default=50.0, gt=0, description="Fixed last delay in us"
    )
    sweep_stop_min_us: float = Field(
        default=1.0, gt=0, description="Auto sweep stop lower bound in us"
    )
    max_length_us: float = Field(
        default=50.0, gt=0, description="Auto sweep stop upper bound in us"
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
    uniform: bool = Field(
        default=True,
        description="Linear sampling; false selects exponential delay spacing",
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
    max_rel_err: float = Field(
        default=0.3, gt=0, le=1, description="Strict T1 error/lifetime upper bound"
    )
    max_fit_residual: float = Field(
        default=0.2, gt=0, le=1, description="Strict normalized residual RMS bound"
    )

    @model_validator(mode="after")
    def validate_window(self) -> T1Tunables:
        """Reject reversed fixed windows or unreachable auto window bounds."""
        if self.sweep_mode == "fixed":
            if self.sweep_stop_us <= self.sweep_start_us:
                raise ValueError("Fixed T1 stop must exceed start")
        elif not self.sweep_start_us < self.sweep_stop_min_us <= self.max_length_us:
            raise ValueError("Auto T1 requires start < stop minimum <= max length")
        return self


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
