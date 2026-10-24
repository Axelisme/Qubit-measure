"""Raw GE shot policy, point outcome and calibrated acquisition configuration."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from zcu_lab.v2.singleshot.ge.core import GE_Cfg
from zcu_lab.workflows.coherence_fluxdep.t1 import T1Plan
from zcu_lab.workflows.common.calibration import (
    Calibrated,
    RabiResult,
    apply_readout_calibration,
)


class SingleshotTunables(BaseModel):
    """Choose flux spacing and raw shots per preparation after accepted Rabi.

    min_spacing is the absolute distance from the last acquired GE point in
    plan flux units; zero allows every
    eligible point. shots counts each probe-off/on cloud. Defaults are
    provisional simulation values, not calibrated hardware settings.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    min_spacing: float = Field(
        default=0.0, ge=0, description="Minimum GE spacing in plan flux units"
    )
    shots: int = Field(default=100000, ge=1, description="Shots per probe-off/on cloud")


@dataclass(frozen=True)
class SingleshotResult:
    """Raw GE outcome, without fit, discrimination metrics or workflow figures.

    status is measured for complete acquisition, skipped for a disabled stage,
    unaccepted Rabi, spacing gate or cancellation, and failed for Schedule
    failure or partial acquisition. reason is None only when measured.
    file is relative to the point directory after Completed (including partial),
    otherwise None. Unexpected acquisition/writer errors propagate.
    """

    status: Literal["measured", "skipped", "failed"]
    reason: str | None
    file: str | None


@dataclass(frozen=True)
class SingleshotInputs:
    """Capture point calibration, GE spacing history and T1-derived relaxation.

    cal is the current accepted frequency/pi with retained legal readout, or
    None when calibration was not accepted. last_flux is the last Completed
    GE setpoint in plan units, including partial, or None before any shots.
    relax_delay is T1 relax_factor times its history-based guess in us.
    """

    cal: Calibrated | None
    last_flux: float | None
    relax_delay: float


def singleshot_skip_reason(
    rabi: RabiResult, tun: SingleshotTunables, flux: float, last_flux: float | None
) -> str | None:
    """Return a GE dependency/spacing skip reason, or None if eligible.

    Require current accepted Rabi and absolute distance from last acquired GE >=
    min_spacing in plan flux units. A zero spacing allows repeated setpoints.
    Does not acquire, change history or consider the fixed plan enable flag.
    """
    if rabi.status != "accepted":
        return "Requires this point's accepted Rabi"
    if last_flux is not None and abs(flux - last_flux) < tun.min_spacing:
        return "Minimum singleshot flux spacing"
    return None


def singleshot_cfg(
    plan: T1Plan, tun: SingleshotTunables, cal: Calibrated, *, relax_delay: float
) -> GE_Cfg:
    """Build GE with this point's calibrated pi pulse and retained readout.

    relax_delay is the T1 stage's relax_factor times its history-based guess,
    in microseconds, also when T1 is disabled. Use no reset or init pulse;
    compare probe off/on. Copy modules, retaining drive waveform/gain and
    readout routing. Native cfg validation errors propagate; no acquisition,
    fitting or input mutation occurs.
    """
    pulse = plan.qub_pulse.model_copy(deep=True)
    pulse.set_param("freq", cal.f01_mhz)
    pulse.set_param("length", cal.pi_length_us)
    readout = plan.readout.model_copy(deep=True)
    apply_readout_calibration(readout, cal)
    return GE_Cfg(
        probe_pulse=pulse,
        readout=readout,
        shots=tun.shots,
        relax_delay=relax_delay,
    )
