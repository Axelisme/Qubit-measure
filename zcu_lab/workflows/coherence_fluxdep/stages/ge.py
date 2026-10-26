"""Acquire and save raw GE from this point's calibrated pi/readout."""

from dataclasses import dataclass
from typing import Literal

from zcu_tools.experiment.scope import Scope
from zcu_tools.program import shot_time_s

from zcu_lab.v2.singleshot.ge.core import GE_Cfg, GE_Exp
from zcu_lab.workflows.coherence_fluxdep.pulses import CalibratedPulses
from zcu_lab.workflows.coherence_fluxdep.singleshot import SingleshotTunables
from zcu_lab.workflows.common.gain import GainClamp, clamp_gain
from zcu_lab.workflows.common.stages import NativeOutput, StageEnv, acquire_saved


@dataclass(frozen=True, kw_only=True)
class SingleshotResult:
    """Raw GE outcome without fitting, discrimination metrics or workflow plots.

    status is measured for complete acquisition, skipped for workflow eligibility
    gates or cancellation, and failed for acquisition failure or partial data.
    reason is None only when measured. file is singleshot-ge.h5 after any
    Completed, including partial, otherwise None. gain_clamped records truncated
    DAC requests from cfg construction, including acquisition failures.
    Unexpected acquisition/writer errors propagate. Construct with keywords.
    """

    status: Literal["measured", "skipped", "failed"]
    reason: str | None = None
    file: str | None = None
    gain_clamped: tuple[GainClamp, ...] = ()


def estimate_ge_time_s(
    pulses: CalibratedPulses, settings: SingleshotTunables, *, relax_delay_us: float
) -> float:
    """Return nominal GE shot seconds for equal probe-off/on preparations.

    pulses supplies scalar pi and retained readout modules, with timing in us;
    pi2 is unused. settings.shots counts shots per preparation; min_spacing is
    the caller's eligibility rule and does not change this per-stage estimate.
    relax_delay_us is the caller's nonnegative finite wait in us, normally the
    workflow's T1 relax_factor times its explicit T1 guess. No history is read.
    Count one full round per preparation, without retries, early stop, frequency
    widening or multiple candidates. Exclude overhead/flux ramp and clock
    rounding; inputs unchanged. Invalid counts/timing raise ValueError;
    non-scalar module timing raises TypeError. Caller decides whether GE runs.
    """
    pi, ro = pulses.pi, pulses.readout
    pulse_us = float(pi.pre_delay) + float(pi.waveform.length) + float(pi.post_delay)
    readout_us = max(
        float(ro.trig_offset) + float(ro.ro_length),
        float(ro.pre_delay) + float(ro.waveform.length) + float(ro.post_delay),
    )
    return shot_time_s(
        rounds=1,
        reps=settings.shots,
        shot_lengths_us=[readout_us, pulse_us + readout_us],
        relax_delay_us=relax_delay_us,
    )


async def measure_ge(
    scope: Scope,
    env: StageEnv,
    pulses: CalibratedPulses,
    settings: SingleshotTunables,
    *,
    relax_delay_us: float,
) -> SingleshotResult:
    """Save probe-off/on GE shots after the caller's eligibility decision.

    pulses carries this point's accepted pi and retained readout; pi2 is unused.
    settings.shots is the number per preparation. relax_delay_us is the caller's
    T1 relax_factor times its history estimate in us, even when T1 is disabled.
    env supplies captured saving identity and optional summary. Save every
    Completed as singleshot-ge.h5; partial returns failed/partial with the file,
    acquisition failure returns failed without a file. Unexpected errors
    propagate. Inputs, point records and spacing history are never modified.
    """
    gain_clamped: list[GainClamp] = []
    pi = pulses.pi.model_copy(deep=True)
    pi.set_param("gain", clamp_gain(float(pi.gain), "pi", gain_clamped))
    readout = pulses.readout.model_copy(deep=True)
    readout.set_param(
        "gain", clamp_gain(float(readout.gain), "readout", gain_clamped, lower=0.0)
    )
    cfg = GE_Cfg(
        probe_pulse=pi,
        readout=readout,
        shots=settings.shots,
        relax_delay=relax_delay_us,
    )
    acquisition = await acquire_saved(
        scope,
        env,
        GE_Exp().run,
        cfg,
        name="singleshot GE",
        output=NativeOutput(filename="singleshot-ge.h5", spec=GE_Exp.AXES_SPEC),
    )
    return SingleshotResult(
        status="measured" if acquisition.status == "completed" else "failed",
        reason="partial" if acquisition.status == "partial" else acquisition.reason,
        file=acquisition.file,
        gain_clamped=tuple(gain_clamped),
    )
