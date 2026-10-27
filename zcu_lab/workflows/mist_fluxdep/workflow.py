"""Calibrated MIST flux/gain sweep; raw IQ stays native, no threshold fit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.cfg_model import DeviceName, Gain
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import (
    Live,
    WorkflowEnv,
    workflow,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg, ResetCfg

from zcu_lab.v2.mist.power_dep.single_trace.core import (
    PowerDepCfg,
    PowerDepExp,
    PowerDepResult,
)
from zcu_lab.workflows.common.calibration import (
    CalCarry,
    Calibrated,
    CalibrationSource,
    CalibrationTunables,
    FrequencyResult,
    RabiResult,
    ReadoutResult,
    calibration_source,
    current_pi_calibration,
)
from zcu_lab.workflows.common.calibration_results import StageStatus
from zcu_lab.workflows.common.calibration_stage_types import (
    CalibrationPoint,
    CalibrationPulses,
    PointGuess,
)
from zcu_lab.workflows.common.devices import set_flux
from zcu_lab.workflows.common.frequency_tracking import FrequencyModel, FrequencyTracker
from zcu_lab.workflows.common.gain import GainClamp, clamp_gain
from zcu_lab.workflows.common.point_calibration import calibrate_point
from zcu_lab.workflows.common.stages import NativeOutput, StageEnv, acquire_saved
from zcu_lab.workflows.common.summary import RunSummary, SummaryAccumulator
from zcu_lab.workflows.common.summary_points import mist_summary_point
from zcu_lab.workflows.mist_fluxdep.acquire import run_mist
from zcu_lab.workflows.mist_fluxdep.analysis import (
    mist_fluxdep_signal2real,
    mist_signal2real,
)


class MISTPlan(BaseModel):
    """Scan disturbance gain across flux to measure MIST without a threshold fit.

    seed_flux is the seed pulse's calibration setpoint in flux_unit; None
    disables matrix-element feed-forward.
    flux_dev names the connected driver and flux_unit is explicit A/V/native.
    fluxes are ordered absolute setpoints. qub_pulse supplies the calibrated
    pi preparation waveform/gain and Length-Rabi drive. probe_pulse supplies
    independent spectroscopy waveform and initial gain. reset is the
    explicitly selected native reset or None to omit it. mist_pulse supplies
    disturbance routing/waveform/frequency, not a generic missing probe.
    readout supplies pulse-backed calibrated readout routing. gain_start/stop
    are finite dimensionless gains with stop>start; gain_points>=2 fixes the
    run-locked uniform sweep. snapshot is the historical Start capture.
    measure_mode defaults excited, ground omits pi, and both acquires each at
    one setpoint. t1_us is captured $qubit.t1 in microseconds or None if unknown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    flux_dev: DeviceName = Field(min_length=1)
    flux_unit: Literal["A", "V", "native"]
    fluxes: tuple[float, ...]
    qub_pulse: PulseCfg
    mist_pulse: PulseCfg
    reset: ResetCfg | None
    readout: PulseReadoutCfg
    gain_start: Gain
    gain_stop: Gain
    gain_points: int = Field(ge=2)
    seed_flux: float | None = Field(
        default=None,
        description="Seed pulse calibration flux in flux_unit; None disables matrix feed-forward",
    )
    probe_pulse: PulseCfg = Field(
        description="Independent spectroscopy pulse with its own waveform and initial gain"
    )
    predictor: FrequencyModel = Field(
        description="Captured fluxonium model and device affine map"
    )
    snapshot: RunSnapshot
    measure_mode: Literal["excited", "ground", "both"] = "excited"
    t1_us: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _gain_range(self) -> MISTPlan:
        if self.gain_stop <= self.gain_start:
            raise ValueError("MIST gain stop must exceed start")
        return self


class MISTTunables(BaseModel):
    """Control calibration and averaging to resolve the raw MIST gain response.

    cal is native calibration policy; reps/rounds average the MIST acquisition.
    guess_us is the positive T1 estimate in microseconds for calibration timing;
    relax_factor multiplies the guess when no captured T1 exists; otherwise
    MIST relax delay is five times plan.t1_us. There is no extra T1
    acquisition, gain-grid knob, fitted threshold or new acceptance criterion.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cal: CalibrationTunables = Field(default_factory=CalibrationTunables)
    reps: int = Field(default=1000, ge=1)
    rounds: int = Field(default=10, ge=1)
    acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    guess_us: float = Field(default=20.0, gt=0)
    relax_factor: float = Field(default=3.0, gt=0)


class MISTTrace(BaseModel):
    """One preparation's outcome, raw gain_file and any truncated DAC gains.

    gain_clamped records field names and before/after values at cfg construction.
    """

    status: StageStatus = Field(
        default="skipped",
        description="Accepted complete trace, rejected partial, failed acquisition, or dependency/unvisited skip",
    )
    success: bool = False
    reason: str | None = None
    gain_file: str | None = None
    gain_clamped: list[GainClamp] = Field(default_factory=list)


class MISTPoint(BaseModel):
    """Summarize one calibrated MIST gain trace without deriving a threshold.

    index is zero-based in plan.fluxes; flux is the actual device setpoint.
    frequency/rabi/readout are independent named stage results, including their
    accepted scalar parts, reasons and immutable native references.
    success is True only after full native acquisition completes, otherwise False.
    reason is None on success, stopped on partial acquisition, or a dependency/
    Schedule diagnostic on failure. gain_file names every Completed native trace,
    including partial, or is None on failed calibration/acquisition.
    traces maps each requested preparation to its success/reason/native file;
    gain_file references the first mode for single-trace readers. Overall success
    requires every requested mode. calibration_sources/readout_source identify
    pulse/readout producers and fallback use. relax_source/delay report whether
    captured T1 or guess supplied the actual microsecond relaxation delay.
    """

    model_config = ConfigDict(extra="forbid")

    index: int
    flux: float
    frequency: FrequencyResult | None = None
    rabi: RabiResult | None = None
    readout: ReadoutResult | None = None
    success: bool = False
    reason: str | None = None
    gain_file: str | None = None
    traces: dict[Literal["excited", "ground"], MISTTrace] = Field(default_factory=dict)
    calibration_sources: dict[str, CalibrationSource] = Field(default_factory=dict)
    readout_source: CalibrationSource | None = None
    relax_source: Literal["captured_t1", "guess"] | None = None
    relax_delay_us: float | None = None
    status: Literal["collecting", "completed", "skipped"] = "collecting"


@dataclass(frozen=True)
class MISTPulses:
    """Calibrated pulses of one MIST preparation.

    pi is this point's accepted pi preparation, or None for ground. mist is the
    disturbance pulse template and reset the optional fixed reset. readout
    holds the retained readout freq/gain. Stages copy these before changing.
    """

    pi: PulseCfg | None
    mist: PulseCfg
    reset: ResetCfg | None
    readout: PulseReadoutCfg


@dataclass(frozen=True)
class MISTSweep:
    """Requested disturbance-gain scan: start/stop in a.u. and point count."""

    start: float
    stop: float
    points: int


@dataclass(frozen=True, kw_only=True)
class MISTResult:
    """One preparation's measured outcome, independent of point/trace records.

    status is accepted for a complete trace, rejected for a stopped partial
    trace, or failed on acquisition failure. reason is None on acceptance,
    otherwise the acquisition reason or stopped. gain_file is the saved raw
    filename after any Completed, including partial, else None. gain_clamped
    lists truncated DAC requests. relax_source/relax_delay_us document the
    timing. result holds the actual rounded axes only for a complete trace.
    Construct with keywords; the stage never changes point records or history.
    """

    status: StageStatus
    reason: str | None = None
    gain_file: str | None = None
    gain_clamped: tuple[GainClamp, ...] = ()
    relax_source: Literal["captured_t1", "guess"]
    relax_delay_us: float
    result: PowerDepResult | None = None


def mist_pulses(
    plan: MISTPlan, cal: Calibrated, *, mode: Literal["excited", "ground"]
) -> MISTPulses:
    """Prepare one MIST preparation's pulses from the plan and calibrated scalars.

    Excited applies the source pi's frequency/length (and gain, when not None, else the
    template gain) to a copy of plan.qub_pulse; ground has no pi. Readout copies
    the plan and takes cal's retained frequency and gain. Inputs stay unchanged.
    """
    pi = None
    if mode == "excited":
        pi = plan.qub_pulse.model_copy(deep=True)
        pi.set_param("freq", cal.f01_mhz)
        pi.set_param("length", cal.pi_length_us)
        if cal.drive_gain is not None:
            pi.set_param("gain", cal.drive_gain)
    readout = plan.readout.model_copy(deep=True)
    readout.set_param("freq", cal.readout_freq_mhz)
    readout.set_param("gain", cal.readout_gain)
    return MISTPulses(
        pi=pi,
        mist=plan.mist_pulse.model_copy(deep=True),
        reset=None if plan.reset is None else plan.reset.model_copy(deep=True),
        readout=readout,
    )


def _mist_cfg(
    pulses: MISTPulses,
    settings: MISTTunables,
    sweep: MISTSweep,
    *,
    t1_us: float | None,
    gain_clamped: list[GainClamp],
) -> PowerDepCfg:
    pi = None
    if pulses.pi is not None:
        pi = pulses.pi.model_copy(deep=True)
        pi.set_param("gain", clamp_gain(float(pi.gain), "pi", gain_clamped))
    readout = pulses.readout.model_copy(deep=True)
    readout.set_param(
        "gain", clamp_gain(float(readout.gain), "readout", gain_clamped, lower=0.0)
    )
    start = clamp_gain(sweep.start, "mist.start", gain_clamped)
    stop = clamp_gain(sweep.stop, "mist.stop", gain_clamped)
    return PowerDepCfg(
        reps=settings.reps,
        rounds=settings.rounds,
        acquire_retry=settings.acquire_retry,
        relax_delay=(
            5.0 * t1_us
            if t1_us is not None
            else settings.relax_factor * settings.guess_us
        ),
        reset=None if pulses.reset is None else pulses.reset.model_copy(deep=True),
        init_pulse=pi,
        probe_pulse=pulses.mist.model_copy(deep=True),
        readout=readout,
        gain=SweepCfg(
            start=start,
            stop=stop,
            expts=sweep.points,
            step=(stop - start) / (sweep.points - 1),
        ),
    )


def _save_gain(
    plots: Plots, result: PowerDepResult, directory: Path, mode: str
) -> None:
    name = f"mist/{mode}/gain"
    if name not in plots:
        plots.subplots(name)

    def draw(figure: Figure) -> None:
        figure.clear()
        axis = figure.subplots()
        axis.plot(result.gains, mist_signal2real(result.signals), ".-")
        axis.set(xlabel="MIST gain (a.u.)", ylabel="MIST")

    plots.draw(name, draw)
    plots.save(name, directory / f"gain-{mode}.png")


def prepare_mist_map(plots: Plots, mode: str) -> None:
    """Retain a separate map for each preparation, even for an empty sweep."""
    name = f"mist/{mode}/vs_flux"
    if name not in plots:
        plots.subplots(name)


def draw_mist_map(
    plots: Plots,
    plan: MISTPlan,
    traces: NDArray[np.complex128],
    gains: NDArray[np.float64],
    mode: str,
) -> None:
    """Render a preparation's median-subtracted, three-sigma-clipped flux map."""
    prepare_mist_map(plots, mode)

    def draw(figure: Figure) -> None:
        figure.clear()
        ax = figure.subplots()
        if plan.fluxes:
            projected = mist_fluxdep_signal2real(traces)
            ax.pcolormesh(gains, plan.fluxes, projected, shading="nearest")
        ax.set(xlabel="MIST gain (a.u.)", ylabel=f"Flux ({plan.flux_unit})")

    plots.draw(f"mist/{mode}/vs_flux", draw)


async def run_mist_stage(
    scope: Scope,
    env: StageEnv,
    pulses: MISTPulses,
    settings: MISTTunables,
    *,
    sweep: MISTSweep,
    t1_us: float | None,
) -> MISTResult:
    """Acquire one preparation, save every Completed raw trace and draw its gain.

    pulses are the calibrated preparation, MIST disturbance, reset and readout;
    settings supplies averaging and the fallback relax policy; sweep is the
    requested disturbance-gain scan. The preparation is excited when pulses.pi exists, otherwise ground; it names
    the file, operation and plot.
    t1_us is the captured T1 (relax five times it), or None to use
    relax_factor * guess_us. Inputs stay unchanged; env supplies plots, snapshot
    and summary. Failure returns failed without a file; Partial saves raw then
    returns rejected/stopped without drawing. Complete returns accepted with the
    actual rounded axes. Unexpected exceptions propagate. No point record or
    history is modified.
    """
    mode = "ground" if pulses.pi is None else "excited"
    gain_clamped: list[GainClamp] = []
    cfg = _mist_cfg(pulses, settings, sweep, t1_us=t1_us, gain_clamped=gain_clamped)
    relax_source: Literal["captured_t1", "guess"] = (
        "captured_t1" if t1_us is not None else "guess"
    )
    acquisition = await acquire_saved(
        scope,
        env,
        run_mist,
        cfg,
        name=f"MIST {mode}",
        output=NativeOutput(filename=f"mist-{mode}.h5", spec=PowerDepExp.AXES_SPEC),
    )
    if acquisition.result is None:
        return MISTResult(
            status="rejected" if acquisition.status == "partial" else "failed",
            reason=acquisition.reason,
            gain_file=acquisition.file,
            gain_clamped=tuple(gain_clamped),
            relax_source=relax_source,
            relax_delay_us=cfg.relax_delay,
        )
    _save_gain(env.plots, acquisition.result, scope.dir, mode)
    return MISTResult(
        status="accepted",
        gain_file=acquisition.file,
        gain_clamped=tuple(gain_clamped),
        relax_source=relax_source,
        relax_delay_us=cfg.relax_delay,
        result=acquisition.result,
    )


@workflow("mist_fluxdep", "Calibrate and acquire MIST gain traces versus flux.")
async def mist_fluxdep(  # noqa: C901, PLR0912, PLR0915  # tolerated: workflow flow reads top to bottom
    env: WorkflowEnv[Calibrated], plan: MISTPlan, tun: Live[MISTTunables]
) -> RunSummary:
    """Calibrate and acquire a fixed gain trace at each absolute flux setpoint.

    Excited preparation uses latest accepted pi; ground omits preparation.
    both acquires each preparation at the same setpoint with separate raw files
    and plots. Records name fallback producer points and captured-T1/guess timing.
    Nonaccepted readout retains legal prior settings. Commit one MISTPoint;
    normal calibration/acquisition failure retains a NaN map row and its reason.
    Partial saves raw IQ then records success=False/reason=stopped, without
    projecting the trace into the map or declaring success. The point still
    commits and later points follow the same flow.
    Unexpected native/plot/save errors propagate. Pause resumes the pending
    operation, not the whole point. Tunables reads use tun.current.
    To revisit past points, start a new run with a sliced plan and seed local
    history from the previous run using the saved_run reader.
    """
    summary = SummaryAccumulator(len(plan.fluxes), plan.flux_unit)
    stage_env = StageEnv(plots=env.plots, snapshot=plan.snapshot, summary=summary)
    live_calibration = Live(lambda: tun.current.cal)
    calibration_pulses = CalibrationPulses(
        qub=plan.qub_pulse, probe=plan.probe_pulse, readout=plan.readout
    )
    sweep = MISTSweep(plan.gain_start, plan.gain_stop, plan.gain_points)
    carry = CalCarry(
        env.context,
        seed_flux=plan.seed_flux,
        frequency_tracker=FrequencyTracker(plan.predictor),
    )
    modes: tuple[Literal["excited", "ground"], ...] = (
        ("excited", "ground") if plan.measure_mode == "both" else (plan.measure_mode,)
    )
    traces = {
        mode: np.full((len(plan.fluxes), plan.gain_points), complex(np.nan, np.nan))
        for mode in modes
    }
    gains = np.linspace(plan.gain_start, plan.gain_stop, plan.gain_points)
    for mode in modes:
        prepare_mist_map(env.plots, mode)
    for index, (flux, iteration) in enumerate(
        zip(plan.fluxes, env.iterations(), strict=False)
    ):
        point_record = MISTPoint(
            index=index,
            flux=flux,
            traces={mode: MISTTrace(reason="not reached") for mode in modes},
        )
        try:
            d = iteration.dir
            env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
            for mode in modes:
                draw_mist_map(env.plots, plan, traces[mode], gains, mode)
            # Set this point before calibrating its drive and readout.
            await set_flux(iteration, plan.flux_dev, flux, unit=plan.flux_unit)
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
            calibration = await calibrate_point(
                iteration,
                stage_env,
                calibration_pulses,
                live_calibration,
                guess=PointGuess(
                    point=CalibrationPoint(
                        index=index, flux=flux, flux_unit=plan.flux_unit
                    ),
                    relax_us=tun.current.guess_us,
                ),
                carry=carry,
            )
            point_record.frequency = calibration.frequency
            point_record.rabi = calibration.rabi
            point_record.readout = calibration.readout
            if (
                point_record.readout is not None
                and point_record.readout.status != "skipped"
            ):
                point_record.calibration_sources["readout"] = CalibrationSource(
                    need="now", source_index=index, source_flux=flux, fallback=False
                )
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
            assert point_record.rabi is not None
            cal = current_pi_calibration(point_record.rabi, carry)
            point_record.readout_source = calibration_source(
                carry.readout_source, index
            )
            for mode in modes:
                trace = MISTTrace(reason=point_record.rabi.reason)
                point_record.traces[mode] = trace
                if iteration.cancelled:
                    trace.reason = "stopped"
                    continue
                if mode == "excited" and cal is None:
                    trace.reason = "Requires this point's accepted pi pulse (NOW)"
                    continue
                if mode == "excited":
                    point_record.calibration_sources[mode] = CalibrationSource(
                        need="now", source_index=index, source_flux=flux, fallback=False
                    )
                stage = await run_mist_stage(
                    iteration,
                    stage_env,
                    mist_pulses(plan, cal or carry.last or carry.seed, mode=mode),
                    tun.current,
                    sweep=sweep,
                    t1_us=plan.t1_us,
                )
                trace.status = stage.status
                trace.success = stage.status == "accepted"
                trace.reason = stage.reason
                trace.gain_file = stage.gain_file
                trace.gain_clamped = list(stage.gain_clamped)
                point_record.relax_source = stage.relax_source
                point_record.relax_delay_us = stage.relax_delay_us
                if stage.result is None:
                    continue
                traces[mode][index] = stage.result.signals
                gains = stage.result.gains.copy()
                draw_mist_map(env.plots, plan, traces[mode], gains, mode)
                if index == len(plan.fluxes) - 1:
                    env.plots.save(f"mist/{mode}/vs_flux", d / f"map-{mode}.png")
            point_record.success = all(
                trace.success for trace in point_record.traces.values()
            )
            point_record.reason = next(
                (
                    trace.reason
                    for trace in point_record.traces.values()
                    if not trace.success
                ),
                None,
            )
            point_record.gain_file = point_record.traces[modes[0]].gain_file
            point_record.status = "skipped" if iteration.cancelled else "completed"
        finally:
            iteration.commit(point_record)
            summary.points.append(mist_summary_point(point_record))
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(len(plan.fluxes))
    return summary.finish()
