"""Calibrated MIST flux/gain sweep; raw IQ stays native, no threshold fit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.cfg_model import Gain
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import (
    Failed,
    Live,
    WorkflowEnv,
    workflow,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg, ResetCfg

from zcu_lab.v2.mist.power_dep.single_trace.core import PowerDepCfg, PowerDepResult
from zcu_lab.workflows.common.calibration import (
    CalCarry,
    Calibrated,
    CalibrationInputs,
    CalibrationSource,
    CalibrationTunables,
    FrequencyResult,
    RabiResult,
    ReadoutResult,
    apply_readout_calibration,
    calibrate_frequency,
    calibrate_rabi,
    calibrate_readout,
    calibration_source,
    current_point_calibration,
    readout_due,
)
from zcu_lab.workflows.common.calibration_results import StageStatus
from zcu_lab.workflows.common.devices import set_flux
from zcu_lab.workflows.common.frequency_tracking import FrequencyModel, FrequencyTracker
from zcu_lab.workflows.common.summary import RunSummary, SummaryAccumulator
from zcu_lab.workflows.common.summary_points import mist_summary_point
from zcu_lab.workflows.mist_fluxdep.acquire import run_mist
from zcu_lab.workflows.mist_fluxdep.analysis import (
    mist_fluxdep_signal2real,
    mist_signal2real,
)
from zcu_lab.workflows.mist_fluxdep.saving import save_mist


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
    flux_dev: str = Field(min_length=1)
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
    """One preparation's outcome; gain_file retains raw IQ even for partial stops."""

    status: StageStatus = Field(
        default="skipped",
        description="Accepted complete trace, rejected partial, failed acquisition, or dependency/unvisited skip",
    )
    success: bool = False
    reason: str | None = None
    gain_file: str | None = None


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


def mist_cfg(
    plan: MISTPlan,
    tun: MISTTunables,
    cal: Calibrated,
    *,
    mode: Literal["excited", "ground"] = "excited",
) -> PowerDepCfg:
    """Configure a fixed disturbance gain scan to measure MIST after a calibrated pi.

    Native init_pulse is the calibrated pi preparation; probe_pulse is the
    selected MIST disturbance pulse. Preserve fixed waveform/routing and the
    disturbance pulse. Apply the source pi's gain/duration/frequency and latest
    readout scalars. None drive_gain retains template preparation gain.
    mode=ground omits preparation. gain SweepCfg comes entirely from plan;
    relax is five times captured T1, or tun.relax_factor * tun.guess_us. Native
    validation errors propagate; input models stay unchanged and no I/O occurs.
    """
    pi = None
    if mode == "excited":
        pi = plan.qub_pulse.model_copy(deep=True)
        pi.set_param("freq", cal.f01_mhz)
        pi.set_param("length", cal.pi_length_us)
        if cal.drive_gain is not None:
            pi.set_param("gain", cal.drive_gain)
    readout = plan.readout.model_copy(deep=True)
    apply_readout_calibration(readout, cal)
    return PowerDepCfg(
        reps=tun.reps,
        rounds=tun.rounds,
        acquire_retry=tun.acquire_retry,
        relax_delay=(
            5.0 * plan.t1_us
            if plan.t1_us is not None
            else tun.relax_factor * tun.guess_us
        ),
        reset=None if plan.reset is None else plan.reset.model_copy(deep=True),
        init_pulse=pi,
        probe_pulse=plan.mist_pulse.model_copy(deep=True),
        readout=readout,
        gain=SweepCfg(
            start=plan.gain_start,
            stop=plan.gain_stop,
            expts=plan.gain_points,
            step=(plan.gain_stop - plan.gain_start) / (plan.gain_points - 1),
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


async def run_frequency_stage(
    iteration: Scope,
    inputs: CalibrationInputs,
    point_record: MISTPoint,
    plots: Plots,
    *,
    carry: CalCarry,
) -> bool:
    """Collect frequency calibration before checking whether this point was cancelled."""
    point_record.frequency = await calibrate_frequency(
        iteration,
        plots=plots,
        inputs=inputs,
        flux=point_record.flux,
        carry=carry,
    )
    if iteration.cancelled:
        point_record.status = "skipped"
    return iteration.cancelled


async def run_rabi_stage(
    iteration: Scope,
    inputs: CalibrationInputs,
    point_record: MISTPoint,
    plots: Plots,
    *,
    carry: CalCarry,
    estimate: float,
) -> bool:
    """Collect Rabi at the current frequency without starting work after point cancellation."""
    assert point_record.frequency is not None
    point_record.rabi = await calibrate_rabi(
        iteration,
        plots=plots,
        inputs=inputs,
        frequency=point_record.frequency,
        carry=carry,
        t1_estimate_us=estimate,
    )
    if iteration.cancelled:
        point_record.status = "skipped"
    return iteration.cancelled


async def run_readout_stage(
    iteration: Scope,
    inputs: CalibrationInputs,
    point_record: MISTPoint,
    plots: Plots,
    *,
    carry: CalCarry,
    estimate: float,
) -> bool:
    """Scan readout on cadence with LATEST pi, then check point cancellation."""
    assert point_record.rabi is not None
    if readout_due(inputs.point_index, inputs.tun):
        point_record.readout = await calibrate_readout(
            iteration,
            plots=plots,
            inputs=inputs,
            rabi=point_record.rabi,
            carry=carry,
            t1_estimate_us=estimate,
        )
    else:
        point_record.readout = ReadoutResult(
            "skipped",
            "Readout flux-point cadence",
            None,
            None,
            None,
        )
    if point_record.readout.status != "skipped" and carry.pi_source is not None:
        point_record.calibration_sources["readout"] = calibration_source(
            carry.pi_source, inputs.point_index
        )
    if iteration.cancelled:
        point_record.status = "skipped"
    return iteration.cancelled


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


@dataclass(frozen=True)
class MISTStageInputs:
    """Captured preparation plan and run-local delivered-facts report."""

    plan: MISTPlan
    summary: SummaryAccumulator


async def run_mist_stage(
    iteration: Scope,
    cfg: PowerDepCfg,
    point: MISTPoint,
    plots: Plots,
    *,
    inputs: MISTStageInputs,
    mode: Literal["excited", "ground"],
) -> PowerDepResult | None:
    """Acquire one preparation, save every Completed raw trace, and draw complete IQ.

    point.traces[mode] must exist. Failures and partial stops update its reason;
    return None without a map row. Complete traces return actual rounded axes.
    """
    trace = point.traces[mode]
    point.relax_source = "captured_t1" if inputs.plan.t1_us is not None else "guess"
    point.relax_delay_us = cfg.relax_delay
    outcome = await iteration.run(run_mist, cfg, name=f"MIST {mode}")
    inputs.summary.observe(outcome)
    if isinstance(outcome, Failed):
        trace.status = "failed"
        trace.reason = outcome.reason
        return None
    trace.gain_file = f"mist-{mode}.h5"
    save_mist(
        outcome,
        iteration.dir / trace.gain_file,
        snapshot=inputs.plan.snapshot,
        identity=iteration.identity,
    )
    if outcome.partial_result:
        trace.status = "rejected"
        trace.reason = "stopped"
        return None
    result = outcome.value.result
    _save_gain(plots, result, iteration.dir, mode)
    trace.status = "accepted"
    trace.success, trace.reason = True, None
    return result


@workflow("mist_fluxdep", "Calibrate and acquire MIST gain traces versus flux.")
async def mist_fluxdep(
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
    carry = CalCarry(
        env.context,
        seed_flux=plan.seed_flux,
        frequency_tracker=FrequencyTracker(plan.predictor.make_predictor()),
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
            captured = CalibrationInputs(
                qub_pulse=plan.qub_pulse,
                probe_pulse=plan.probe_pulse,
                readout=plan.readout,
                tun=tun.current.cal,
                snapshot=plan.snapshot,
                directory=d,
                point_index=index,
                flux=flux,
                flux_unit=plan.flux_unit,
                summary=summary,
            )
            # Locate frequency, calibrate pulses, then refresh readout on cadence.
            if (
                await run_frequency_stage(
                    iteration, captured, point_record, env.plots, carry=carry
                )
                or await run_rabi_stage(
                    iteration,
                    replace(captured, tun=tun.current.cal),
                    point_record,
                    env.plots,
                    carry=carry,
                    estimate=tun.current.guess_us,
                )
                or await run_readout_stage(
                    iteration,
                    replace(captured, tun=tun.current.cal),
                    point_record,
                    env.plots,
                    carry=carry,
                    estimate=tun.current.guess_us,
                )
            ):
                continue
            assert point_record.rabi is not None
            cal = current_point_calibration(
                point_record.rabi, carry, point_index=index, flux=flux
            )
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
                    trace.reason = "Requires a latest accepted pi pulse"
                    continue
                if mode == "excited":
                    point_record.calibration_sources[mode] = calibration_source(
                        carry.pi_source, index
                    )
                cfg = mist_cfg(
                    plan, tun.current, cal or carry.last or carry.seed, mode=mode
                )
                result = await run_mist_stage(
                    iteration,
                    cfg,
                    point_record,
                    env.plots,
                    inputs=MISTStageInputs(plan, summary),
                    mode=mode,
                )
                if result is None:
                    continue
                traces[mode][index] = result.signals
                gains = result.gains.copy()
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
