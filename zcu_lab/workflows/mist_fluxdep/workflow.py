"""Calibrated MIST flux/gain sweep; raw IQ stays native, no threshold fit."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    CalibratedPi2,
    CalibrationInputs,
    CalibrationTunables,
    FrequencyResult,
    RabiResult,
    ReadoutResult,
    apply_readout_calibration,
    calibrate_frequency,
    calibrate_rabi,
    calibrate_readout,
    current_point_calibration,
    readout_due,
)
from zcu_lab.workflows.common.devices import set_flux
from zcu_lab.workflows.common.frequency_tracking import FrequencyModel, FrequencyTracker
from zcu_lab.workflows.mist_fluxdep.acquire import run_mist
from zcu_lab.workflows.mist_fluxdep.analysis import mist_signal2real
from zcu_lab.workflows.mist_fluxdep.saving import save_mist


class MISTPlan(BaseModel):
    """Scan disturbance gain across flux to measure MIST without a threshold fit.

    flux_dev names the connected driver and flux_unit is explicit A/V/native.
    fluxes are ordered absolute setpoints. qub_pulse supplies the calibrated
    pi preparation waveform/gain and native calibration drive. reset is the
    explicitly selected native reset or None to omit it. mist_pulse supplies
    disturbance routing/waveform/frequency, not a generic missing probe.
    readout supplies pulse-backed calibrated readout routing. gain_start/stop
    are finite dimensionless gains with stop>start; gain_points>=2 fixes the
    run-locked uniform sweep. snapshot is the historical Start capture.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    flux_dev: str = Field(min_length=1)
    flux_unit: Literal["A", "V", "native"]
    fluxes: tuple[float, ...]
    qub_pulse: PulseCfg
    mist_pulse: PulseCfg
    reset: ResetCfg | None
    readout: PulseReadoutCfg
    gain_start: float
    gain_stop: float
    gain_points: int = Field(ge=2)
    predictor: FrequencyModel = Field(
        description="Captured fluxonium model and device affine map"
    )
    snapshot: RunSnapshot

    @model_validator(mode="after")
    def _gain_range(self) -> MISTPlan:
        if self.gain_stop <= self.gain_start:
            raise ValueError("MIST gain stop must exceed start")
        return self


class MISTTunables(BaseModel):
    """Control calibration and averaging to resolve the raw MIST gain response.

    cal is native calibration policy; reps/rounds average the MIST acquisition.
    guess_us is the positive T1 estimate in microseconds for calibration timing;
    relax_factor multiplies it into MIST relax delay. There is no extra T1
    acquisition, gain-grid knob, fitted threshold or new acceptance criterion.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cal: CalibrationTunables = Field(default_factory=CalibrationTunables)
    reps: int = Field(default=100, ge=1)
    rounds: int = Field(default=10, ge=1)
    guess_us: float = Field(default=20.0, gt=0)
    relax_factor: float = Field(default=3.0, gt=0)


class MISTPoint(BaseModel):
    """Summarize one calibrated MIST gain trace without deriving a threshold.

    index is zero-based in plan.fluxes; flux is the actual device setpoint.
    frequency/rabi/readout are independent named stage results, including their
    accepted scalar parts, reasons and immutable native references.
    success is True only after full native acquisition completes, otherwise False.
    reason is None on success, stopped on partial acquisition, or a dependency/
    Schedule diagnostic on failure. gain_file names every Completed native trace,
    including partial, or is None on failed calibration/acquisition.
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
    status: Literal["collecting", "completed", "skipped"] = "collecting"


def mist_cfg(plan: MISTPlan, tun: MISTTunables, cal: CalibratedPi2) -> PowerDepCfg:
    """Configure a fixed disturbance gain scan to measure MIST after a calibrated pi.

    Native init_pulse is the calibrated pi preparation; probe_pulse is the
    selected MIST disturbance pulse. Preserve fixed waveform/routing and the
    disturbance pulse. Apply calibrated preparation gain/duration/frequency and
    readout scalars. None drive_gain retains template preparation gain.
    gain SweepCfg comes entirely from plan and relax timing from tun. Native
    validation errors propagate; input models stay unchanged and no I/O occurs.
    """
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
        relax_delay=tun.relax_factor * tun.guess_us,
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


def _save_gain(plots: Plots, result: PowerDepResult, directory: Path) -> None:
    name = "mist/gain"
    if name not in plots:
        plots.subplots(name)

    def draw(figure: Figure) -> None:
        figure.clear()
        axis = figure.subplots()
        axis.plot(result.gains, mist_signal2real(result.signals), ".-")
        axis.set(xlabel="MIST gain (a.u.)", ylabel="MIST")

    plots.draw(name, draw)
    plots.save(name, directory / "gain.png")


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
    """Keep the existing Rabi dependency and readout cadence before point cancellation."""
    assert point_record.rabi is not None
    if point_record.rabi.status != "accepted" or readout_due(
        inputs.point_index, inputs.tun
    ):
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
    if iteration.cancelled:
        point_record.status = "skipped"
    return iteration.cancelled


MIST_MAP_NAME = "mist/vs_flux"


def prepare_mist_map(plots: Plots) -> None:
    """Retain the workflow map presentation even for an empty sweep."""
    if MIST_MAP_NAME not in plots:
        plots.subplots(MIST_MAP_NAME)


def draw_mist_map(
    plots: Plots,
    plan: MISTPlan,
    traces: NDArray[np.complex128],
    gains: NDArray[np.float64],
) -> None:
    """Render the same live map before calibration and after acquiring a row."""
    prepare_mist_map(plots)

    def draw(figure: Figure) -> None:
        figure.clear()
        ax = figure.subplots()
        if plan.fluxes:
            projected = np.array([mist_signal2real(trace) for trace in traces])
            ax.pcolormesh(gains, plan.fluxes, projected, shading="nearest")
        ax.set(xlabel="MIST gain (a.u.)", ylabel=f"Flux ({plan.flux_unit})")

    plots.draw(MIST_MAP_NAME, draw)


@workflow("mist_fluxdep", "Calibrate and acquire MIST gain traces versus flux.")
async def mist_fluxdep(
    env: WorkflowEnv[Calibrated], plan: MISTPlan, tun: Live[MISTTunables]
) -> None:
    """Calibrate and acquire a fixed gain trace at each absolute flux setpoint.

    Current frequency/Rabi are required. Nonaccepted readout retains legal
    prior settings. Save raw IQ and commit one MISTPoint without a threshold fit;
    normal calibration/acquisition failure retains a NaN map row and its reason.
    Partial saves raw IQ then records success=False/reason=stopped, without
    projecting the trace into the map or declaring success. The point still
    commits and later points follow the same flow.
    Unexpected native/plot/save errors propagate. Pause resumes the pending
    operation, not the whole point. Tunables reads use tun.current.
    To revisit past points, start a new run with a sliced plan and seed local
    history from the previous run using the saved_run reader.
    """
    _ = env.soc
    _ = env.devices
    carry = CalCarry(
        env.context, frequency_tracker=FrequencyTracker(plan.predictor.make_predictor())
    )
    traces = np.full((len(plan.fluxes), plan.gain_points), complex(np.nan, np.nan))
    gains = np.linspace(plan.gain_start, plan.gain_stop, plan.gain_points)
    prepare_mist_map(env.plots)
    for index, (flux, iteration) in enumerate(
        zip(plan.fluxes, env.iterations(), strict=False)
    ):
        point_record = MISTPoint(index=index, flux=flux)
        try:
            d = iteration.dir
            env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
            draw_mist_map(env.plots, plan, traces, gains)
            # Set this point before calibrating its drive and readout.
            await set_flux(iteration, plan.flux_dev, flux, unit=plan.flux_unit)
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
            captured = CalibrationInputs(
                qub_pulse=plan.qub_pulse,
                readout=plan.readout,
                tun=tun.current.cal,
                snapshot=plan.snapshot,
                directory=d,
                point_index=index,
                flux=flux,
                flux_unit=plan.flux_unit,
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
            cal = current_point_calibration(point_record.rabi, carry)
            reason = point_record.rabi.reason
            success = False
            gain_file = None
            if cal is not None:
                # The custom native sequence rebuilds its gain plot on each attempt.
                outcome = await iteration.run(
                    run_mist, mist_cfg(plan, tun.current, cal), name="MIST"
                )
                if isinstance(outcome, Failed):
                    reason = outcome.reason
                else:
                    save_mist(
                        outcome,
                        d / "mist.h5",
                        snapshot=plan.snapshot,
                        identity=env.identity,
                    )
                    gain_file = "mist.h5"
                    if outcome.partial_result:
                        reason = "stopped"
                    else:
                        result = outcome.value.result
                        traces[index] = result.signals
                        gains = result.gains.copy()
                        _save_gain(env.plots, result, d)
                        success, reason = True, None
            point_record.success = success
            point_record.reason = reason
            point_record.gain_file = gain_file
            point_record.status = "skipped" if iteration.cancelled else "completed"
            draw_mist_map(env.plots, plan, traces, gains)
            if index == len(plan.fluxes) - 1:
                env.plots.save(MIST_MAP_NAME, d / "map.png")
        finally:
            iteration.commit(point_record)
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(len(plan.fluxes))
