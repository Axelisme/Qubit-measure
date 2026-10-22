"""Calibrated MIST flux/gain sweep; raw IQ stays native, no threshold fit."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    Failed,
    Live,
    WorkflowEnv,
    workflow,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg, ResetCfg

from zcu_lab.v2.mist.power_dep.single_trace.core import (
    PowerDepCfg,
    PowerDepModuleCfg,
    PowerDepResult,
    PowerDepSweepCfg,
)
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
    success is True only after native acquisition completes, otherwise False.
    reason is None on success or a dependency/Schedule diagnostic on failure.
    gain_file names the final native trace, or is None on failed
    calibration/acquisition.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    flux: float
    frequency: FrequencyResult
    rabi: RabiResult
    readout: ReadoutResult
    success: bool
    reason: str | None
    gain_file: str | None


def mist_cfg(plan: MISTPlan, tun: MISTTunables, cal: CalibratedPi2) -> PowerDepCfg:
    """Configure a fixed disturbance gain scan to measure MIST after a calibrated pi.

    Native init_pulse is the calibrated pi preparation; probe_pulse is the
    selected MIST disturbance pulse. Preserve fixed waveform/gain/routing,
    replacing only fitted preparation duration/frequency and readout scalars.
    gain SweepCfg comes entirely from plan and relax timing from tun. Native
    validation errors propagate; input models stay unchanged and no I/O occurs.
    """
    pi = plan.qub_pulse.model_copy(deep=True)
    pi.set_param("freq", cal.f01_mhz)
    pi.set_param("length", cal.pi_length_us)
    readout = plan.readout.model_copy(deep=True)
    apply_readout_calibration(readout, cal)
    return PowerDepCfg(
        reps=tun.reps,
        rounds=tun.rounds,
        relax_delay=tun.relax_factor * tun.guess_us,
        modules=PowerDepModuleCfg(
            reset=None if plan.reset is None else plan.reset.model_copy(deep=True),
            init_pulse=pi,
            probe_pulse=plan.mist_pulse.model_copy(deep=True),
            readout=readout,
        ),
        sweep=PowerDepSweepCfg(
            gain=SweepCfg(
                start=plan.gain_start,
                stop=plan.gain_stop,
                expts=plan.gain_points,
                step=(plan.gain_stop - plan.gain_start) / (plan.gain_points - 1),
            )
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


@workflow("mist_fluxdep", "Calibrate and acquire MIST gain traces versus flux.")
async def mist_fluxdep(
    env: WorkflowEnv[Calibrated], plan: MISTPlan, tun: Live[MISTTunables]
) -> None:
    """Calibrate and acquire a fixed gain trace at each absolute flux setpoint.

    Current frequency/Rabi are required. Nonaccepted readout retains legal
    prior settings. Save raw IQ and commit one MISTPoint without a threshold fit;
    normal calibration/acquisition failure retains a NaN map row and its reason.
    Unexpected native/plot/save errors propagate. Pause resumes the pending
    operation, not the whole point. Tunables reads use tun.current.
    To revisit past points, start a new run with a sliced plan and seed local
    history from the previous run using the saved_run reader.
    """
    _ = env.soc
    _ = env.devices
    carry = CalCarry(env.context)
    traces = np.full((len(plan.fluxes), plan.gain_points), complex(np.nan, np.nan))
    gains = np.linspace(plan.gain_start, plan.gain_stop, plan.gain_points)
    name = "mist/vs_flux"
    if name not in env.plots:
        env.plots.subplots(name)

    def draw_map(figure: Figure) -> None:
        figure.clear()
        ax = figure.subplots()
        if plan.fluxes:
            projected = np.array([mist_signal2real(trace) for trace in traces])
            ax.pcolormesh(gains, plan.fluxes, projected, shading="nearest")
        ax.set(xlabel="MIST gain (a.u.)", ylabel=f"Flux ({plan.flux_unit})")

    for index, (flux, d) in enumerate(
        zip(plan.fluxes, env.commit_dirs(), strict=False)
    ):
        env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
        env.plots.draw(name, draw_map)
        # Set this point before calibrating its drive and readout.
        await set_flux(env, plan.flux_dev, flux, unit=plan.flux_unit)
        captured = CalibrationInputs(
            qub_pulse=plan.qub_pulse,
            readout=plan.readout,
            tun=tun.current.cal,
            snapshot=plan.snapshot,
            directory=d,
        )
        # Find the current transition frequency.
        frequency = await calibrate_frequency(
            env,
            inputs=captured,
            flux=flux,
            carry=carry,
        )
        # Measure this point's pi and pi/2 durations.
        rabi = await calibrate_rabi(
            env,
            inputs=replace(captured, tun=tun.current.cal),
            frequency=frequency,
            carry=carry,
            t1_estimate_us=tun.current.guess_us,
            point_index=index,
        )
        # Refresh readout only when the existing cadence or Rabi failure requires it.
        if rabi.status != "accepted" or readout_due(index, tun.current.cal):
            readout = await calibrate_readout(
                env,
                inputs=replace(captured, tun=tun.current.cal),
                rabi=rabi,
                carry=carry,
                t1_estimate_us=tun.current.guess_us,
            )
        else:
            readout = ReadoutResult(
                "skipped", "Readout flux-point cadence", None, None, None
            )
        cal = current_point_calibration(rabi, carry)
        reason = rabi.reason
        success = False
        if cal is not None:
            # The custom native sequence rebuilds its gain plot on each attempt.
            outcome = await env.run(run_mist, mist_cfg(plan, tun.current, cal))
            if isinstance(outcome, Failed):
                reason = outcome.reason
            else:
                save_mist(
                    outcome,
                    d / "mist.h5",
                    snapshot=plan.snapshot,
                    identity=env.identity,
                )
                result = outcome.value.result
                traces[index] = result.signals
                gains = result.gains.copy()
                _save_gain(env.plots, result, d)
                success, reason = True, None
        await env.commit(
            MISTPoint(
                index=index,
                flux=flux,
                frequency=frequency,
                rabi=rabi,
                readout=readout,
                success=success,
                reason=reason,
                gain_file="mist.h5" if success else None,
            ),
            d,
        )
        env.plots.draw(name, draw_map)
        if index == len(plan.fluxes) - 1:
            env.plots.save(name, d / "map.png")
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(len(plan.fluxes))
