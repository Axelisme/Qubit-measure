"""Calibrated MIST flux/gain sweep; raw IQ stays native, no threshold fit."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    Done,
    Failed,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg, ResetCfg

from zcu_lab.v2.mist.power_dep.single_trace.core import (
    PowerDepCfg,
    PowerDepModuleCfg,
    PowerDepSweepCfg,
)

from .acquire import run_mist
from .analysis import mist_signal2real
from .calibration import (
    CalCarry,
    Calibrated,
    CalibratedPi2,
    CalibrationFailed,
    CalibrationFiles,
    CalibrationSavers,
    CalibrationTunables,
    apply_readout_calibration,
    calibrate,
)
from .devices import set_flux
from .saving import save_frequency, save_mist, save_rabi, save_readout


class MISTPlan(BaseModel):
    """Fixed flux/gain grids and native preparation/disturbance modules.

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
    """Captured calibration, averaging and existing T1-guess timing knobs.

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


@dataclass
class MISTState:
    """Committed MIST progress and scalar calibration carry.

    remaining is the unattempted absolute flux sequence; cal retains only
    all-success native calibrations. traces is the raw IQ map, one row per
    attempted flux, with NaN rows for calibration/acquisition failures.
    gains is the display coordinate: requested grid before any acquisition,
    then the actual native rounded gain grid for the committed traces.
    """

    remaining: list[float]
    cal: CalCarry
    traces: NDArray[np.complex128]
    gains: NDArray[np.float64]


@dataclass(frozen=True)
class MISTPoint:
    """Committed scalar summary, without raw IQ or a fitted threshold.

    index is zero-based in plan.fluxes; flux is the actual device setpoint.
    calibration is the accepted pi/readout result or stage/reason failure.
    success is True only after native acquisition completes, otherwise False.
    reason is None on success or a calibration/Schedule diagnostic on failure.
    calibration_files names completed stages relative to this iteration on
    calibration success, or is None on calibration failure. gain_file names the
    final native trace, or is None on failed calibration/acquisition.
    """

    index: int
    flux: float
    calibration: CalibratedPi2 | CalibrationFailed
    success: bool
    reason: str | None
    calibration_files: CalibrationFiles | None
    gain_file: str | None


def init_mist(env: InitEnv[Calibrated], plan: MISTPlan) -> MISTState:
    """Copy fixed setpoints, initialize calibration carry and an all-NaN IQ map.

    Context must contain the original four-field Calibrated seed. No I/O occurs.
    """
    return MISTState(
        list(plan.fluxes),
        CalCarry(env.context),
        np.full((len(plan.fluxes), plan.gain_points), complex(np.nan, np.nan)),
        np.linspace(plan.gain_start, plan.gain_stop, plan.gain_points),
    )


def mist_cfg(plan: MISTPlan, tun: MISTTunables, cal: CalibratedPi2) -> PowerDepCfg:
    """Build detached native mist/gain cfg with reset, pi, MIST and readout.

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


def _draw_map(env: WorkflowEnv[Calibrated], plan: MISTPlan, state: MISTState) -> None:
    name = "mist/vs_flux"
    if name not in env.plots:
        env.plots.subplots(name)

    def draw(figure: Figure) -> None:
        figure.clear()
        ax = figure.subplots()
        if plan.fluxes:
            projected = np.array([mist_signal2real(trace) for trace in state.traces])
            ax.pcolormesh(state.gains, plan.fluxes, projected, shading="nearest")
        ax.set(xlabel="MIST gain (a.u.)", ylabel=f"Flux ({plan.flux_unit})")

    env.plots.draw(name, draw)


@workflow("MIST_fluxdep", plan=MISTPlan, tunables=MISTTunables, init=init_mist)
def mist_fluxdep(
    env: WorkflowEnv[Calibrated], plan: MISTPlan, tun: MISTTunables, state: MISTState
) -> Step[MISTState, MISTPoint]:
    """Calibrate and acquire the fixed gain trace at each absolute flux setpoint.

    Check soc/devices/context before effects. Calibration follows the existing
    T1 guess/timing/cadence policy. Successful MIST uses old magnitude/spread
    projection for display, saves raw IQ and returns success without fitting or
    deriving a threshold. Normal native/calibration failures commit a reason and
    keep a NaN map row. Engine owns cancellation/discard/commit. Unexpected
    native, projection, plotting and saving errors propagate, not empty success.
    """
    _ = env.soc
    _ = env.devices
    _ = env.context
    index = len(plan.fluxes) - len(state.remaining)
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
    _draw_map(env, plan, state)
    if not state.remaining:
        env.plots.save("mist/vs_flux", env.iteration_dir / "map.png")
        return Done()
    flux = state.remaining.pop(0)
    yield from set_flux(env, plan.flux_dev, flux, unit=plan.flux_unit)
    calibration = yield from calibrate(
        env,
        qub_pulse=plan.qub_pulse,
        readout=plan.readout,
        tun=tun.cal,
        flux=flux,
        carry=state.cal,
        t1_estimate_us=tun.guess_us,
        savers=CalibrationSavers(
            partial(save_frequency, snapshot=plan.snapshot),
            partial(save_rabi, snapshot=plan.snapshot),
            partial(save_readout, snapshot=plan.snapshot),
        ),
    )
    if isinstance(calibration, CalibrationFailed):
        return Next(
            MISTPoint(index, flux, calibration, False, calibration.reason, None, None),
            state,
        )
    cal, files = calibration.calibration, calibration.files
    name = "mist/gain"
    axes = None
    if name in env.plots:

        def rebuild(figure: Figure) -> None:
            figure.clear()
            figure.subplots()

        env.plots.draw(name, rebuild)
        axes = env.plots[name].axes[0]
    plot = env.plots.liveplot_1d(name, "MIST gain (a.u.)", "MIST", axes=axes)
    outcome = yield from env.run(
        run_mist,
        mist_cfg(plan, tun, cal),
        live=lambda axes, samples: plot.update(axes[0], mist_signal2real(samples)),
    )
    if isinstance(outcome, Failed):
        return Next(
            MISTPoint(index, flux, cal, False, outcome.reason, files, None), state
        )
    save_mist(outcome, env.iteration_dir / "mist.h5", snapshot=plan.snapshot)
    state.traces[index] = outcome.result.signals
    state.gains = outcome.result.gains.copy()
    env.plots.save(name, env.iteration_dir / "gain.png")
    return Next(MISTPoint(index, flux, cal, True, None, files, "mist.h5"), state)
