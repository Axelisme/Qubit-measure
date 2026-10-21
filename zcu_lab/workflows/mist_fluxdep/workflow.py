"""Calibrated MIST flux/gain sweep; raw IQ stays native, no threshold fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    Cancelled,
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


@dataclass
class MISTState:
    """Track raw MIST response across flux while retaining accepted calibration.

    remaining is the unattempted absolute flux sequence; cal retains accepted
    scalar parts independently, even after another stage fails. traces is the raw
    IQ map, one row per attempted flux, with NaN rows for calibration/acquisition
    failures.
    gains is the display coordinate: requested grid before any acquisition,
    then the actual native rounded gain grid for the committed traces.
    """

    remaining: list[float]
    cal: CalCarry
    traces: NDArray[np.complex128]
    gains: NDArray[np.float64]


@dataclass(frozen=True)
class MISTPoint:
    """Summarize one calibrated MIST gain trace without deriving a threshold.

    index is zero-based in plan.fluxes; flux is the actual device setpoint.
    frequency/rabi/readout are independent named stage results, including their
    accepted scalar parts, reasons and immutable native references.
    success is True only after native acquisition completes, otherwise False.
    reason is None on success or a dependency/Schedule diagnostic on failure.
    gain_file names the final native trace, or is None on failed
    calibration/acquisition.
    """

    index: int
    flux: float
    frequency: FrequencyResult
    rabi: RabiResult
    readout: ReadoutResult
    success: bool
    reason: str | None
    gain_file: str | None


def init_mist(env: InitEnv[Calibrated], plan: MISTPlan) -> MISTState:
    """Initialize a flux/gain IQ map to track MIST across the fixed setpoints.

    Context must contain the original four-field Calibrated seed. No I/O occurs.
    """
    return MISTState(
        list(plan.fluxes),
        CalCarry(env.context),
        np.full((len(plan.fluxes), plan.gain_points), complex(np.nan, np.nan)),
        np.linspace(plan.gain_start, plan.gain_stop, plan.gain_points),
    )


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


@workflow("mist_fluxdep", plan=MISTPlan, tunables=MISTTunables, init=init_mist)
def mist_fluxdep(
    env: WorkflowEnv[Calibrated], plan: MISTPlan, tun: MISTTunables, state: MISTState
) -> Step[MISTState, MISTPoint]:
    """Calibrate and acquire the fixed gain trace at each absolute flux setpoint.

    Check soc/devices/context before effects. Calibration follows the existing
    T1 guess/timing policy and zero-based flux-point readout cadence.
    Readout rejection/failure retains legal seed/prior settings and does not
    block MIST; missing current frequency/Rabi skips it without borrowing prior
    pi. Successful MIST uses the existing magnitude/spread
    projection for display, saves raw IQ and returns success without fitting or
    deriving a threshold. Normal native/calibration failures commit a reason and
    keep a NaN map row. Engine owns cancellation/discard/commit. Unexpected
    native, projection, plotting and saving errors propagate, not empty success.
    """
    # Check required capabilities before any effects.
    _ = env.soc
    _ = env.devices
    _ = env.context
    index = len(plan.fluxes) - len(state.remaining)
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
    # Refresh the flux map from the previously completed traces.
    name = "mist/vs_flux"
    if name not in env.plots:
        env.plots.subplots(name)

    def draw_map(figure: Figure) -> None:
        figure.clear()
        ax = figure.subplots()
        if plan.fluxes:
            projected = np.array([mist_signal2real(trace) for trace in state.traces])
            ax.pcolormesh(state.gains, plan.fluxes, projected, shading="nearest")
        ax.set(xlabel="MIST gain (a.u.)", ylabel=f"Flux ({plan.flux_unit})")

    env.plots.draw(name, draw_map)
    if not state.remaining:
        env.plots.save("mist/vs_flux", env.iteration_dir / "map.png")
        return Done()
    # Set this point before calibrating its drive and readout.
    flux = state.remaining.pop(0)
    yield from set_flux(env, plan.flux_dev, flux, unit=plan.flux_unit)
    captured = CalibrationInputs(
        qub_pulse=plan.qub_pulse,
        readout=plan.readout,
        tun=tun.cal,
        snapshot=plan.snapshot,
    )
    # Find the current transition frequency.
    frequency = yield from calibrate_frequency(
        env,
        inputs=captured,
        flux=flux,
        carry=state.cal,
    )
    # Measure this point's pi and pi/2 durations.
    rabi = yield from calibrate_rabi(
        env,
        inputs=captured,
        frequency=frequency,
        carry=state.cal,
        t1_estimate_us=tun.guess_us,
        point_index=index,
    )
    # Refresh readout only when the existing cadence or Rabi failure requires it.
    if rabi.status != "accepted" or readout_due(index, tun.cal):
        readout = yield from calibrate_readout(
            env,
            inputs=captured,
            rabi=rabi,
            carry=state.cal,
            t1_estimate_us=tun.guess_us,
        )
    else:
        readout = ReadoutResult(
            "skipped", "Readout flux-point cadence", None, None, None
        )
    # Skip MIST if this point has no accepted drive calibration.
    cal = current_point_calibration(rabi, state.cal)
    if cal is None:
        return Next(
            MISTPoint(index, flux, frequency, rabi, readout, False, rabi.reason, None),
            state,
        )
    # Acquire the fixed gain sweep after calibrated pi, with magnitude/spread display.
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
    if isinstance(outcome, (Failed, Cancelled)):
        return Next(
            MISTPoint(
                index, flux, frequency, rabi, readout, False, outcome.reason, None
            ),
            state,
        )
    # Save native IQ and publish the successful map row without a threshold fit.
    save_mist(outcome, env.iteration_dir / "mist.h5", snapshot=plan.snapshot)
    state.traces[index] = outcome.result.signals
    state.gains = outcome.result.gains.copy()
    env.plots.save(name, env.iteration_dir / "gain.png")
    return Next(
        MISTPoint(index, flux, frequency, rabi, readout, True, None, "mist.h5"), state
    )
