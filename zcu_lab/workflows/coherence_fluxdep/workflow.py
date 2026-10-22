"""Shared-calibration coherence flux sweep with selectable independent lifetimes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import (
    Live,
    WorkflowEnv,
    workflow,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg
from zcu_lab.v2.twotone.time_domain.t2ramsey.core import T2RamseyCfg
from zcu_lab.workflows.coherence_fluxdep.measurements import (
    LifetimeHistory,
    LifetimeKind,
    LifetimeResult,
    history_guess,
    measure_echo,
    measure_ramsey,
    measure_t1,
)
from zcu_lab.workflows.coherence_fluxdep.t1 import T1Plan, T1Tunables, t1_cfg
from zcu_lab.workflows.coherence_fluxdep.t2 import (
    T2EchoTunables,
    T2RamseyTunables,
    t2echo_cfg,
    t2ramsey_cfg,
)
from zcu_lab.workflows.common.calibration import (
    CalCarry,
    Calibrated,
    CalibrationInputs,
    CalibrationTunables,
    FrequencyResult,
    RabiResult,
    ReadoutResult,
    calibrate_frequency,
    calibrate_rabi,
    calibrate_readout,
    current_point_calibration,
    readout_due,
)
from zcu_lab.workflows.common.devices import set_flux


class CoherencePlan(T1Plan):
    """Choose flux points and lifetimes to map relaxation and dephasing.

    Inherited fields retain T1Plan's explicit unit, resolved native routing and
    delay resolution. measure_t1/measure_t2ramsey/measure_t2echo are fixed bools,
    default true. At least one must be true; all false raises before acquisition.
    Only enabled items acquire/record/save, in T1, Ramsey, Echo order.
    """

    measure_t1: bool = True
    measure_t2ramsey: bool = True
    measure_t2echo: bool = True

    @model_validator(mode="after")
    def _enabled(self) -> CoherencePlan:
        if not (self.measure_t1 or self.measure_t2ramsey or self.measure_t2echo):
            raise ValueError("At least one lifetime must be enabled")
        return self


class CoherenceTunables(BaseModel):
    """Control resonant calibration and the independent lifetime measurements.

    cal is shared by the three calibration stages. t1/t2ramsey/t2echo retain
    their own averaging, initial microsecond guesses, timing and fit policies;
    none contains another calibration policy. Ramsey detune_ratio is a finite
    fringes/step value whether enabled or not, with native default .05.
    Rabi/readout use five times latest accepted T1, or t1.guess_us initially
    and when T1 is disabled. T2's own relax and history policies do not change.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cal: CalibrationTunables = Field(default_factory=CalibrationTunables)
    t1: T1Tunables = Field(default_factory=T1Tunables)
    t2ramsey: T2RamseyTunables = Field(default_factory=T2RamseyTunables)
    t2echo: T2EchoTunables = Field(default_factory=T2EchoTunables)


class CoherencePoint(BaseModel):
    """One committed flux point, with partial stage and enabled lifetime evidence.

    index is zero-based in plan.fluxes; flux is its explicit device-unit setpoint.
    frequency/rabi/readout retain each named stage's status, reason, accepted
    scalar parts and own relative filenames even if later stages fail.
    measurements contains enabled LifetimeResult items only, each kind once,
    ordered T1, Ramsey, Echo. Failed/skipped enabled items retain missing-value
    semantics; disabled items have no element. Native references are not an
    existence/completeness guarantee and are never inferred from effect order.
    """

    model_config = ConfigDict(extra="forbid")

    index: int
    flux: float
    frequency: FrequencyResult | None = None
    rabi: RabiResult | None = None
    readout: ReadoutResult | None = None
    measurements: tuple[LifetimeResult, ...] = ()
    status: Literal["collecting", "completed", "skipped"] = "collecting"


def _summary(
    plots: Plots,
    kind: LifetimeKind,
    history: LifetimeHistory,
    *,
    committed_points: int,
    directory: Path | None = None,
) -> None:
    name = f"{kind}/vs_flux"
    if name not in plots:
        plots.subplots(name)

    def draw(figure: Figure) -> None:
        figure.clear()
        axis = figure.subplots()
        if history:
            fluxes, lifetimes = zip(*history, strict=True)
            axis.scatter(fluxes, lifetimes)
        elif committed_points == 0:
            axis.text(
                0.5,
                0.5,
                "尚無已提交點",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontfamily="Noto Sans CJK TC",
            )
        axis.set(xlabel="flux setpoint", ylabel=f"{kind} (us)")

    plots.draw(name, draw)
    if directory is not None:
        plots.save(name, directory / f"{kind}-summary.png")


def draw_summaries(
    plots: Plots,
    summaries: tuple[tuple[LifetimeKind, LifetimeHistory, bool], ...],
    *,
    committed_points: int,
    directory: Path | None = None,
) -> None:
    """Draw every enabled history before a point and save the same set after the sweep."""
    for kind, history, enabled in summaries:
        if enabled:
            _summary(
                plots,
                kind,
                history,
                committed_points=committed_points,
                directory=directory,
            )


def dependency_skips(plan: CoherencePlan) -> tuple[LifetimeResult, ...]:
    """Describe enabled measurements that require this point's accepted calibration."""
    enabled_lifetimes: tuple[tuple[LifetimeKind, bool], ...] = (
        ("t1", plan.measure_t1),
        ("t2ramsey", plan.measure_t2ramsey),
        ("t2echo", plan.measure_t2echo),
    )
    return tuple(
        LifetimeResult(
            kind,
            "skipped",
            "Requires this point's accepted frequency and Rabi",
            None,
            None,
            None,
        )
        for kind, enabled in enabled_lifetimes
        if enabled
    )


def _append(history: LifetimeHistory, flux: float, result: LifetimeResult) -> None:
    """Keep only accepted lifetime values in this lifetime's delay-grid history."""
    if result.status == "accepted":
        if result.lifetime_us is None:
            raise ValueError("Accepted lifetime requires microseconds")
        history.append((flux, result.lifetime_us))


async def run_t1_stage(
    iteration: Scope,
    cfg: T1Cfg,
    point_record: CoherencePoint,
    plots: Plots,
    *,
    snapshot: RunSnapshot,
    tun: T1Tunables,
) -> bool:
    """Save and classify T1, retaining collected values before point cancellation."""
    t1 = await measure_t1(
        iteration,
        cfg,
        tun=tun,
        snapshot=snapshot,
        directory=iteration.dir,
        plots=plots,
    )
    point_record.measurements += (t1,)
    if iteration.cancelled:
        point_record.status = "skipped"
    return iteration.cancelled


async def run_ramsey_stage(
    iteration: Scope,
    cfg: T2RamseyCfg,
    point_record: CoherencePoint,
    plots: Plots,
    *,
    snapshot: RunSnapshot,
) -> bool:
    """Save and classify Ramsey, retaining collected values before point cancellation."""
    ramsey = await measure_ramsey(
        iteration,
        cfg,
        snapshot=snapshot,
        directory=iteration.dir,
        plots=plots,
    )
    point_record.measurements += (ramsey,)
    if iteration.cancelled:
        point_record.status = "skipped"
    return iteration.cancelled


async def run_echo_stage(
    iteration: Scope,
    cfg: T2EchoCfg,
    point_record: CoherencePoint,
    plots: Plots,
    *,
    snapshot: RunSnapshot,
) -> bool:
    """Save and classify Echo, retaining collected values before point cancellation."""
    echo = await measure_echo(
        iteration,
        cfg,
        snapshot=snapshot,
        directory=iteration.dir,
        plots=plots,
    )
    point_record.measurements += (echo,)
    if iteration.cancelled:
        point_record.status = "skipped"
    return iteration.cancelled


async def run_frequency_stage(
    iteration: Scope,
    inputs: CalibrationInputs,
    point_record: CoherencePoint,
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
    point_record: CoherencePoint,
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
    point_record: CoherencePoint,
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


@workflow(
    "coherence_fluxdep", "Calibrate and measure independent lifetimes versus flux."
)
async def coherence_fluxdep(
    env: WorkflowEnv[Calibrated],
    plan: CoherencePlan,
    tun: Live[CoherenceTunables],
) -> None:
    """Measure coherence versus flux with independent lifetime histories.

    Await calibration, then enabled T1, Ramsey and Echo at each flux. Current
    frequency/Rabi are required; nonaccepted readout retains legal prior values.
    Save each completed trace before fitting and commit one CoherencePoint per
    setpoint. Each accepted lifetime updates only its own delay-grid history.
    Unexpected adapter/writer/fit/plot errors propagate. Pause resumes the pending
    operation, not the whole point. Tunables reads use tun.current.
    To revisit past points, start a new run with a sliced plan and seed local
    history from the previous run using the saved_run reader.
    """
    _ = env.soc
    _ = env.devices
    carry = CalCarry(env.context)
    t1_by_flux: LifetimeHistory = []
    t2ramsey_by_flux: LifetimeHistory = []
    t2echo_by_flux: LifetimeHistory = []
    directory: Path | None = None
    summaries: tuple[tuple[LifetimeKind, LifetimeHistory, bool], ...] = (
        ("t1", t1_by_flux, plan.measure_t1),
        ("t2ramsey", t2ramsey_by_flux, plan.measure_t2ramsey),
        ("t2echo", t2echo_by_flux, plan.measure_t2echo),
    )
    for index, (flux, iteration) in enumerate(
        zip(plan.fluxes, env.iterations(), strict=False)
    ):
        point_record = CoherencePoint(index=index, flux=flux)
        try:
            d = iteration.dir
            env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
            draw_summaries(env.plots, summaries, committed_points=index)
            # Set the next flux and capture the shared calibration policy.
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
            )
            estimate = t1_by_flux[-1][1] if t1_by_flux else tun.current.t1.guess_us
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
                    estimate=estimate,
                )
                or await run_readout_stage(
                    iteration,
                    replace(captured, tun=tun.current.cal),
                    point_record,
                    carry=carry,
                    estimate=estimate,
                )
            ):
                continue
            assert point_record.rabi is not None
            cal = current_point_calibration(point_record.rabi, carry)
            # Without current calibration, record only the enabled dependency skips.
            if cal is None:
                point_record.measurements = dependency_skips(plan)
            else:
                # Measure relaxation on T1's own history-based delay grid.
                if plan.measure_t1:
                    skip = await run_t1_stage(
                        iteration,
                        t1_cfg(
                            plan,
                            tun.current.t1,
                            cal,
                            history_guess(tun.current.t1.guess_us, t1_by_flux),
                        ),
                        point_record,
                        env.plots,
                        snapshot=plan.snapshot,
                        tun=tun.current.t1,
                    )
                    _append(t1_by_flux, flux, point_record.measurements[-1])
                    if skip:
                        continue
                # Measure Ramsey dephasing with its own grid and detuning policy.
                if plan.measure_t2ramsey:
                    skip = await run_ramsey_stage(
                        iteration,
                        t2ramsey_cfg(
                            plan,
                            tun.current.t2ramsey,
                            cal,
                            history_guess(
                                tun.current.t2ramsey.guess_us, t2ramsey_by_flux
                            ),
                        ),
                        point_record,
                        env.plots,
                        snapshot=plan.snapshot,
                    )
                    _append(t2ramsey_by_flux, flux, point_record.measurements[-1])
                    if skip:
                        continue
                # Measure echo coherence with fitted pi/2 pulses and its own grid.
                if plan.measure_t2echo:
                    skip = await run_echo_stage(
                        iteration,
                        t2echo_cfg(
                            plan,
                            tun.current.t2echo,
                            cal,
                            history_guess(tun.current.t2echo.guess_us, t2echo_by_flux),
                        ),
                        point_record,
                        env.plots,
                        snapshot=plan.snapshot,
                    )
                    _append(t2echo_by_flux, flux, point_record.measurements[-1])
                    if skip:
                        continue
            point_record.status = "completed"
            directory = d
        finally:
            iteration.commit(point_record)
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(len(plan.fluxes))
    draw_summaries(
        env.plots, summaries, committed_points=len(plan.fluxes), directory=directory
    )
