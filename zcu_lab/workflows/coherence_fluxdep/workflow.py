"""Shared-calibration coherence flux sweep with selectable independent lifetimes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.experiment.workflows import (
    Live,
    WorkflowEnv,
    workflow,
)

from zcu_lab.workflows.coherence_fluxdep.measurements import (
    LifetimeHistory,
    LifetimeKind,
    LifetimeResult,
    history_guess,
    measure_echo,
    measure_ramsey,
    measure_t1,
)
from zcu_lab.workflows.coherence_fluxdep.t1 import T1Plan, T1Tunables
from zcu_lab.workflows.coherence_fluxdep.t2 import T2EchoTunables, T2RamseyTunables
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

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    flux: float
    frequency: FrequencyResult
    rabi: RabiResult
    readout: ReadoutResult
    measurements: tuple[LifetimeResult, ...]


def _summary(
    env: WorkflowEnv[Calibrated],
    kind: LifetimeKind,
    history: LifetimeHistory,
    *,
    committed_points: int,
    directory: Path | None = None,
) -> None:
    name = f"{kind}/vs_flux"
    if name not in env.plots:
        env.plots.subplots(name)

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

    env.plots.draw(name, draw)
    if directory is not None:
        env.plots.save(name, directory / f"{kind}-summary.png")


def _append(history: LifetimeHistory, flux: float, result: LifetimeResult) -> None:
    """Keep only accepted lifetime values in this lifetime's delay-grid history."""
    if result.status == "accepted":
        if result.lifetime_us is None:
            raise ValueError("Accepted lifetime requires microseconds")
        history.append((flux, result.lifetime_us))


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
    for index, (flux, d) in enumerate(
        zip(plan.fluxes, env.commit_dirs(), strict=False)
    ):
        env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
        for kind, history, enabled in summaries:
            if enabled:
                _summary(env, kind, history, committed_points=index)
        # Set the next flux and capture the shared calibration policy.
        await set_flux(env, plan.flux_dev, flux, unit=plan.flux_unit)
        captured = CalibrationInputs(
            qub_pulse=plan.qub_pulse,
            readout=plan.readout,
            tun=tun.current.cal,
            snapshot=plan.snapshot,
            directory=d,
        )
        estimate = t1_by_flux[-1][1] if t1_by_flux else tun.current.t1.guess_us
        # Locate the resonance, then calibrate pulse durations at that frequency.
        frequency = await calibrate_frequency(
            env,
            inputs=captured,
            flux=flux,
            carry=carry,
        )
        rabi = await calibrate_rabi(
            env,
            inputs=replace(captured, tun=tun.current.cal),
            frequency=frequency,
            carry=carry,
            t1_estimate_us=estimate,
            point_index=index,
        )
        # Optimize readout on cadence; retain legal readout after nonacceptance.
        if rabi.status != "accepted" or readout_due(index, tun.current.cal):
            readout = await calibrate_readout(
                env,
                inputs=replace(captured, tun=tun.current.cal),
                rabi=rabi,
                carry=carry,
                t1_estimate_us=estimate,
            )
        else:
            readout = ReadoutResult(
                "skipped", "Readout flux-point cadence", None, None, None
            )
        # Lifetimes require current frequency and Rabi, never borrowed prior values.
        cal = current_point_calibration(rabi, carry)
        measurements: list[LifetimeResult] = []
        # Without current calibration, record only the enabled dependency skips.
        if cal is None:
            enabled_lifetimes: tuple[tuple[LifetimeKind, bool], ...] = (
                ("t1", plan.measure_t1),
                ("t2ramsey", plan.measure_t2ramsey),
                ("t2echo", plan.measure_t2echo),
            )
            measurements = [
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
            ]
        else:
            # Measure relaxation on T1's own history-based delay grid.
            if plan.measure_t1:
                t1 = await measure_t1(
                    env,
                    plan,
                    tun.current.t1,
                    cal,
                    history_guess(tun.current.t1.guess_us, t1_by_flux),
                    directory=d,
                )
                _append(t1_by_flux, flux, t1)
                measurements.append(t1)
            # Measure Ramsey dephasing with its own grid and detuning policy.
            if plan.measure_t2ramsey:
                ramsey = await measure_ramsey(
                    env,
                    plan,
                    tun.current.t2ramsey,
                    cal,
                    history_guess(tun.current.t2ramsey.guess_us, t2ramsey_by_flux),
                    directory=d,
                )
                _append(t2ramsey_by_flux, flux, ramsey)
                measurements.append(ramsey)
            # Measure echo coherence with fitted pi/2 pulses and its own grid.
            if plan.measure_t2echo:
                echo = await measure_echo(
                    env,
                    plan,
                    tun.current.t2echo,
                    cal,
                    history_guess(tun.current.t2echo.guess_us, t2echo_by_flux),
                    directory=d,
                )
                _append(t2echo_by_flux, flux, echo)
                measurements.append(echo)
        point = CoherencePoint(
            index=index,
            flux=flux,
            frequency=frequency,
            rabi=rabi,
            readout=readout,
            measurements=tuple(measurements),
        )
        await env.commit(point, d)
        directory = d
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(len(plan.fluxes))
    for kind, history, enabled in summaries:
        if enabled:
            _summary(
                env,
                kind,
                history,
                committed_points=len(plan.fluxes),
                directory=directory,
            )
