"""Shared-calibration coherence flux sweep with selectable independent lifetimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)

from .calibration import (
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
from .coherence_measurements import (
    LifetimeHistory,
    LifetimeKind,
    LifetimeResult,
    history_guess,
    measure_echo,
    measure_ramsey,
    measure_t1,
)
from .devices import set_flux
from .saving import save_frequency, save_rabi, save_readout
from .t1 import T1Plan, T1Tunables
from .t2 import T2EchoTunables, T2RamseyTunables


class CoherencePlan(T1Plan):
    """Fixed flux/modules/snapshot and lifetime selection, captured at Start.

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
    """Captured shared calibration and independent lifetime generation policies.

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


@dataclass
class CoherenceState:
    """Committed remaining points, partial calibration carry and separate histories.

    remaining contains unattempted absolute setpoints. cal updates stage-local
    scalar parts independently. t1/t2ramsey/t2echo_by_flux contain only accepted
    (device setpoint, lifetime_us) for that lifetime. Engine deep-copies this state
    at every invocation; only Next publishes changes, Pause/Stop discards them.
    """

    remaining: list[float]
    cal: CalCarry
    t1_by_flux: LifetimeHistory = field(default_factory=list)
    t2ramsey_by_flux: LifetimeHistory = field(default_factory=list)
    t2echo_by_flux: LifetimeHistory = field(default_factory=list)


@dataclass(frozen=True)
class CoherencePoint:
    """One committed flux point, with partial stage and enabled lifetime evidence.

    index is zero-based in plan.fluxes; flux is its explicit device-unit setpoint.
    frequency/rabi/readout retain each named stage's status, reason, accepted
    scalar parts and own relative filenames even if later stages fail.
    measurements contains enabled LifetimeResult items only, each kind once,
    ordered T1, Ramsey, Echo. Failed/skipped enabled items retain missing-value
    semantics; disabled items have no element. Native references are not an
    existence/completeness guarantee and are never inferred from effect order.
    """

    index: int
    flux: float
    frequency: FrequencyResult
    rabi: RabiResult
    readout: ReadoutResult
    measurements: tuple[LifetimeResult, ...]


def init_coherence(env: InitEnv[Calibrated], plan: CoherencePlan) -> CoherenceState:
    """Copy fixed points and initialize independent histories from legal scalar seed.

    env.context must contain Calibrated; missing context raises through env.
    No acquisition, source capture, plotting or saving occurs here.
    """
    return CoherenceState(list(plan.fluxes), CalCarry(env.context))


def _summary(
    env: WorkflowEnv[Calibrated],
    kind: LifetimeKind,
    history: LifetimeHistory,
    *,
    final: bool,
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
        axis.set(xlabel="flux setpoint", ylabel=f"{kind} (us)")

    env.plots.draw(name, draw)
    if final:
        env.plots.save(name, env.iteration_dir / f"{kind}-summary.png")


def _append(history: LifetimeHistory, flux: float, result: LifetimeResult) -> None:
    if result.status == "accepted":
        if result.lifetime_us is None:
            raise ValueError("Accepted lifetime requires microseconds")
        history.append((flux, result.lifetime_us))


def _skip(kind: LifetimeKind) -> LifetimeResult:
    return LifetimeResult(
        kind,
        "skipped",
        "Requires this point's accepted frequency and Rabi",
        None,
        None,
        None,
    )


@workflow(
    "coherence_fluxdep",
    plan=CoherencePlan,
    tunables=CoherenceTunables,
    init=init_coherence,
)
def coherence_fluxdep(
    env: WorkflowEnv[Calibrated],
    plan: CoherencePlan,
    tun: CoherenceTunables,
    state: CoherenceState,
) -> Step[CoherenceState, CoherencePoint]:
    """Run independent calibration then enabled T1/Ramsey/Echo at one flux point.

    Use through Engine with soc/devices/context/plots. Stages save native after
    Completed and update only their own carry. Current frequency and Rabi are
    required for lifetimes; readout nonacceptance retains legal readout and does
    not block them. Each lifetime fits/accepts independently and updates only its
    own history. Rabi/readout estimate is latest accepted T1 or its initial guess;
    disabled T1 is never measured to obtain it. Next publishes the whole point,
    Done follows the last point. Engine owns copied state, cancellation and
    revision capture. Unexpected adapter/writer/fit/plot exceptions propagate.
    """
    _ = env.soc
    _ = env.devices
    _ = env.context
    index = len(plan.fluxes) - len(state.remaining)
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
    final = not state.remaining
    if plan.measure_t1:
        _summary(env, "t1", state.t1_by_flux, final=final)
    if plan.measure_t2ramsey:
        _summary(env, "t2ramsey", state.t2ramsey_by_flux, final=final)
    if plan.measure_t2echo:
        _summary(env, "t2echo", state.t2echo_by_flux, final=final)
    if final:
        return Done()
    flux = state.remaining.pop(0)
    yield from set_flux(env, plan.flux_dev, flux, unit=plan.flux_unit)
    captured = CalibrationInputs(plan.qub_pulse, plan.readout, tun.cal)
    estimate = state.t1_by_flux[-1][1] if state.t1_by_flux else tun.t1.guess_us
    frequency = yield from calibrate_frequency(
        env,
        inputs=captured,
        flux=flux,
        carry=state.cal,
        save=partial(save_frequency, snapshot=plan.snapshot),
    )
    rabi = yield from calibrate_rabi(
        env,
        inputs=captured,
        frequency=frequency,
        carry=state.cal,
        t1_estimate_us=estimate,
        save=partial(save_rabi, snapshot=plan.snapshot),
    )
    if rabi.status != "accepted" or readout_due(index, tun.cal):
        readout = yield from calibrate_readout(
            env,
            inputs=captured,
            rabi=rabi,
            carry=state.cal,
            t1_estimate_us=estimate,
            save=partial(save_readout, snapshot=plan.snapshot),
        )
    else:
        readout = ReadoutResult(
            "skipped", "Readout flux-point cadence", None, None, None
        )
    cal = current_point_calibration(rabi, state.cal)
    measurements: list[LifetimeResult] = []
    if plan.measure_t1:
        t1 = (
            _skip("t1")
            if cal is None
            else (
                yield from measure_t1(
                    env,
                    plan,
                    tun.t1,
                    cal,
                    history_guess(tun.t1.guess_us, state.t1_by_flux),
                )
            )
        )
        _append(state.t1_by_flux, flux, t1)
        measurements.append(t1)
    if plan.measure_t2ramsey:
        ramsey = (
            _skip("t2ramsey")
            if cal is None
            else (
                yield from measure_ramsey(
                    env,
                    plan,
                    tun.t2ramsey,
                    cal,
                    history_guess(tun.t2ramsey.guess_us, state.t2ramsey_by_flux),
                )
            )
        )
        _append(state.t2ramsey_by_flux, flux, ramsey)
        measurements.append(ramsey)
    if plan.measure_t2echo:
        echo = (
            _skip("t2echo")
            if cal is None
            else (
                yield from measure_echo(
                    env,
                    plan,
                    tun.t2echo,
                    cal,
                    history_guess(tun.t2echo.guess_us, state.t2echo_by_flux),
                )
            )
        )
        _append(state.t2echo_by_flux, flux, echo)
        measurements.append(echo)
    return Next(
        CoherencePoint(index, flux, frequency, rabi, readout, tuple(measurements)),
        state,
    )
