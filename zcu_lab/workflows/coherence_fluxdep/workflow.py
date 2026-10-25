"""Shared-calibration coherence flux sweep with selectable independent lifetimes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
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
from zcu_tools.program.v2.modules import PulseCfg

from zcu_lab.v2.singleshot.ge.core import GE_Exp
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
from zcu_lab.workflows.coherence_fluxdep.singleshot import (
    SingleshotInputs,
    SingleshotResult,
    SingleshotTunables,
    singleshot_cfg,
    singleshot_skip_reason,
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
    CalibrationSource,
    CalibrationTunables,
    FrequencyResult,
    RabiResult,
    ReadoutResult,
    calibrate_frequency,
    calibrate_rabi,
    calibrate_readout,
    calibration_source,
    current_pi_calibration,
    current_point_calibration,
    readout_due,
)
from zcu_lab.workflows.common.devices import set_flux
from zcu_lab.workflows.common.frequency_tracking import FrequencyModel, FrequencyTracker
from zcu_lab.workflows.common.saving import save_native


class CoherencePlan(T1Plan):
    """Choose flux points and lifetimes to map relaxation and dephasing.

    probe_pulse supplies independent spectroscopy waveform/gain, while the
    inherited qub_pulse remains the Rabi and downstream pi drive.
    Inherited fields retain T1Plan's explicit unit, resolved native routing and
    delay resolution. measure_t1/measure_t2ramsey/measure_t2echo are fixed bools,
    default true. At least one must be true; all false raises before acquisition.
    Only enabled lifetimes acquire/record/save, in T1, Ramsey, Echo order.
    measure_singleshot defaults false; when true, eligible points acquire raw
    GE shots after calibration and before lifetimes, without fitting.
    """

    probe_pulse: PulseCfg = Field(
        description="Independent spectroscopy pulse with its own waveform and initial gain"
    )
    predictor: FrequencyModel = Field(
        description="Captured fluxonium model and device affine map"
    )
    measure_singleshot: bool = False
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

    cal is shared by the three calibration stages. singleshot controls raw GE
    flux spacing and shots after accepted Rabi, with provisional simulation
    defaults; GE relax timing follows T1. t1/t2ramsey/t2echo retain
    their own averaging, initial microsecond guesses, timing and fit policies;
    none contains another calibration policy. Ramsey detune_ratio is a finite
    fringes/step value whether enabled or not, with native default .05.
    Rabi/readout use five times latest accepted T1, or t1.guess_us initially
    and when T1 is disabled. T2 auto relaxation uses this point's accepted T1,
    then T1 history EWMA, then t1.guess_us; T2 delay histories stay independent.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cal: CalibrationTunables = Field(default_factory=CalibrationTunables)
    singleshot: SingleshotTunables = Field(default_factory=SingleshotTunables)
    t1: T1Tunables = Field(default_factory=T1Tunables)
    t2ramsey: T2RamseyTunables = Field(default_factory=T2RamseyTunables)
    t2echo: T2EchoTunables = Field(default_factory=T2EchoTunables)


class CoherencePoint(BaseModel):
    """One committed flux point, with partial stage and enabled lifetime evidence.

    index is zero-based in plan.fluxes; flux is its explicit device-unit setpoint.
    frequency/rabi/readout retain each named stage's status, reason, accepted
    scalar parts and own relative filenames even if later stages fail.
    singleshot retains measured/skipped/failed, its reason and raw GE filename;
    None means the stage has no result yet (including historical records).
    measurements contains enabled LifetimeResult items only, each kind once,
    ordered T1, Ramsey, Echo. Failed/skipped enabled items retain missing-value
    semantics; disabled items have no element. Native references are not an
    existence/completeness guarantee and are never inferred from effect order.
    calibration_sources names each consumer's pulse producer and fallback use;
    readout_source identifies the retained readout producer or fixed host seed.
    """

    model_config = ConfigDict(extra="forbid")

    index: int
    flux: float
    frequency: FrequencyResult | None = None
    rabi: RabiResult | None = None
    readout: ReadoutResult | None = None
    singleshot: SingleshotResult | None = None
    calibration_sources: dict[str, CalibrationSource] = Field(default_factory=dict)
    readout_source: CalibrationSource | None = None
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
    if kind not in plots:
        plots.subplots(kind, nrows=2)

    def draw(figure: Figure) -> None:
        figure.set_layout_engine("constrained")
        axis = figure.axes[0]
        axis.clear()
        if history:
            fluxes, lifetimes, errors = zip(*history, strict=True)
            axis.errorbar(fluxes, lifetimes, yerr=errors, fmt="o", capsize=3)
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
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))

    plots.draw(kind, draw)
    if directory is not None:
        plots.save(kind, directory / f"{kind}-decay.png")


def draw_summaries(
    plots: Plots,
    summaries: tuple[tuple[LifetimeKind, LifetimeHistory, bool], ...],
    *,
    committed_points: int,
    directory: Path | None = None,
) -> None:
    """Update enabled lifetime top panels without clearing current decay/fit.

    Histories contain accepted (flux, lifetime_us, error_us) triples only.
    Save each combined figure as <kind>-decay.png when directory is supplied.
    """
    for kind, history, enabled in summaries:
        if enabled:
            _summary(
                plots,
                kind,
                history,
                committed_points=committed_points,
                directory=directory,
            )


def dependency_skips(
    plan: CoherencePlan, *, include_t1: bool = True
) -> tuple[LifetimeResult, ...]:
    """Describe enabled measurements lacking pulses; omit T1 if already recorded."""
    enabled_lifetimes: tuple[tuple[LifetimeKind, bool], ...] = (
        ("t1", plan.measure_t1 and include_t1),
        ("t2ramsey", plan.measure_t2ramsey),
        ("t2echo", plan.measure_t2echo),
    )
    return tuple(
        LifetimeResult(
            kind,
            "skipped",
            "Requires this point's accepted pi pulse (NOW)"
            if kind == "t1"
            else "Requires a latest accepted pi/pi2 pulse",
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
        if result.lifetime_us is None or result.error_us is None:
            raise ValueError("Accepted lifetime requires microseconds and error")
        history.append((flux, result.lifetime_us, result.error_us))


async def run_t1_stage(
    iteration: Scope,
    cfg: T1Cfg | None,
    point_record: CoherencePoint,
    plots: Plots,
    *,
    snapshot: RunSnapshot,
    tun: T1Tunables,
) -> bool:
    """Use cfg from current_pi_calibration, independent of pi2 trust.

    None cfg or nonaccepted Rabi records a NOW dependency skip. Otherwise record
    this point as the pi producer, save and classify T1, then return whether the
    iteration was cancelled. cfg contains this point\'s pi frequency and gain.
    """
    if (
        cfg is None
        or point_record.rabi is None
        or point_record.rabi.status != "accepted"
    ):
        point_record.measurements += (
            LifetimeResult(
                "t1",
                "skipped",
                "Requires this point's accepted pi pulse (NOW)",
                None,
                None,
                None,
            ),
        )
        return iteration.cancelled
    point_record.calibration_sources["t1"] = CalibrationSource(
        need="now",
        source_index=point_record.index,
        source_flux=point_record.flux,
        fallback=False,
    )
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
    source: CalibrationSource,
) -> bool:
    """Record the pi2 producer, then save and classify Ramsey before cancellation."""
    point_record.calibration_sources["t2ramsey"] = source
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
    source: CalibrationSource,
) -> bool:
    """Record the pi/pi2 producer, then save and classify Echo before cancellation."""
    point_record.calibration_sources["t2echo"] = source
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


@dataclass(frozen=True)
class CalibrationStageHistory:
    """Coroutine-local history borrowed by the calibration stages.

    carry retains accepted frequency/pi/readout and calibration plot traces.
    t1_by_flux contains accepted (flux, lifetime_us, error_us) triples; its most
    recent lifetime supplies Rabi/readout relaxation, not a delay-grid EWMA.
    """

    carry: CalCarry
    t1_by_flux: LifetimeHistory


async def run_calibration_stages(
    iteration: Scope,
    plan: CoherencePlan,
    tun: Live[CoherenceTunables],
    plots: Plots,
    point_record: CoherencePoint,
    history: CalibrationStageHistory,
) -> None:
    """Initialize GE status, set flux and collect frequency, Rabi, then readout.

    history supplies calibration carry and an alpha=.5 accepted T1 estimate.
    With no accepted T1, use the legal initial guess for Rabi/readout relaxation.
    Read live calibration tunables before each stage. Return None, filling the supplied
    mutable record as each result arrives. status=skipped means cancellation,
    so no later stage should run. The caller owns the single finally commit.
    Unexpected errors propagate with already collected fields retained.
    """
    point_record.singleshot = SingleshotResult(
        "skipped",
        "Not reached (point cancelled or interrupted)"
        if plan.measure_singleshot
        else "Singleshot disabled",
        None,
    )
    await set_flux(iteration, plan.flux_dev, point_record.flux, unit=plan.flux_unit)
    if iteration.cancelled:
        point_record.status = "skipped"
        return
    estimate = (
        history_guess(history.t1_by_flux[0][1], history.t1_by_flux[1:])
        if history.t1_by_flux
        else tun.current.t1.guess_us
    )
    captured = CalibrationInputs(
        qub_pulse=plan.qub_pulse,
        probe_pulse=plan.probe_pulse,
        readout=plan.readout,
        tun=tun.current.cal,
        snapshot=plan.snapshot,
        directory=iteration.dir,
        point_index=point_record.index,
        flux=point_record.flux,
        flux_unit=plan.flux_unit,
    )
    if await run_frequency_stage(
        iteration, captured, point_record, plots, carry=history.carry
    ):
        return
    if await run_rabi_stage(
        iteration,
        replace(captured, tun=tun.current.cal),
        point_record,
        plots,
        carry=history.carry,
        estimate=estimate,
    ):
        return
    await run_readout_stage(
        iteration,
        replace(captured, tun=tun.current.cal),
        point_record,
        plots,
        carry=history.carry,
        estimate=estimate,
    )


async def run_singleshot_stage(
    iteration: Scope,
    plan: CoherencePlan,
    tun: SingleshotTunables,
    point_record: CoherencePoint,
    inputs: SingleshotInputs,
) -> float | None:
    """Gate, acquire and save raw GE, returning the last acquired flux setpoint.

    Call only when plan.measure_singleshot is true, after calibration/readout.
    inputs captures current calibration, last GE flux and T1-derived relaxation.
    Skip unaccepted Rabi or insufficient spacing. Save every Completed as
    singleshot-ge.h5, including partial; partial is failed/reason=partial.
    Failed acquisition has no file and does not advance spacing history.
    Neither fit nor workflow plotting runs. Record this point as the NOW pi
    producer even when pi2 is untrusted. Update point_record before checking
    iteration cancellation. Unexpected writer/acquisition errors propagate.
    """
    if iteration.cancelled:
        return inputs.last_flux
    if point_record.rabi is None:
        raise ValueError("Singleshot stage requires this point's Rabi result")
    reason = singleshot_skip_reason(
        point_record.rabi, tun, point_record.flux, inputs.last_flux
    )
    if reason is not None:
        point_record.singleshot = SingleshotResult("skipped", reason, None)
        return inputs.last_flux
    if inputs.cal is None:
        raise ValueError("Eligible singleshot requires current accepted calibration")
    point_record.calibration_sources["singleshot"] = CalibrationSource(
        need="now",
        source_index=point_record.index,
        source_flux=point_record.flux,
        fallback=False,
    )
    cfg = singleshot_cfg(plan, tun, inputs.cal, relax_delay=inputs.relax_delay)
    ge = await iteration.run(GE_Exp().run, cfg, name="singleshot GE")
    if isinstance(ge, Failed):
        point_record.singleshot = SingleshotResult("failed", ge.reason, None)
        return inputs.last_flux
    save_native(
        ge,
        iteration.dir / "singleshot-ge.h5",
        spec=GE_Exp.AXES_SPEC,
        snapshot=plan.snapshot,
        identity=iteration.identity,
    )
    point_record.singleshot = SingleshotResult(
        "failed" if ge.partial_result else "measured",
        "partial" if ge.partial_result else None,
        "singleshot-ge.h5",
    )
    return point_record.flux


@workflow(
    "coherence_fluxdep", "Calibrate and measure independent lifetimes versus flux."
)
async def coherence_fluxdep(
    env: WorkflowEnv[Calibrated],
    plan: CoherencePlan,
    tun: Live[CoherenceTunables],
) -> None:
    """Measure coherence versus flux with independent lifetime histories.

    Await calibration, optional raw GE, then enabled T1, Ramsey and Echo. GE
    requires accepted Rabi and absolute flux distance >= singleshot.min_spacing
    from the last Completed GE, including partial. GE saves raw only and uses
    T1's history-based relax timing. Rabi frequency is NOW; T1/GE require this
    point's accepted pi, independent of pi2 trust. Ramsey/Echo use latest
    accepted pi/pi2, with producer coordinates in calibration_sources.
    Nonaccepted readout retains legal prior values.
    Save each completed trace before fitting and commit one CoherencePoint per
    setpoint. Each accepted lifetime updates only its own delay-grid history.
    Unexpected adapter/writer/fit/plot errors propagate. Pause resumes the pending
    operation, not the whole point. Tunables reads use tun.current.
    To revisit past points, start a new run with a sliced plan and seed local
    history from the previous run using the saved_run reader.
    """
    _ = env.soc
    _ = env.devices
    carry = CalCarry(
        env.context, frequency_tracker=FrequencyTracker(plan.predictor.make_predictor())
    )
    last_ge_flux: float | None = None
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
            env.pbar("flux points", total=len(plan.fluxes)).set_progress(index)
            draw_summaries(env.plots, summaries, committed_points=index)
            # Set flux, locate frequency, calibrate pulses and refresh readout.
            await run_calibration_stages(
                iteration,
                plan,
                tun,
                env.plots,
                point_record,
                CalibrationStageHistory(carry, t1_by_flux),
            )
            # T1/GE resolve NOW pi independently of T2 LATEST pi/pi2.
            now_cal = current_pi_calibration(point_record.rabi, carry)
            cal = (
                current_point_calibration(
                    point_record.rabi, carry, point_index=index, flux=flux
                )
                if point_record.rabi is not None
                else None
            )
            point_record.readout_source = calibration_source(
                carry.readout_source, index
            )
            # Optionally collect raw GE after readout, before any lifetime.
            if plan.measure_singleshot:
                last_ge_flux = await run_singleshot_stage(
                    iteration,
                    plan,
                    tun.current.singleshot,
                    point_record,
                    SingleshotInputs(
                        now_cal,
                        last_ge_flux,
                        tun.current.t1.relax_factor
                        * history_guess(tun.current.t1.guess_us, t1_by_flux),
                    ),
                )
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
            # T1 needs current pi, not pi2 or a prior pair.
            if plan.measure_t1:
                skip = await run_t1_stage(
                    iteration,
                    t1_cfg(
                        plan,
                        tun.current.t1,
                        now_cal,
                        history_guess(tun.current.t1.guess_us, t1_by_flux),
                    )
                    if now_cal is not None
                    else None,
                    point_record,
                    env.plots,
                    snapshot=plan.snapshot,
                    tun=tun.current.t1,
                )
                _append(t1_by_flux, flux, point_record.measurements[-1])
                if skip:
                    continue
            # Missing LATEST pair skips only T2; T1 was independently resolved.
            if cal is None:
                point_record.measurements += dependency_skips(plan, include_t1=False)
            else:
                # Prefer current accepted T1; failed/disabled T1 uses its EWMA/seed.
                t1_estimate = next(
                    (
                        item.lifetime_us
                        for item in point_record.measurements
                        if item.kind == "t1"
                        and item.status == "accepted"
                        and item.lifetime_us is not None
                    ),
                    history_guess(tun.current.t1.guess_us, t1_by_flux),
                )
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
                            t1_estimate_us=t1_estimate,
                        ),
                        point_record,
                        env.plots,
                        snapshot=plan.snapshot,
                        source=calibration_source(carry.pi_source, index),
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
                            t1_estimate_us=t1_estimate,
                        ),
                        point_record,
                        env.plots,
                        snapshot=plan.snapshot,
                        source=calibration_source(carry.pi_source, index),
                    )
                    _append(t2echo_by_flux, flux, point_record.measurements[-1])
                    if skip:
                        continue
            point_record.status = "completed"
            directory = iteration.dir
        finally:
            iteration.commit(point_record)
            draw_summaries(
                env.plots,
                summaries,
                committed_points=index + 1,
                directory=iteration.dir,
            )
    env.pbar("flux points", total=len(plan.fluxes)).set_progress(len(plan.fluxes))
    draw_summaries(
        env.plots, summaries, committed_points=len(plan.fluxes), directory=directory
    )
