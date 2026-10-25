"""Shared-calibration coherence flux sweep with selectable independent lifetimes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from pydantic import BaseModel, ConfigDict, Field, model_validator
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
from zcu_lab.workflows.coherence_fluxdep.measurements import (
    EchoResult,
    LifetimeHistory,
    LifetimeKind,
    LifetimeResult,
    RamseyResult,
    history_guess,
    lifetime_guess,
    measure_echo,
    measure_ramsey,
)
from zcu_lab.workflows.coherence_fluxdep.pulses import (
    calibrated_pulses,
)
from zcu_lab.workflows.coherence_fluxdep.singleshot import (
    SingleshotInputs,
    SingleshotResult,
    SingleshotTunables,
    singleshot_cfg,
    singleshot_skip_reason,
)
from zcu_lab.workflows.coherence_fluxdep.stages.calibration import calibrate_point
from zcu_lab.workflows.coherence_fluxdep.stages.t1 import T1Result, measure_t1
from zcu_lab.workflows.coherence_fluxdep.t1 import T1Plan, T1Tunables
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
    calibration_source,
    current_pi_calibration,
    current_point_calibration,
)
from zcu_lab.workflows.common.devices import set_flux
from zcu_lab.workflows.common.frequency_tracking import FrequencyModel, FrequencyTracker
from zcu_lab.workflows.common.saving import save_native
from zcu_lab.workflows.common.stages import StageEnv
from zcu_lab.workflows.common.summary import (
    RunSummary,
    SummaryAccumulator,
    observe_operation,
)
from zcu_lab.workflows.common.summary_points import coherence_summary_point


class CoherencePlan(T1Plan):
    """Choose flux points and lifetimes to map relaxation and dephasing.

    seed_flux is the seed pulse's calibration setpoint in flux_unit; None
    disables matrix-element feed-forward. probe_pulse supplies independent
    spectroscopy waveform/gain, while the
    inherited qub_pulse remains the Rabi and downstream pi drive.
    Inherited fields retain T1Plan's explicit unit, resolved native routing and
    delay resolution. measure_t1/measure_t2ramsey/measure_t2echo are fixed bools,
    default true. At least one must be true; all false raises before acquisition.
    Only enabled lifetimes acquire/record/save, in T1, Ramsey, Echo order.
    measure_singleshot defaults false; when true, eligible points acquire raw
    GE shots after calibration and before lifetimes, without fitting.
    """

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
    t1/t2ramsey/t2echo hold each experiment's named result, or None when disabled
    or not reached. Failed/skipped enabled items retain missing-value semantics. Native references are not an
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
    t1: T1Result | None = None
    t2ramsey: RamseyResult | None = None
    t2echo: EchoResult | None = None
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
        plots.save(kind, directory / f"{kind}-decay.png", figsize=(6.4, 9.6))


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


def _append(
    history: LifetimeHistory, flux: float, result: T1Result | LifetimeResult
) -> None:
    """Keep only accepted lifetime values in this lifetime's delay-grid history."""
    if result.status == "accepted":
        if result.lifetime_us is None or result.error_us is None:
            raise ValueError("Accepted lifetime requires microseconds and error")
        history.append((flux, result.lifetime_us, result.error_us))


async def run_singleshot_stage(
    iteration: Scope,
    plan: CoherencePlan,
    tun: SingleshotTunables,
    point_record: CoherencePoint,
    inputs: SingleshotInputs,
    summary: SummaryAccumulator | None = None,
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
    observe_operation(summary, ge)
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
# tolerated(radon): Keep the point's stage sequence together for lab authors.
async def coherence_fluxdep(  # noqa: C901, PLR0912, PLR0915  # tolerated: workflow flow reads top to bottom
    env: WorkflowEnv[Calibrated],
    plan: CoherencePlan,
    tun: Live[CoherenceTunables],
) -> RunSummary:
    """Measure coherence versus flux with independent lifetime histories.

    Await calibration, optional raw GE, then enabled T1, Ramsey and Echo. GE
    requires accepted Rabi and absolute flux distance >= singleshot.min_spacing
    from the last Completed GE, including partial. GE saves raw only and uses
    T1's history-based relax timing. Rabi frequency is NOW; T1/GE require this
    point's accepted pi, independent of pi2 trust. Ramsey/Echo require this
    point's accepted pi/pi2, with producer coordinates in calibration_sources.
    Nonaccepted readout retains legal prior values.
    Save each completed trace before fitting and commit one CoherencePoint per
    setpoint. Each accepted lifetime updates only its own delay-grid history.
    Unexpected adapter/writer/fit/plot errors propagate. Pause resumes the pending
    operation, not the whole point. Tunables reads use tun.current.
    Calibration returns its aggregate only on normal completion. An unexpected
    Rabi exception after frequency success leaves frequency's h5 on disk but
    no frequency result in this point record; the exception still fails the run.
    To revisit past points, start a new run with a sliced plan and seed local
    history from the previous run using the saved_run reader.
    """
    summary = SummaryAccumulator(len(plan.fluxes), plan.flux_unit)
    stage_env = StageEnv(plots=env.plots, snapshot=plan.snapshot, summary=summary)
    live_calibration = Live(lambda: tun.current.cal)
    carry = CalCarry(
        env.context,
        seed_flux=plan.seed_flux,
        frequency_tracker=FrequencyTracker(plan.predictor.make_predictor()),
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
            point_record.singleshot = SingleshotResult(
                "skipped",
                "Not reached (point cancelled or interrupted)"
                if plan.measure_singleshot
                else "Singleshot disabled",
                None,
            )
            await set_flux(
                iteration, plan.flux_dev, point_record.flux, unit=plan.flux_unit
            )
            if iteration.cancelled:
                point_record.status = "skipped"
                point_record.readout_source = calibration_source(
                    carry.readout_source, index
                )
                continue
            estimate = (
                history_guess(t1_by_flux[0][1], t1_by_flux[1:])
                if t1_by_flux
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
                summary=summary,
            )
            calibration_result = await calibrate_point(
                iteration,
                env.plots,
                captured,
                live_calibration,
                carry=carry,
                t1_estimate_us=estimate,
            )
            point_record.frequency = calibration_result.frequency
            point_record.rabi = calibration_result.rabi
            point_record.readout = calibration_result.readout
            if (
                point_record.readout is not None
                and point_record.readout.status != "skipped"
            ):
                point_record.calibration_sources["readout"] = CalibrationSource(
                    need="now", source_index=index, source_flux=flux, fallback=False
                )
            # Every pulse consumer requires current Rabi; only T2 needs pi2.
            now_cal = current_pi_calibration(point_record.rabi, carry)
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
                    summary=summary,
                )
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
            pulses = calibrated_pulses(plan, now_cal)
            if pulses is None:
                # T1, Ramsey and Echo all drive this point's pi; record each enabled skip.
                if plan.measure_t1:
                    point_record.t1 = T1Result(
                        status="skipped",
                        reason="Requires this point's accepted pi pulse (NOW)",
                    )
                reason = "Requires this point's accepted pi/pi2 pulse (NOW)"
                if plan.measure_t2ramsey:
                    point_record.t2ramsey = RamseyResult(
                        status="skipped", reason=reason
                    )
                if plan.measure_t2echo:
                    point_record.t2echo = EchoResult(status="skipped", reason=reason)
                point_record.status = "completed"
                directory = iteration.dir
                continue
            cal = current_point_calibration(point_record.rabi, carry)
            # T1 needs current pi, not pi2 or a prior pair.
            if plan.measure_t1:
                point_record.calibration_sources["t1"] = CalibrationSource(
                    need="now", source_index=index, source_flux=flux, fallback=False
                )
                t1_result = await measure_t1(
                    iteration,
                    stage_env,
                    pulses,
                    tun.current.t1,
                    points=plan.points,
                    guess_us=lifetime_guess(tun.current.t1.guess_us, t1_by_flux),
                )
                point_record.t1 = t1_result
                # Accepted T1 values guide the next point's scan window and relax delay.
                if t1_result.status == "accepted":
                    # The stage guarantees both scalars on acceptance.
                    assert t1_result.lifetime_us is not None
                    assert t1_result.error_us is not None
                    t1_by_flux.append((flux, t1_result.lifetime_us, t1_result.error_us))
                if iteration.cancelled:
                    point_record.status = "skipped"
                    continue
            # A current pi without a trusted pi2 still permits T1, but not T2.
            if cal is None:
                reason = "Requires this point's accepted pi/pi2 pulse (NOW)"
                if plan.measure_t2ramsey:
                    point_record.t2ramsey = RamseyResult(
                        status="skipped", reason=reason
                    )
                if plan.measure_t2echo:
                    point_record.t2echo = EchoResult(status="skipped", reason=reason)
                point_record.status = "completed"
                directory = iteration.dir
                continue
            # Prefer current accepted T1; failed/disabled T1 uses its EWMA/seed.
            t1_estimate = (
                point_record.t1.lifetime_us
                if point_record.t1 is not None
                and point_record.t1.status == "accepted"
                and point_record.t1.lifetime_us is not None
                else history_guess(tun.current.t1.guess_us, t1_by_flux)
            )
            # Measure Ramsey dephasing with its own grid and detuning policy.
            if plan.measure_t2ramsey:
                point_record.calibration_sources["t2ramsey"] = CalibrationSource(
                    need="now", source_index=index, source_flux=flux, fallback=False
                )
                point_record.t2ramsey = await measure_ramsey(
                    iteration,
                    t2ramsey_cfg(
                        plan,
                        tun.current.t2ramsey,
                        cal,
                        history_guess(tun.current.t2ramsey.guess_us, t2ramsey_by_flux),
                        t1_estimate_us=t1_estimate,
                    ),
                    tun=tun.current.t2ramsey,
                    snapshot=plan.snapshot,
                    plots=env.plots,
                    summary=summary,
                )
                _append(t2ramsey_by_flux, flux, point_record.t2ramsey)
                if iteration.cancelled:
                    point_record.status = "skipped"
                    continue
            # Measure echo coherence with fitted pi/2 pulses and its own grid.
            if plan.measure_t2echo:
                point_record.calibration_sources["t2echo"] = CalibrationSource(
                    need="now", source_index=index, source_flux=flux, fallback=False
                )
                point_record.t2echo = await measure_echo(
                    iteration,
                    t2echo_cfg(
                        plan,
                        tun.current.t2echo,
                        cal,
                        history_guess(tun.current.t2echo.guess_us, t2echo_by_flux),
                        t1_estimate_us=t1_estimate,
                    ),
                    tun=tun.current.t2echo,
                    snapshot=plan.snapshot,
                    plots=env.plots,
                    summary=summary,
                )
                _append(t2echo_by_flux, flux, point_record.t2echo)
                if iteration.cancelled:
                    point_record.status = "skipped"
                    continue
            point_record.status = "completed"
            directory = iteration.dir
        finally:
            iteration.commit(point_record)
            summary.points.append(coherence_summary_point(point_record, plan))
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
    return summary.finish()
