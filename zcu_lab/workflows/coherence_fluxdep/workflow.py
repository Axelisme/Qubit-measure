"""Shared-calibration coherence flux sweep with selectable independent lifetimes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zcu_tools.experiment.workflows import (
    Live,
    WorkflowEnv,
    workflow,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2.modules import PulseCfg

from zcu_lab.components.qubit import QubitFluxSchema
from zcu_lab.workflows.coherence_fluxdep.history import (
    LifetimeHistory,
    LifetimeKind,
    lifetime_guess,
)
from zcu_lab.workflows.coherence_fluxdep.pulses import calibrated_pulses
from zcu_lab.workflows.coherence_fluxdep.samples import append_point_sample
from zcu_lab.workflows.coherence_fluxdep.singleshot import SingleshotTunables
from zcu_lab.workflows.coherence_fluxdep.stages.echo import EchoResult, measure_echo
from zcu_lab.workflows.coherence_fluxdep.stages.ge import SingleshotResult, measure_ge
from zcu_lab.workflows.coherence_fluxdep.stages.ramsey import (
    RamseyResult,
    measure_ramsey,
)
from zcu_lab.workflows.coherence_fluxdep.stages.t1 import T1Result, measure_t1
from zcu_lab.workflows.coherence_fluxdep.stages.t2_support import T2Guess
from zcu_lab.workflows.coherence_fluxdep.t1 import T1Plan, T1Tunables
from zcu_lab.workflows.coherence_fluxdep.t2 import T2EchoTunables, T2RamseyTunables
from zcu_lab.workflows.common.calibration import (
    CalCarry,
    Calibrated,
    CalibrationSource,
    CalibrationTunables,
    FrequencyResult,
    RabiResult,
    ReadoutResult,
    calibration_source,
    current_pi_calibration,
    current_point_calibration,
)
from zcu_lab.workflows.common.calibration_stage_types import (
    CalibrationPoint,
    CalibrationPulses,
    PointGuess,
)
from zcu_lab.workflows.common.devices import set_flux
from zcu_lab.workflows.common.frequency_tracking import FrequencyModel, FrequencyTracker
from zcu_lab.workflows.common.point_calibration import calibrate_point
from zcu_lab.workflows.common.stages import StageEnv
from zcu_lab.workflows.common.summary import (
    RunSummary,
    SummaryAccumulator,
)
from zcu_lab.workflows.common.summary_points import coherence_summary_point


class CoherencePlan(T1Plan):
    """Choose flux points and lifetimes to map relaxation and dephasing.

    flux_frame is the qubit's Start-captured device calibration block in native
    units. It remains fixed for all points and record exports; {} means this
    source has no calibration, never a missing capture.
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

    flux_frame: QubitFluxSchema = Field(
        description="Start-captured qubit flux block in native device units; empty means no calibration"
    )
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
    result_path: Path | None = Field(
        default=None,
        description="Exact entry Result path captured at Start. None means no entry (e.g. headless); skip auto samples and use offline --result later.",
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
    Each lifetime and GE result stores its own gain_clamped DAC changes.
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
    setpoint. After commit, append a sample to the captured entry's record table
    when result_path is set; append failure propagates without undoing the commit.
    Each accepted lifetime updates only its own delay-grid history.
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
    calibration_pulses = CalibrationPulses(
        qub=plan.qub_pulse, probe=plan.probe_pulse, readout=plan.readout
    )
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
                status="skipped",
                reason="Not reached (point cancelled or interrupted)"
                if plan.measure_singleshot
                else "Singleshot disabled",
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
                lifetime_guess(t1_by_flux[0][1], t1_by_flux[1:])
                if t1_by_flux
                else tun.current.t1.guess_us
            )
            calibration_result = await calibrate_point(
                iteration,
                stage_env,
                calibration_pulses,
                live_calibration,
                guess=PointGuess(
                    point=CalibrationPoint(
                        index=point_record.index,
                        flux=point_record.flux,
                        flux_unit=plan.flux_unit,
                    ),
                    relax_us=estimate,
                ),
                carry=carry,
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
            pulses = calibrated_pulses(plan, now_cal)
            # GE eligibility and spacing history belong to the point loop.
            if plan.measure_singleshot and not iteration.cancelled:
                if point_record.rabi is None:
                    raise ValueError(
                        "Singleshot stage requires this point's Rabi result"
                    )
                ge_reason = None
                if point_record.rabi.status != "accepted":
                    ge_reason = "Requires this point's accepted Rabi"
                elif (
                    last_ge_flux is not None
                    and abs(flux - last_ge_flux) < tun.current.singleshot.min_spacing
                ):
                    ge_reason = "Minimum singleshot flux spacing"
                if ge_reason is not None:
                    point_record.singleshot = SingleshotResult(
                        status="skipped", reason=ge_reason
                    )
                else:
                    if pulses is None:
                        raise ValueError(
                            "Eligible singleshot requires current accepted calibration"
                        )
                    point_record.calibration_sources["singleshot"] = CalibrationSource(
                        need="now", source_index=index, source_flux=flux, fallback=False
                    )
                    point_record.singleshot = await measure_ge(
                        iteration,
                        stage_env,
                        pulses,
                        tun.current.singleshot,
                        relax_delay_us=tun.current.t1.relax_factor
                        * lifetime_guess(tun.current.t1.guess_us, t1_by_flux),
                    )
                    # Every Completed, including partial, advances GE spacing.
                    if point_record.singleshot.file is not None:
                        last_ge_flux = flux
            if iteration.cancelled:
                point_record.status = "skipped"
                continue
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
            pulses = calibrated_pulses(plan, cal)
            assert pulses is not None
            # Prefer current accepted T1; failed/disabled T1 uses its EWMA/seed.
            t1_estimate = (
                point_record.t1.lifetime_us
                if point_record.t1 is not None
                and point_record.t1.status == "accepted"
                and point_record.t1.lifetime_us is not None
                else lifetime_guess(tun.current.t1.guess_us, t1_by_flux)
            )
            # Measure Ramsey dephasing with its own grid and detuning policy.
            if plan.measure_t2ramsey:
                point_record.calibration_sources["t2ramsey"] = CalibrationSource(
                    need="now", source_index=index, source_flux=flux, fallback=False
                )
                ramsey_result = await measure_ramsey(
                    iteration,
                    stage_env,
                    pulses,
                    tun.current.t2ramsey,
                    points=plan.points,
                    guess=T2Guess(
                        lifetime_us=lifetime_guess(
                            tun.current.t2ramsey.guess_us, t2ramsey_by_flux
                        ),
                        t1_us=t1_estimate,
                    ),
                )
                point_record.t2ramsey = ramsey_result
                if ramsey_result.status == "accepted":
                    assert ramsey_result.lifetime_us is not None
                    assert ramsey_result.error_us is not None
                    t2ramsey_by_flux.append(
                        (flux, ramsey_result.lifetime_us, ramsey_result.error_us)
                    )
                if iteration.cancelled:
                    point_record.status = "skipped"
                    continue
            # Measure echo coherence with fitted pi/2 pulses and its own grid.
            if plan.measure_t2echo:
                point_record.calibration_sources["t2echo"] = CalibrationSource(
                    need="now", source_index=index, source_flux=flux, fallback=False
                )
                echo_result = await measure_echo(
                    iteration,
                    stage_env,
                    pulses,
                    tun.current.t2echo,
                    points=plan.points,
                    guess=T2Guess(
                        lifetime_us=lifetime_guess(
                            tun.current.t2echo.guess_us, t2echo_by_flux
                        ),
                        t1_us=t1_estimate,
                    ),
                )
                point_record.t2echo = echo_result
                if echo_result.status == "accepted":
                    assert echo_result.lifetime_us is not None
                    assert echo_result.error_us is not None
                    t2echo_by_flux.append(
                        (flux, echo_result.lifetime_us, echo_result.error_us)
                    )
                if iteration.cancelled:
                    point_record.status = "skipped"
                    continue
            point_record.status = "completed"
            directory = iteration.dir
        finally:
            iteration.commit(point_record)
            append_point_sample(env.identity.run_id, plan, point_record, iteration.dir)
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
