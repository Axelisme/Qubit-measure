"""Measure attachment for common-offset mux spectroscopy."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, JsonValue
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.cfg_model import MuxReadoutSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ModuleWriteback,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.mux.onetone.core import (
    MuxOnetoneAnalysis,
    MuxOnetoneAnalyzeOptions,
    MuxOnetoneCfg,
    MuxOnetoneExp,
    MuxOnetoneResult,
    ToneFitFailure,
    ToneFitSuccess,
    validate_analysis_source,
)
from zcu_lab.v2.mux.onetone.form import make_form
from zcu_lab.v2.onetone.freq.gui import OneToneFreqAnalyzeParams
from zcu_lab.v2.support.resonator import route_matched_delay

MuxOnetoneRunResult: TypeAlias = RunRecord[MuxOnetoneCfg, MuxOnetoneResult]


class MuxOnetoneFitSummary(BaseModel):
    """JSON-shaped resonance fit; params retains all native model fields.

    freq/fwhm use MHz, theta0 rad and edelay inverse MHz. params includes the
    selected model's scalars, complex {real,imag} objects and circle arrays.
    Nonfinite model values become null; fit_quality uses the fitting owner.
    """

    freq: float
    fwhm: float
    theta0: float
    edelay: float
    params: dict[str, JsonValue]
    fit_quality: FitQuality


class MuxOnetoneSuccessSummary(BaseModel):
    """A native successful outcome with original index and display-only label."""

    index: int
    label: str
    fit: MuxOnetoneFitSummary
    edelay_source: Literal["global", "calibrated", "manual"]
    edelay_persistable: bool


class MuxOnetoneFailureSummary(BaseModel):
    """A native failed outcome with its original index, label and fit reason."""

    index: int
    label: str
    reason: str


class MuxOnetoneSummary(BaseModel):
    """JSON-shaped native outcomes and complete captured resonator bindings.

    outcomes retains acquisition order, distinguishing fits from reason-only
    failures. resonators is the complete original-index binding or None.
    resonators_diagnostic explains absent scalar writeback, not module failure.
    """

    outcomes: list[MuxOnetoneSuccessSummary | MuxOnetoneFailureSummary]
    resonators: list[str] | None
    resonators_diagnostic: str | None


@dataclass
class MuxOnetoneAnalyzeResult(MuxOnetoneAnalysis):
    """Core outcomes with the GUI's complete Run/Load resonator capture.

    resonators contains the complete N bindings, or None when absent/mismatched.
    resonators_diagnostic explains unavailable scalar proposals without
    suppressing successful module proposals. No live context is retained.
    """

    resonators: tuple[str, ...] | None
    resonators_diagnostic: str | None


def _resonators(
    parameters: ParameterContext | None, count: int
) -> tuple[tuple[str, ...] | None, str | None]:
    if parameters is None:
        return None, "No Run/Load parameter capture; resonator parameters unavailable"
    lookup = parameters.lookup("$mux.resonators")
    raw = lookup.value
    if not lookup.present or raw is None:
        raise ValueError(
            "$mux.resonators is missing; expected an ordered list of component names"
        )
    if not isinstance(raw, list):
        raise ValueError("$mux.resonators must be an ordered list of component names")
    resonators: list[str] = []
    for resonator in raw:
        if not isinstance(resonator, str):
            raise ValueError(
                "$mux.resonators must be an ordered list of component names"
            )
        resonators.append(resonator)
    if len(raw) != count:
        return (
            None,
            f"$mux.resonators length {len(raw)} does not match complete readout tone count N={count}; parameter proposals omitted",
        )
    return tuple(resonators), None


def _analyze(
    req: AnalyzeRequest[MuxOnetoneRunResult, OneToneFreqAnalyzeParams], *, plots: Plots
) -> MuxOnetoneAnalyzeResult:
    """Fit each selected resonator at its saved base frequency plus MHz offset.

    req supplies measured IQ, the saved cfg, common fit options and captured
    resonator calibrations. plots receives a fit_i figure per original tone i.
    Return frequency, linewidth, phase and electrical delay, or a failure
    reason for each tone. Missing cfg, inconsistent data, invalid options or
    source errors raise for the whole analysis. One tone's numerical fit
    failure retains its reason without preventing the other fits.
    """
    # 1. Validate numeric inputs before resolving capture-dependent sources.
    cfg = validate_analysis_source(req.run_result)
    params = req.analyze_params
    options = MuxOnetoneAnalyzeOptions(
        model_type=params.model_type,
        fit_bg_amp_slope=params.fit_bg_amp_slope,
        fit_bg_phase_curvature=params.fit_bg_phase_curvature,
        edelay=params.edelay,
    )
    resonators, diagnostic = _resonators(req.parameters, len(cfg.readout.tones))

    # 2. Resolve route priors from the fixed capture, never the current point.
    priors: list[float | None] = []
    for index in req.run_result.result.tone_indices:
        prior = None
        if (
            resonators is not None
            and req.parameters is not None
            and params.edelay is None
        ):
            tone = cfg.readout.tones[index]
            raw = req.parameters.lookup(f"{resonators[index]}.ext.edelay")
            if raw.present:
                prior = route_matched_delay(raw.value, (cfg.readout.gen_ch, tone.ro_ch))
        priors.append(prior)
    analysis = MuxOnetoneExp().analyze(
        req.run_result, options, route_priors=priors, plots=plots
    )
    return MuxOnetoneAnalyzeResult(analysis.outcomes, resonators, diagnostic)


def _get_writeback_items(
    req: WritebackRequest[MuxOnetoneRunResult, MuxOnetoneAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose fitted frequencies without changing inactive or failed tones.

    req supplies the saved run and its analysis. Return a full readout module
    for $mux.readout.readout_rf, a following readout slot row when capture
    is available, and separate resonator calibration rows when bindings are
    available. The slot points to the captured module root; alias and source
    failures propagate. All-failed fits return no proposals; missing or
    inconsistent saved cfg raises. Proposals do not write until Apply.
    """
    # 1. Only successful fits can propose changes.
    cfg = validate_analysis_source(req.run_result)
    answer = req.analyze_result
    successes = [
        outcome for outcome in answer.outcomes if isinstance(outcome, ToneFitSuccess)
    ]
    if not successes:
        return []

    # 2. Replace successful frequencies in the complete readout, not a subset.
    readout = cfg.readout
    tones = list(readout.tones)
    changes: list[str] = []
    for outcome in answer.outcomes:
        if isinstance(outcome, ToneFitFailure):
            changes.append(
                f"tone {outcome.index} ({outcome.label}): failed: {outcome.reason}; frequency unchanged"
            )
        else:
            old = tones[outcome.index]
            tones[outcome.index] = old.with_updates(freq=outcome.fit.freq)
            changes.append(
                f"tone {outcome.index} ({outcome.label}): {old.freq:g} -> {outcome.fit.freq:g} MHz"
            )
    proposal = MuxReadoutSpec.model_validate(
        readout.with_updates(tones=tones).model_dump()
    )
    items: list[WritebackItem] = [
        ModuleWriteback(
            target_name="$mux.readout.readout_rf",
            description="; ".join(changes),
            proposal=proposal,
            role_id="readout",
        )
    ]

    if req.ctx.parameters is not None:
        items.append(
            ParameterWriteback(
                target_name="$mux.module.readout",
                description="Select fitted mux readout for downstream experiments",
                proposed_value=req.ctx.parameters.resolve_path(
                    "$mux.readout.readout_rf"
                ),
            )
        )

    # 3. Map scalar calibrations by original tone index to captured resonators.
    if answer.resonators is not None:
        for outcome in successes:
            resonator, fit = answer.resonators[outcome.index], outcome.fit
            for suffix, description, value in (
                ("freq", "Resonator frequency (MHz)", fit.freq),
                ("kappa", "Resonator linewidth FWHM (MHz)", fit.fwhm),
                ("ext.theta0", "Resonator circle phase offset (rad)", fit.theta0),
            ):
                items.append(
                    ParameterWriteback(
                        target_name=f"{resonator}.{suffix}",
                        description=description,
                        proposed_value=value,
                    )
                )
            if outcome.edelay_persistable:
                items.append(
                    ParameterWriteback(
                        target_name=f"{resonator}.ext.edelay",
                        description="Route-qualified resonator electrical-delay calibration",
                        proposed_value={
                            "edelay": fit.edelay,
                            "res_ch": readout.gen_ch,
                            "ro_ch": readout.tones[outcome.index].ro_ch,
                        },
                    )
                )
    return items


def run(
    cfg: MuxOnetoneCfg, *, context: RunContext
) -> RunRecord[MuxOnetoneCfg, MuxOnetoneResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=MuxOnetoneExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=MuxOnetoneExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: OneToneFreqAnalyzeParams(),
        writeback=_get_writeback_items,
    ),
)
