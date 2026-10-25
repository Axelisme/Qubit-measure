from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
from pydantic import BaseModel, Field, JsonValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.onetone.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
    HomophasalSamplingCfg,
)
from zcu_lab.v2.onetone.freq.form import FreqInput, HomophasalInput, make_form
from zcu_lab.v2.support.measure.writeback_helpers import (
    pulse_readout_module_writeback_items,
)
from zcu_lab.v2.support.resonator import finite_real, route_matched_delay

OneToneFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]
SamplingMode: TypeAlias = Literal["linear", "homophasal"]
EDelaySource: TypeAlias = Literal["global", "calibrated", "manual"]

_EDELAY_PARAMETER_PATH = "resonator.ext.edelay"
_DEFAULT_MAX_EDELAY_SEARCH_RADIUS = 100.0


@dataclass
class OneToneFreqAnalyzeParams:
    """Shared resonance fit controls for single-tone and mux analysis."""

    model_type: Annotated[Literal["hm", "t", "auto"], ParamMeta(label="Model type")] = (
        "hm"
    )
    """Resonance model: "hm" for hanger, "t" for transmission, or "auto".

    "auto" selects the model from the measured magnitude response.
    The default is "hm".
    """

    fit_bg_amp_slope: Annotated[bool, ParamMeta(label="Fit amplitude background")] = (
        True
    )
    """Fit the background's log-amplitude slope when True, the default.

    True fits a frequency-dependent exponential amplitude envelope.
    False keeps the log-amplitude slope at zero, giving a constant envelope.
    """

    fit_bg_phase_curvature: Annotated[bool, ParamMeta(label="Fit phase curvature")] = (
        False
    )
    """Fit quadratic background phase curvature when True.

    True fits a quadratic phase term alongside the electrical-delay phase.
    False, the default, keeps the quadratic coefficient at zero.
    """

    edelay: Annotated[
        float | None, ParamMeta(label="Electrical delay seed (us; blank = auto)")
    ] = None
    """Electrical-delay branch seed in us, default None, shown as blank.

    A finite number, including zero or a negative value, seeds the delay branch.
    Blank first uses a finite route-matched calibration as the seed.
    Otherwise automatic branch selection uses an internal 100 us search cap.
    Fitting starts from the selected branch and stores the fitted delay in the
    analysis result. The input field retains the supplied seed or blank state.
    Invalid seeds raise ValueError at analyze entry.
    """


class OnetoneSpectrumSummary(BaseModel):
    """Decode the framework's native result JSON; no experiment serialization."""

    freq: float | None = Field(description="Fitted resonator frequency in MHz.")
    fwhm: float | None = Field(description="Fitted linewidth FWHM in MHz.")
    params: dict[str, JsonValue] = Field(
        description="Native model parameters, with non-finite numbers represented by null."
    )
    edelay: float | None = Field(description="Fitted electrical delay in us.")
    edelay_source: EDelaySource = Field(
        description="Delay seed source: global, calibrated, or manual."
    )
    edelay_persistable: bool = Field(
        description="Whether the fitted delay satisfies the existing writeback gate."
    )


@dataclass
class OneToneFreqAnalyzeResult:
    freq: float
    fwhm: float
    params: dict[str, Any]
    edelay: float | None = None
    edelay_source: EDelaySource = "global"
    edelay_persistable: bool = False


def _readout_route(result: OneToneFreqRunResult) -> tuple[int, int] | None:
    cfg = result.cfg
    if cfg is None:
        return None
    res_ch = cfg.readout.gen_ch
    ro_ch = cfg.readout.ro_ch
    if (
        isinstance(res_ch, bool)
        or not isinstance(res_ch, Integral)
        or isinstance(ro_ch, bool)
        or not isinstance(ro_ch, Integral)
    ):
        return None
    return int(res_ch), int(ro_ch)


def _route_matched_prior(
    parameters: ParameterContext | None, route: tuple[int, int] | None
) -> float | None:
    if parameters is None or route is None:
        return None
    lookup = parameters.lookup(f"${_EDELAY_PARAMETER_PATH}")
    return route_matched_delay(lookup.value, route) if lookup.present else None


def _resolve_edelay_seed(
    params: OneToneFreqAnalyzeParams,
    parameters: ParameterContext | None,
    route: tuple[int, int] | None,
) -> tuple[float | None, EDelaySource]:
    if params.edelay is not None:
        seed = finite_real(params.edelay)
        if seed is None:
            raise ValueError("edelay must be a finite real scalar or None")
        return seed, "manual"

    prior = _route_matched_prior(parameters, route)
    if prior is not None:
        return prior, "calibrated"
    return None, "global"


def _has_nonuniform_fitting_grid(result: OneToneFreqRunResult) -> bool:
    fitting_freqs = result.result.freqs[1:-1]
    if len(fitting_freqs) < 3:
        return False
    steps = np.abs(np.diff(fitting_freqs))
    return not bool(np.allclose(steps, np.median(steps), rtol=1e-8, atol=0.0))


def _analyze(
    req: AnalyzeRequest[OneToneFreqRunResult, OneToneFreqAnalyzeParams],
    *,
    plots: Plots,
) -> OneToneFreqAnalyzeResult:
    params = req.analyze_params
    route = _readout_route(req.run_result)
    edelay_seed, edelay_source = _resolve_edelay_seed(params, req.parameters, route)
    max_search_radius = (
        _DEFAULT_MAX_EDELAY_SEARCH_RADIUS if edelay_seed is None else None
    )
    answer = FreqExp().analyze(
        req.run_result,
        FreqAnalyzeOptions(
            model_type=params.model_type,
            fit_bg_amp_slope=params.fit_bg_amp_slope,
            fit_bg_phase_curvature=params.fit_bg_phase_curvature,
            edelay_branch_seed=edelay_seed,
            edelay_max_search_radius=max_search_radius,
        ),
        plots=plots,
    )
    edelay = finite_real(answer.params.get("edelay"))
    if edelay is None:
        raise ValueError("one-tone fit result must contain a finite edelay")
    return OneToneFreqAnalyzeResult(
        freq=answer.freq,
        fwhm=answer.fwhm,
        params=answer.params,
        edelay=edelay,
        edelay_source=edelay_source,
        edelay_persistable=(
            route is not None
            and (
                edelay_seed is not None or _has_nonuniform_fitting_grid(req.run_result)
            )
        ),
    )


def _get_writeback_items(
    req: WritebackRequest[OneToneFreqRunResult, OneToneFreqAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose scalars and a readout from the saved pulse-readout cfg.

    A captured parameter context adds a following readout slot row pointing
    to the resolved module root. Missing cfg skips the module and slot;
    persistable delay requires a finite fitted delay and its saved route.
    Missing capture retains scalars and any module. Alias and source
    failures propagate.
    """
    result = req.analyze_result
    theta0 = result.params.get("theta0")
    if isinstance(theta0, bool) or not isinstance(theta0, Real):
        raise ValueError(
            "OneToneFreqAnalyzeResult.params must contain numeric 'theta0' "
            "for writeback"
        )
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$resonator.freq",
            description="Resonator frequency (MHz)",
            proposed_value=result.freq,
        ),
        ParameterWriteback(
            target_name="$resonator.kappa",
            description="Resonator linewidth FWHM (MHz)",
            proposed_value=result.fwhm,
        ),
        ParameterWriteback(
            target_name="$resonator.ext.theta0",
            description="Resonator circle phase offset (rad)",
            proposed_value=float(theta0),
        ),
    ]
    if result.edelay_persistable:
        route = _readout_route(req.run_result)
        edelay = finite_real(result.edelay)
        if route is None or edelay is None:
            raise ValueError(
                "persistable electrical delay requires a finite fitted edelay "
                "and pulse-readout route"
            )
        res_ch, ro_ch = route
        items.append(
            ParameterWriteback(
                target_name=f"${_EDELAY_PARAMETER_PATH}",
                description="Route-qualified resonator electrical-delay calibration",
                proposed_value={
                    "edelay": edelay,
                    "res_ch": res_ch,
                    "ro_ch": ro_ch,
                },
            )
        )
    module_items = pulse_readout_module_writeback_items(
        req.run_result.cfg,
        target="$resonator.readout.readout_rf",
        desc="Readout at fitted resonator frequency",
        field_updates=(("freq", float(result.freq)),),
        role_id="readout",
    )
    items.extend(module_items)
    if module_items and req.ctx.parameters is not None:
        items.append(
            ParameterWriteback(
                target_name="$resonator.module.readout",
                description="Select fitted readout for downstream experiments",
                proposed_value=req.ctx.parameters.resolve_path(
                    "$resonator.readout.readout_rf"
                ),
            )
        )
    return items


def run(inp: FreqInput, *, context: RunContext) -> RunRecord[FreqCfg, FreqResult]:
    """Acquire the selected grid and record its actual core sampling configuration."""
    sampling = inp.sampling
    homophasal = (
        HomophasalSamplingCfg(
            r_f=sampling.r_f, rf_w=sampling.rf_w, theta0=sampling.theta0
        )
        if isinstance(sampling, HomophasalInput)
        else None
    )
    cfg = FreqCfg(
        reset=inp.reset,
        readout=inp.readout,
        freq=inp.freq,
        reps=inp.reps,
        rounds=inp.rounds,
        relax_delay=inp.relax_delay,
        sampling_mode="homophasal" if homophasal is not None else "linear",
        homophasal=homophasal,
    )
    return RunRecord(cfg=cfg, result=FreqExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=FreqExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: OneToneFreqAnalyzeParams(),
        writeback=_get_writeback_items,
    ),
)
