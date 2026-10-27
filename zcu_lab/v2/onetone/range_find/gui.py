"""Measure GUI attachment for coarse pulse-readout resonance discovery."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

import numpy as np
from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    AnalyzeRequest,
    Interactive,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.analyze_params import ParamMeta
from zcu_tools.gui.app.measure.range_find_frontend import (
    draw_range_find,
    make_range_find_frontend,
)
from zcu_tools.gui.app.measure.range_find_plugin import (
    DipSelection,
    RangeFindPlugin,
    RangeFindState,
)
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from .core import (
    DEFAULT_MIN_SEPARATION_MHZ,
    RangeFindCfg,
    RangeFindExp,
    RangeFindOptions,
    RangeFindResult,
)
from .form import make_form, resonator_frequencies

RangeFindRun: TypeAlias = RunRecord[RangeFindCfg, RangeFindResult]


@dataclass
class RangeFindParams:
    """Seed the interactive guess and available labels from a JSON name list.

    resonators is JSON text such as '["R2", "R1"]', or '[]' for unnamed discovery.
    Names initially pair in frequency order; the picker can reassign them freely.
    min_separation is positive finite MHz, default 1 MHz as in core Options.
    smoothing_sigma is Gaussian sigma in samples, default zero. prominence_factor
    is the noise multiplier, default five. max_fwhm limits width in MHz, default 25.
    to_options rejects invalid JSON and delegates the domain checks to core.
    """

    resonators: Annotated[str, ParamMeta(label="Resonators (JSON)")] = "[]"
    min_separation: Annotated[float, ParamMeta(label="Min separation (MHz)")] = (
        DEFAULT_MIN_SEPARATION_MHZ
    )
    smoothing_sigma: Annotated[float, ParamMeta(label="Smoothing sigma (samples)")] = (
        0.0
    )
    prominence_factor: Annotated[float, ParamMeta(label="Prominence noise factor")] = (
        5.0
    )
    max_fwhm: Annotated[float, ParamMeta(label="Max FWHM (MHz)")] = 25.0

    def to_options(self) -> RangeFindOptions:
        """Parse JSON names and return validated core Options, or raise ValueError.

        Blank/malformed JSON, non-list values and non-string elements fail with
        input guidance. Core rejects blank/duplicate names and invalid spacing.
        No invalid value is replaced with a default or an empty list.
        """
        try:
            decoded: object = json.loads(self.resonators)
        except json.JSONDecodeError as error:
            raise ValueError(
                f'Resonators must be a JSON list such as ["R2", "R1"]: {error.msg}'
            ) from error
        if not isinstance(decoded, list):
            raise ValueError('Resonators must be a JSON list such as ["R2", "R1"]')
        names: list[str] = []
        for name in decoded:
            if not isinstance(name, str):
                raise ValueError("Resonators JSON list must contain only strings")
            names.append(name)
        options = RangeFindOptions(
            resonators=names,
            min_separation=self.min_separation,
            smoothing_sigma=self.smoothing_sigma,
            prominence_factor=self.prominence_factor,
            max_fwhm=self.max_fwhm,
        )
        options.validate()
        return options


class ResonatorDipSummary(BaseModel):
    """Decode one native dip; null scalars retain unavailable estimates."""

    name: str | None = Field(description="Analysis name, or null during discovery.")
    freq: float | None = Field(description="Absolute dip frequency in MHz.")
    depth: float | None = Field(description="Magnitude prominence depth in a.u.")
    fwhm: float | None = Field(description="Half-prominence dip width in MHz.")
    score: float | None = Field(description="Depth divided by FWHM in a.u./MHz.")


class RangeFindSummary(BaseModel):
    """Decode accepted dips, final picker state and captured writeback limitations."""

    dips: list[ResonatorDipSummary] = Field(
        description="Dips in ascending frequency order."
    )
    selection: RangeFindState = Field(
        description="Final numerical controls, candidates and selected points with component assignments."
    )
    writable: list[str] = Field(description="Names present in the analysis capture.")
    diagnostic: str | None = Field(
        description="Reason frequency proposals were omitted."
    )
    identity_diagnostic: str | None = Field(
        description="Pairing uncertainty; null is not proof of identity."
    )


@dataclass
class RangeFindAnalysis:
    """Accepted frequency-ordered dips, final state and writeback eligibility.

    selection retains the final controls, selected samples and component mapping.
    dips contains optional name, absolute MHz, magnitude prominence depth,
    half-prominence FWHM in MHz and depth / FWHM score in a.u./MHz.
    Manual points outside the current candidates have null depth/FWHM/score.
    Unnamed discovery has no writable names or identity diagnostic.
    writable contains only names present in the analysis capture. diagnostic
    explains omitted frequency proposals; None means every dip is writable.
    identity_diagnostic warns that dip/component pairing needs confirmation when
    captured frequencies do not establish strictly ascending name order. None
    means that ordering has a finite captured frequency basis, not proven identity.
    """

    dips: tuple[ResonatorDipSummary, ...]
    selection: RangeFindState
    writable: tuple[str, ...]
    diagnostic: str | None
    identity_diagnostic: str | None


def _get_analyze_params(result: RangeFindRun, ctx: SessionEnv) -> RangeFindParams:
    """Start Run/Load analysis with unnamed discovery and default MHz spacing.

    result and ctx do not seed names. The operator may name dips after
    inspecting the discovery plot, without changing the acquisition cfg.
    """
    del result, ctx
    return RangeFindParams()


def _make_interactive_plugin(
    req: AnalyzeRequest[RangeFindRun, RangeFindParams], *, plots: Plots
) -> RangeFindPlugin[RangeFindAnalysis]:
    """Seed one picker from captured IQ and Analyze controls, publishing only on Done.

    Names supply the available component labels and initial frequency-order pairing.
    The shared static finder supplies the initial guess. Invalid curves/options
    raise ValueError; an empty guess stays editable for manual selection.
    Final controls and assignments, not the initial form, determine saved results
    and writeback. Cancel leaves the previous accepted result unchanged.
    """
    options = req.analyze_params.to_options()
    result = req.run_result.result

    def build_result(state: RangeFindState) -> RangeFindAnalysis:
        analysis = _build_analysis(req, state)
        plots.subplots("fit", nrows=1, figsize=(9, 4.5))
        plots.draw(
            "fit",
            lambda figure: draw_range_find(
                figure, plugin.freqs, plugin.magnitude, state
            ),
        )
        return analysis

    plugin = RangeFindPlugin(
        np.asarray(result.freqs, dtype=np.float64),
        np.abs(np.asarray(result.signals, dtype=np.complex128)),
        components=tuple(options.resonators),
        options=options.to_dip_options(),
        build_result=build_result,
    )
    return plugin


def _summarize_dip(dip: DipSelection) -> ResonatorDipSummary:
    """Keep measured MHz and optional candidate metrics without inventing a fit."""
    return ResonatorDipSummary(
        name=dip.component,
        freq=dip.freq,
        depth=None if dip.estimate is None else dip.estimate.depth,
        fwhm=None if dip.estimate is None else dip.estimate.fwhm,
        score=None if dip.estimate is None else dip.estimate.score,
    )


def _build_analysis(
    req: AnalyzeRequest[RangeFindRun, RangeFindParams], state: RangeFindState
) -> RangeFindAnalysis:
    """Project accepted selections and resolve eligibility from the fixed capture."""
    dips = tuple(_summarize_dip(dip) for dip in state.selected)
    names = [dip.name for dip in dips if dip.name is not None]
    if not names:
        return RangeFindAnalysis(dips, state, (), None, None)
    components = (
        req.parameters.document().components if req.parameters is not None else {}
    )
    frequencies = (
        resonator_frequencies(
            req.parameters,
            [name for name in names if name in components],
        )
        if req.parameters is not None
        else None
    )
    identity_diagnostic = (
        "dip 與元件身分配對待確認"
        if frequencies is None
        or len(frequencies) != len(names)
        or any(
            left >= right
            for left, right in zip(frequencies, frequencies[1:], strict=False)
        )
        else None
    )
    if req.parameters is None:
        return RangeFindAnalysis(
            dips,
            state,
            (),
            "No Run/Load parameter capture; no frequency proposals",
            identity_diagnostic,
        )
    writable: list[str] = []
    missing: list[str] = []
    for dip in dips:
        if dip.name is None:
            continue
        if dip.name in components:
            writable.append(dip.name)
        else:
            missing.append(dip.name)
    diagnostic = (
        f"Names absent from capture: {', '.join(missing)}; their proposals omitted"
        if missing
        else None
    )
    return RangeFindAnalysis(
        dips, state, tuple(writable), diagnostic, identity_diagnostic
    )


def _get_writeback_items(
    req: WritebackRequest[RangeFindRun, RangeFindAnalysis],
) -> Sequence[WritebackItem]:
    """Propose one independently selectable <name>.freq MHz row per captured dip.

    req pairs the accepted analysis with its fixed source. No module, kappa
    or electrical-delay rows are produced. Unconfirmed dip/component pairing
    is carried from the analysis into every row description. The warning does
    not disable selection or Apply. This method never applies changes.
    """
    if req.ctx.parameters is None:
        return []
    return [
        ParameterWriteback(
            target_name=f"{dip.name}.freq",
            description=(
                "Coarse dip frequency (MHz)"
                + (
                    f", depth {dip.depth:.6g} a.u."
                    if dip.depth is not None
                    else ", manual sample"
                )
                + (
                    f"; {req.analyze_result.identity_diagnostic}"
                    if req.analyze_result.identity_diagnostic is not None
                    else ""
                )
            ),
            proposed_value=dip.freq,
        )
        for dip in req.analyze_result.dips
        if dip.name is not None and dip.name in req.analyze_result.writable
    ]


def run(
    cfg: RangeFindCfg, *, context: RunContext
) -> RunRecord[RangeFindCfg, RangeFindResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=RangeFindExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=RangeFindExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Interactive(
        plugin=_make_interactive_plugin,
        frontend=make_range_find_frontend,
        options=_get_analyze_params,
        writeback=_get_writeback_items,
    ),
)
