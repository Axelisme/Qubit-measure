"""Measure GUI attachment for coarse pulse-readout resonance discovery."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.analyze_params import ParamMeta
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from .core import (
    DEFAULT_MIN_SEPARATION_MHZ,
    RangeFindCfg,
    RangeFindExp,
    RangeFindOptions,
    RangeFindResult,
    ResonatorDip,
)
from .form import make_form, resonator_frequencies

RangeFindRun: TypeAlias = RunRecord[RangeFindCfg, RangeFindResult]


@dataclass
class RangeFindParams:
    """Enter frequency-ordered resonator names as a JSON list for coarse analysis.

    resonators is JSON text such as '["R2", "R1"]', or '[]' for unnamed discovery.
    min_separation is positive finite MHz, default 1 MHz as in core Options.
    to_options rejects invalid JSON and delegates the domain checks to core.
    """

    resonators: Annotated[str, ParamMeta(label="Resonators (JSON)")] = "[]"
    min_separation: Annotated[float, ParamMeta(label="Min separation (MHz)")] = (
        DEFAULT_MIN_SEPARATION_MHZ
    )

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
        options = RangeFindOptions(names, self.min_separation)
        options.validate()
        return options


class ResonatorDipSummary(BaseModel):
    """One frequency-ordered dip; null scalars retain unavailable estimates."""

    name: str | None = Field(
        description="Analysis name, or null for unnamed discovery; not a hardware channel."
    )
    freq: float | None = Field(description="Absolute dip frequency in MHz.")
    depth: float | None = Field(
        description="Background-relative magnitude depth in a.u."
    )


class RangeFindSummary(BaseModel):
    """Dip estimates and captured identity/writeback limitations."""

    resonators: list[ResonatorDipSummary] = Field(
        description="Dips in ascending frequency order."
    )
    writeback_diagnostic: str | None = Field(
        description="Reason some proposals were omitted; null when none were omitted."
    )
    identity_diagnostic: str | None = Field(
        description="Pairing uncertainty; null is not proof of identity."
    )


@dataclass
class RangeFindAnalysis(AnalyzeResultBase):
    """Frequency-ordered dips and captured writeback eligibility.

    dips contains optional name, absolute MHz and magnitude prominence depth.
    Unnamed discovery has no writable names or identity diagnostic.
    writable contains only names present in the analysis capture. diagnostic
    explains omitted frequency proposals; None means every dip is writable.
    identity_diagnostic warns that dip/component pairing needs confirmation when
    captured frequencies do not establish strictly ascending name order. None
    means that ordering has a finite captured frequency basis, not proven identity.
    """

    dips: tuple[ResonatorDip, ...]
    writable: tuple[str, ...]
    diagnostic: str | None
    identity_diagnostic: str | None

    def summary(self) -> RangeFindSummary:
        """Return the typed dip summary consumed by recipes, in native units."""
        return RangeFindSummary(
            resonators=[
                ResonatorDipSummary(name=dip.name, freq=dip.freq, depth=dip.depth)
                for dip in self.dips
            ],
            writeback_diagnostic=self.diagnostic,
            identity_diagnostic=self.identity_diagnostic,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return self.summary().model_dump()


def _get_analyze_params(result: RangeFindRun, ctx: SessionEnv) -> RangeFindParams:
    """Start Run/Load analysis with unnamed discovery and default MHz spacing.

    result and ctx do not seed names. The operator may name dips after
    inspecting the discovery plot, without changing the acquisition cfg.
    """
    del result, ctx
    return RangeFindParams()


def _analyze(
    req: AnalyzeRequest[RangeFindRun, RangeFindParams],
    *,
    plots: Plots,
) -> RangeFindAnalysis:
    """Discover unnamed dips, or find all requested named dips without partial success.

    req supplies raw IQ, options and the fixed Run/Load parameter capture.
    Empty names return ordered dips without proposals or identity claims.
    plots receives the annotated fit. Missing capture/components still allow pure
    calculation but omit their proposals and explain why in the summary.
    Only finite, strictly ascending captured frequencies establish name order.
    Otherwise summary and proposals flag unconfirmed pairing without blocking
    analysis or reordering names. Set but invalid captured frequencies, invalid
    curves/options and insufficient dips raise ValueError.
    """
    options = req.analyze_params.to_options()
    dips = RangeFindExp().analyze(req.run_result, options, plots=plots)
    if not options.resonators:
        return RangeFindAnalysis(dips, (), None, None)
    components = (
        req.parameters.document().components if req.parameters is not None else {}
    )
    frequencies = (
        resonator_frequencies(
            req.parameters,
            [name for name in options.resonators if name in components],
        )
        if req.parameters is not None
        else None
    )
    identity_diagnostic = (
        "dip 與元件身分配對待確認"
        if frequencies is None
        or len(frequencies) != len(options.resonators)
        or any(
            left >= right
            for left, right in zip(frequencies, frequencies[1:], strict=False)
        )
        else None
    )
    if req.parameters is None:
        return RangeFindAnalysis(
            dips,
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
    return RangeFindAnalysis(dips, tuple(writable), diagnostic, identity_diagnostic)


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
                f"Coarse dip frequency (MHz), depth {dip.depth:.6g} a.u."
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
    analysis=Analysis(
        fn=_analyze,
        options=_get_analyze_params,
        writeback=_get_writeback_items,
    ),
)
