"""Measure GUI attachment for coarse pulse-readout resonance discovery."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.analyze_params import ParamMeta
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
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
from .form import make_form

RangeFindRun: TypeAlias = RunRecord[RangeFindCfg, RangeFindResult]


@dataclass
class RangeFindParams:
    """Enter frequency-ordered resonator names as a JSON list for coarse analysis.

    resonators is JSON text such as '["R2", "R1"]', not a comma-separated list.
    min_separation is positive finite MHz, default 1 MHz as in core Options.
    to_options rejects invalid JSON and delegates the domain checks to core.
    """

    resonators: Annotated[str, ParamMeta(label="Resonators (JSON)")]
    min_separation: Annotated[float, ParamMeta(label="Min separation (MHz)")] = (
        DEFAULT_MIN_SEPARATION_MHZ
    )

    def to_options(self) -> RangeFindOptions:
        """Parse JSON names and return validated core Options, or raise ValueError.

        Blank/malformed JSON, non-list values and non-string elements fail with
        input guidance. Core rejects empty/duplicate names and invalid spacing.
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


@dataclass
class RangeFindAnalysis(AnalyzeResultBase):
    """Frequency-ordered dips and captured writeback eligibility.

    dips contains name, absolute MHz and background-relative magnitude depth.
    writable contains only names present in the analysis capture. diagnostic
    explains omitted frequency proposals; None means every dip is writable.
    """

    dips: tuple[ResonatorDip, ...]
    writable: tuple[str, ...]
    diagnostic: str | None

    def to_summary_dict(self) -> dict[str, object]:
        """Show each name, MHz frequency and depth, plus omitted-proposal reasons."""
        return {
            "resonators": [
                {"name": dip.name, "freq": dip.freq, "depth": dip.depth}
                for dip in self.dips
            ],
            "writeback_diagnostic": self.diagnostic,
        }


class RangeFindAdapter(
    BaseAdapter[
        RangeFindCfg,
        RangeFindRun,
        RangeFindAnalysis,
        RangeFindParams,
    ]
):
    """Find coarse resonator frequencies and propose only captured scalar freq rows."""

    exp_cls = RangeFindExp
    experiment_tag: ClassVar[str | None] = RangeFindExp.persistence_tag()
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[RangeFindCfg]:
        """Build freq linear defaults and editable captured analysis names."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: RangeFindCfg, *, context: RunContext
    ) -> RangeFindRun:
        """Acquire one spectrum and pair it with the unchanged acquisition cfg and name list.

        context supplies SoC, cached config, cancellation and live plots. req is
        unused. Stop preserves NaNs for unacquired points; the
        host checks cancellation errors before publishing a successful result.
        """
        del req
        return RunRecord(cfg=cfg, result=RangeFindExp().run(cfg, context=context))

    def get_analyze_params(
        self, result: RangeFindRun, ctx: SessionEnv
    ) -> RangeFindParams:
        """Seed editable JSON names from the Run/Load record, never the current Form.

        result.cfg owns the saved analysis list. A missing/incompatible cfg starts
        empty so the operator can enter names for data-only analysis. ctx supplies
        the fixed environment but does not override the saved list.
        """
        del ctx
        names = result.cfg.resonators if result.cfg is not None else []
        return RangeFindParams(resonators=json.dumps(names))

    def analyze(
        self,
        req: AnalyzeRequest[RangeFindRun, RangeFindParams],
        *,
        plots: Plots,
    ) -> RangeFindAnalysis:
        """Find all requested dips or fail the entire analysis without proposals.

        req supplies raw IQ, options and the fixed Run/Load parameter capture.
        plots receives the annotated fit. Missing capture/names still allow pure
        calculation but omit their proposals and explain why in the summary.
        Invalid curves/options and insufficient dips raise ValueError.
        """
        dips = RangeFindExp().analyze(
            req.run_result, req.analyze_params.to_options(), plots=plots
        )
        if req.parameters is None:
            return RangeFindAnalysis(
                dips, (), "No Run/Load parameter capture; no frequency proposals"
            )
        writable: list[str] = []
        missing: list[str] = []
        components = req.parameters.document().components
        for dip in dips:
            if dip.name in components:
                writable.append(dip.name)
            else:
                missing.append(dip.name)
        diagnostic = (
            f"Names absent from capture: {', '.join(missing)}; their proposals omitted"
            if missing
            else None
        )
        return RangeFindAnalysis(dips, tuple(writable), diagnostic)

    def get_writeback_items(
        self, req: WritebackRequest[RangeFindRun, RangeFindAnalysis]
    ) -> Sequence[WritebackItem]:
        """Propose one independently selectable <name>.freq MHz row per captured dip.

        req pairs the accepted analysis with its fixed source. No module, kappa
        or electrical-delay rows are produced. This method never applies changes.
        """
        if req.ctx.parameters is None:
            return []
        return [
            ParameterWriteback(
                target_name=f"{dip.name}.freq",
                description=f"Coarse dip frequency (MHz), depth {dip.depth:.6g} a.u.",
                proposed_value=dip.freq,
            )
            for dip in req.analyze_result.dips
            if dip.name in req.analyze_result.writable
        ]
