"""Measure GUI attachment for coarse single-tone mux resonance discovery."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

from qick import QickConfig
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
    MuxOnetoneFindCfg,
    MuxOnetoneFindExp,
    MuxOnetoneFindOptions,
    MuxOnetoneFindResult,
    ResonatorDip,
    analyze,
    prepare_frequencies,
)
from .form import make_form

MuxOnetoneFindRun: TypeAlias = RunRecord[MuxOnetoneFindCfg, MuxOnetoneFindResult]


@dataclass
class MuxOnetoneFindParams:
    """Enter frequency-ordered resonator names as a JSON list for coarse analysis.

    resonators is JSON text such as '["R2", "R1"]', not a comma-separated list.
    min_separation is positive finite MHz, default 1 MHz as in core Options.
    to_options rejects invalid JSON and delegates the domain checks to core.
    """

    resonators: Annotated[str, ParamMeta(label="Resonators (JSON)")]
    min_separation: Annotated[float, ParamMeta(label="Min separation (MHz)")] = (
        DEFAULT_MIN_SEPARATION_MHZ
    )

    def to_options(self) -> MuxOnetoneFindOptions:
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
        options = MuxOnetoneFindOptions(names, self.min_separation)
        options.validate()
        return options


@dataclass
class MuxOnetoneFindAnalysis(AnalyzeResultBase):
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


class MuxOnetoneFindAdapter(
    BaseAdapter[
        MuxOnetoneFindCfg,
        MuxOnetoneFindRun,
        MuxOnetoneFindAnalysis,
        MuxOnetoneFindParams,
    ]
):
    """Find coarse resonator frequencies and propose only captured scalar freq rows."""

    exp_cls = MuxOnetoneFindExp
    experiment_tag: ClassVar[str | None] = MuxOnetoneFindExp.persistence_tag()
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[MuxOnetoneFindCfg]:
        """Copy the captured mux readout using the shared ringup-aware initial values."""
        return make_form(ctx)

    def validate_run_request(self, req: RunRequest, cfg: MuxOnetoneFindCfg) -> None:
        """Reject out-of-band absolute frequencies before Run, without hardware I/O.

        req.soccfg must be the cached QickConfig; missing/incompatible handles
        raise. cfg supplies full readout settings and the original tone index.
        """
        if not isinstance(req.soccfg, QickConfig):
            raise TypeError("mux/onetone_find requires a cached QickConfig")
        prepare_frequencies(cfg, soccfg=req.soccfg)

    def run(
        self, req: RunRequest, cfg: MuxOnetoneFindCfg, *, context: RunContext
    ) -> MuxOnetoneFindRun:
        """Acquire one spectrum and pair it with the unchanged complete cfg.

        context supplies SoC, cached config, cancellation and live plots. req is
        unused after preflight. Stop preserves NaNs for unacquired points; the
        host checks cancellation errors before publishing a successful result.
        """
        del req
        return RunRecord(cfg=cfg, result=MuxOnetoneFindExp().run(cfg, context=context))

    def get_analyze_params(
        self, result: MuxOnetoneFindRun, ctx: SessionEnv
    ) -> MuxOnetoneFindParams:
        """Seed JSON names from captured $mux.resonators sorted by captured MHz frequency.

        ctx is the result's fixed Run/Load environment, not the current Form.
        Missing capture or unbound mux starts with an empty editable list; fill
        names before Analyze. If any captured frequency is null, the list is
        also empty: enter names in increasing dip-frequency order instead of
        guessing the component identity. Malformed bindings/frequencies raise.
        result is unused because names are not stored in raw cfg.
        """
        del result
        parameters = ctx.parameters
        if parameters is None:
            return MuxOnetoneFindParams(resonators="[]")
        roles = parameters.roles
        if roles is None or "mux" not in roles.names:
            return MuxOnetoneFindParams(resonators="[]")
        raw = parameters.value("$mux.resonators")
        options = MuxOnetoneFindOptions(resonators=[])
        if not isinstance(raw, list) or not raw:
            raise ValueError("$mux.resonators must be a nonempty list of names")
        names: list[str] = []
        for name in raw:
            if not isinstance(name, str):
                raise ValueError("$mux.resonators must contain only strings")
            names.append(name)
        options.resonators = names
        options.validate()
        captured_frequencies = {
            name: parameters.value(f"{name}.freq") for name in names
        }
        frequencies = {
            name: _captured_frequency(value, name)
            for name, value in captured_frequencies.items()
            if value is not None
        }
        if len(frequencies) != len(names):
            return MuxOnetoneFindParams(resonators="[]")
        return MuxOnetoneFindParams(
            resonators=json.dumps(sorted(names, key=frequencies.__getitem__))
        )

    def analyze(
        self,
        req: AnalyzeRequest[MuxOnetoneFindRun, MuxOnetoneFindParams],
        *,
        plots: Plots,
    ) -> MuxOnetoneFindAnalysis:
        """Find all requested dips or fail the entire analysis without proposals.

        req supplies raw IQ, options and the fixed Run/Load parameter capture.
        plots receives the annotated fit. Missing capture/names still allow pure
        calculation but omit their proposals and explain why in the summary.
        Invalid curves/options and insufficient dips raise ValueError.
        """
        dips = analyze(req.run_result, req.analyze_params.to_options(), plots=plots)
        if req.parameters is None:
            return MuxOnetoneFindAnalysis(
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
        return MuxOnetoneFindAnalysis(dips, tuple(writable), diagnostic)

    def get_writeback_items(
        self, req: WritebackRequest[MuxOnetoneFindRun, MuxOnetoneFindAnalysis]
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


def _captured_frequency(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name}.freq must be finite MHz to seed resonator order")
    return float(value)
