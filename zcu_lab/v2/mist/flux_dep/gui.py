"""Measure adapter for the native MIST flux/gain map."""

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
    ParamMeta,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core import FluxDepAnalyzeOptions, FluxDepCfg, FluxDepExp, FluxDepResult
from .form import make_form

MistFluxDepRun: TypeAlias = RunRecord[FluxDepCfg, FluxDepResult]


@dataclass
class MistFluxDepParams:
    """Display-only calibration; None keeps raw device/gain coordinates.

    flux_half and flux_period use the acquisition device's native unit.
    ac_coeff maps probe gain squared to photon number. secondary_xaxis requires
    both flux calibration values. auto_range uses the full acquired endpoints.
    """

    flux_half: Annotated[float | None, ParamMeta(label="Half-flux device value")] = None
    flux_period: Annotated[
        float | None, ParamMeta(label="Flux period (device units)")
    ] = None
    ac_coeff: Annotated[float | None, ParamMeta(label="Photon coefficient")] = None
    secondary_xaxis: Annotated[bool, ParamMeta(label="Secondary device-value axis")] = (
        False
    )
    auto_range: Annotated[bool, ParamMeta(label="Full acquired range")] = True


@dataclass
class MistFluxDepAnalysis(AnalyzeResultBase):
    """Figure-only MIST analysis; no calibration or writeback is inferred."""


class MistFluxDepAdapter(
    BaseAdapter[FluxDepCfg, MistFluxDepRun, MistFluxDepAnalysis, MistFluxDepParams]
):
    """Acquire flux rows of gain-swept IQ and display their baseline difference."""

    exp_cls = FluxDepExp
    experiment_tag: ClassVar[str | None] = FluxDepExp.persistence_tag()
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, requires_soc=True, load_data=True
    )

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FluxDepCfg]:
        """Build a fresh form using only the fixed Open/Reset capture."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: FluxDepCfg, *, context: RunContext
    ) -> MistFluxDepRun:
        """Acquire mean IQ with the accepted cfg and this Run's device snapshot."""
        del req
        return RunRecord(cfg, FluxDepExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistFluxDepRun, MistFluxDepParams],
        *,
        plots: Plots,
    ) -> MistFluxDepAnalysis:
        """Publish fit from the explicit source; missing historical cfg is allowed.

        Display calibration is entered in the analysis pane, never borrowed from
        the current aliases. The core rejects an uncalibrated secondary axis.
        """
        params = req.analyze_params
        FluxDepExp().analyze(
            req.run_result,
            FluxDepAnalyzeOptions(
                flux_half=params.flux_half,
                flux_period=params.flux_period,
                ac_coeff=params.ac_coeff,
                secondary_xaxis=params.secondary_xaxis,
                auto_range=params.auto_range,
            ),
            plots=plots,
        )
        return MistFluxDepAnalysis()
