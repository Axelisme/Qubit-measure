"""Synthetic parameterized picker for the recipe-to-owner contract."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs, FluxPickState
from zcu_tools.experiment import ExpCfgModel, RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    ParameterWriteback,
    RunRequest,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.flux_pick_plugin import (
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickResult
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.interactive import PluginDefinition
from zcu_tools.plotting.plots import Plots

from tests.gui.app.measure._flux_adapter import SpectrumAdapter


@dataclass
class PickParams:
    tone: int = 0
    half: float = 0.5
    integer: float = -0.5


class PickCfg(ExpCfgModel):
    pass


class RecipePicker(
    BaseAdapter[PickCfg, RunRecord[PickCfg, float], FluxPickResult, PickParams]
):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False, load_data=False, analysis=AnalysisMode.INTERACTIVE
    )
    make_interactive_frontend = SpectrumAdapter.make_interactive_frontend

    @staticmethod
    def make_form(ctx: FormContext) -> Form[PickCfg]:
        del ctx
        return Form(PickCfg, {})

    def run(
        self, req: RunRequest, cfg: PickCfg, *, context: RunContext
    ) -> RunRecord[PickCfg, float]:
        return RunRecord(cfg, 1.0)

    def make_interactive_plugin(
        self,
        req: AnalyzeRequest[RunRecord[PickCfg, float], PickParams],
        *,
        plots: Plots,
    ) -> PluginDefinition[FluxPickState, FluxPickResult]:
        params = req.analyze_params
        if params.tone != 4:
            raise ValueError("Synthetic acquisition contains original tone 4 only")
        return make_flux_pick_plugin(
            FluxPickInputs(
                np.ones((5, 3), dtype=np.complex128),
                np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
                np.array([4.0, 4.5, 5.0]),
            ),
            flux_half=params.half,
            flux_int=params.integer,
            force_magnitude=True,
            plots=plots,
            result_builder=render_flux_pick,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RunRecord[PickCfg, float], FluxPickResult]
    ) -> Sequence[ParameterWriteback]:
        return [
            ParameterWriteback(
                "general.ext.recipe_half",
                "Accepted half line",
                req.analyze_result.flx_half,
            )
        ]
