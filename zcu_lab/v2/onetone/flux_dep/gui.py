from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    ParameterWriteback,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.flux_pick_frontend import make_flux_pick_frontend
from zcu_tools.gui.app.measure.flux_pick_plugin import (
    FluxPickPlugin,
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_tools.gui.app.measure.interactive_flux_pick import (
    FluxPickParams,
    FluxPickResult,
)
from zcu_tools.gui.interactive import PluginDefinition, Session

from zcu_lab.v2._support.measure.ctx_helpers import optional_parameter_float
from zcu_lab.v2.onetone.flux_dep.core import FluxDepCfg, FluxDepExp, FluxDepResult
from zcu_lab.v2.onetone.flux_dep.form import make_form

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.ui.interactive_frontend import (
        InteractiveFrontend,
        InteractiveFrontendEnv,
    )
    from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

OneToneFluxDepRunResult: TypeAlias = RunRecord[FluxDepCfg, FluxDepResult]


class OneToneFluxDepAdapter(
    BaseAdapter[FluxDepCfg, OneToneFluxDepRunResult, FluxPickResult, FluxPickParams]
):
    exp_cls = FluxDepExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FluxDepCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.INTERACTIVE, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: FluxDepCfg,
        *,
        context: RunContext,
    ) -> OneToneFluxDepRunResult:
        del req
        result = FluxDepExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    # -- interactive analysis: user picks the half/integer flux lines ----------

    def make_interactive_plugin(
        self,
        req: AnalyzeRequest[OneToneFluxDepRunResult, FluxPickParams],
        *,
        plots: Plots,
    ) -> FluxPickPlugin:
        result = req.run_result.result
        inputs = FluxPickInputs(result.signals, result.values, result.freqs * 1e-3)
        parameters = req.parameters
        flux_half = (
            None
            if parameters is None
            else optional_parameter_float(parameters, "$qubit.flux_half")
        )
        flux_int = (
            None
            if parameters is None
            else optional_parameter_float(parameters, "$qubit.flux_int")
        )
        return make_flux_pick_plugin(
            inputs,
            flux_half=flux_half,
            flux_int=flux_int,
            force_magnitude=True,
            plots=plots,
            result_builder=render_flux_pick,
        )

    def make_interactive_frontend(
        self,
        plugin: PluginDefinition[Any, Any],
        session: Session[Any],
        env: InteractiveFrontendEnv,
        request_finish: Callable[[], bool],
        request_cancel: Callable[[], bool],
        *,
        plots: Plots,
    ) -> InteractiveFrontend:
        return make_flux_pick_frontend(
            plugin, session, env, request_finish, request_cancel, plots=plots
        )

    def get_writeback_items(
        self, req: WritebackRequest[OneToneFluxDepRunResult, FluxPickResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            ParameterWriteback(
                target_name="$qubit.flux_half",
                description="Half-flux (Φ₀/2) sweet-spot device value",
                proposed_value=result.flx_half,
            ),
            ParameterWriteback(
                target_name="$qubit.flux_int",
                description="Integer-flux sweet-spot device value",
                proposed_value=result.flx_int,
            ),
            ParameterWriteback(
                target_name="$qubit.flux_period",
                description="Flux period (device units) = 2·|flx_int − flx_half|",
                proposed_value=result.flx_period,
            ),
        ]
