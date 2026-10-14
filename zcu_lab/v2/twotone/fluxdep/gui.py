from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    ParameterWriteback,
    RunRequest,
    SessionEnv,
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
from zcu_lab.v2.twotone.fluxdep.core import FreqFluxCfg, FreqFluxExp, FreqFluxResult

from .form import make_form

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.ui.interactive_frontend import (
        InteractiveFrontend,
        InteractiveFrontendEnv,
    )
    from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

FluxDepRunResult: TypeAlias = RunRecord[FreqFluxCfg, FreqFluxResult]


class FluxDepAdapter(
    BaseAdapter[FreqFluxCfg, FluxDepRunResult, FluxPickResult, FluxPickParams]
):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqFluxCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    experiment_tag: ClassVar[str | None] = FreqFluxExp.persistence_tag()
    exp_cls = FreqFluxExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.INTERACTIVE, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "Two-tone qubit flux dependence: a 2D scan stepping an external flux device "
            "(outer) against the qubit-drive frequency (inner), reading out the "
            "resonator at each point, to map the qubit spectrum versus flux (the "
            "Fluxonium flux-dispersion arcs). Runs on real hardware and needs a "
            "connected flux device. No automated fit in the GUI — points are extracted "
            "interactively. This is not the early flux-finder; use it only after "
            "readout and qubit spectroscopy are already credible."
        ),
        expects_md=(
            "Inline templates read qubit.freq, qubit.wiring.L01.ch, resonator.freq, "
            "resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us from the "
            "fixed capture. Missing required calibration remains Invalid; no numeric "
            "fallback. Frequency sweep is qubit.freq ± qubit.kappa over 1001 points. "
            "Flux spans 10% beyond qubit.flux_int and qubit.flux_half; missing "
            "endpoints remain Invalid."
        ),
        expects_ml=(
            "Uses captured qubit.module.readout with a fresh readout template when "
            "unset or dangling. Optional qubit.module.reset stays disabled without a "
            "source. Library references remain selectable; sweeps overwrite their "
            "controlled fields only at Run. flux_dev resolves the captured flux_dev "
            "alias to a direct device name on Open/Reset; missing or unbound alias "
            "remains Invalid. flux_unit stays explicit and is never converted."
        ),
        typical_writeback=(
            "Interactive analysis (not a fit): after the run, the user drags two lines "
            "on the 2D map to mark the half-flux and integer-flux sweet spots, then "
            "clicks Done. The result writes back 'qubit.flux_half', 'qubit.flux_int' "
            "and 'qubit.flux_period' (= 2·|flx_int − flx_half|) to the parameter "
            "context as a preview. Picking is user/agent judgement from the measured "
            "map; do not use this preview to replace the earlier onetone/flux_dep flux "
            "calibration unless the map is clearly trustworthy."
        ),
        recommended=(
            "No fit options. Use after onetone/flux_dep has established the flux period "
            "and twotone/freq has found 'qubit.freq'. Do not run this early to find "
            "flux: with uncalibrated readout or an unknown qubit-drive window the arc "
            "is usually invisible. Default sweep: flux ~101 points across roughly one "
            "period (bracketed by 'qubit.flux_int'/'qubit.flux_half' when calibrated), "
            "frequency ~1001 points spanning ±qubit.kappa around 'qubit.freq'. For a "
            "later model survey use a wide flux range and a broad frequency window at "
            "moderate drive gain to capture the full arc, then narrow both. For "
            "consecutive flux sweeps, reverse the sweep direction (swap start/stop) on "
            "the next run so the flux source need not ramp back across the full range "
            "first. Inspect the 2D map with gui_tab_get_figure (subtab_id='run'). "
            "Confirm 'flux_dev' points at a connected device before running."
        ),
    )

    def run(
        self,
        req: RunRequest,
        cfg: FreqFluxCfg,
        *,
        context: RunContext,
    ) -> FluxDepRunResult:
        del req
        result = FreqFluxExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def make_interactive_plugin(
        self, req: AnalyzeRequest[FluxDepRunResult, FluxPickParams], *, plots: Plots
    ) -> FluxPickPlugin:
        """Build a phase-aware picker from the fixed Run/Load result source.

        req.parameters=None or absent/null $qubit.flux_half/flux_int uses the
        line owner's defaults. A supplied capture resolves those paths using
        its fixed aliases; scalar, alias and source failures propagate. No live
        context or legacy md/ml is read. plots receives the pick figure on Done.
        """
        result = req.run_result.result
        inputs = FluxPickInputs(result.signals, result.values, result.freqs)
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
            force_magnitude=False,
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
        self, req: WritebackRequest[FluxDepRunResult, FluxPickResult]
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

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_qubit_flux"
