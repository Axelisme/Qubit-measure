from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
)
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
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
from zcu_lab.v2.onetone.flux_dep.core import FluxDepCfg, FluxDepExp, FluxDepResult
from zcu_lab.v2.onetone.flux_dep.form import FORM

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.ui.interactive_frontend import (
        InteractiveFrontend,
        InteractiveFrontendEnv,
    )
    from zcu_tools.plotting.plots import Plots

OneToneFluxDepRunResult: TypeAlias = RunRecord[FluxDepCfg, FluxDepResult]


class OneToneFluxDepAdapter(
    BaseAdapter[FluxDepCfg, OneToneFluxDepRunResult, FluxPickResult, FluxPickParams]
):
    experiment_tag: ClassVar[str | None] = FluxDepExp.persistence_tag()
    exp_cls = FluxDepExp
    ExpCfg_cls = FluxDepCfg
    form = FORM
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.INTERACTIVE, load_data=True
    )

    guide_text: ClassVar[AdapterGuide] = AdapterGuide(
        behavior=(
            "One-tone resonator flux dependence: a 2D sweep of an external "
            "flux-bias device value versus readout frequency, tracing how the "
            "resonator frequency moves with flux. The basis for locating a "
            "Fluxonium's flux sweet spots (half-flux and integer-flux points). "
            "Runs on real hardware; requires a SoC connection and a configured "
            "flux-bias device."
        ),
        expects_md=(
            "Reads fixed tab parameters: resonator.freq and resonator.kappa "
            "seed the frequency sweep; resonator.ext.default_probe_length "
            "minus 0.1 us seeds the ADC window; resonator.wiring.ch/ro_ch "
            "seed channels; resonator.ext.time_of_fly plus 0.05 us seeds "
            "trigger timing. Captured qubit.flux_half/flux_int seed the flux "
            "sweep and picker when calibrated. First bring-up must not assume "
            "these are known: absent/null flux values use a [-4e-3, 4e-3] "
            "device sweep that the operator must adjust to a safe, "
            "user-approved range."
        ),
        expects_ml=(
            "Starts with a fresh inline pulse readout and const waveform. "
            "It does not adopt a library module or named waveform."
        ),
        typical_writeback=(
            "Interactive analysis (not a fit): after the run, the user drags "
            "two lines on the 2D map to mark the half-flux and integer-flux "
            "sweet spots, then clicks Done. The preview proposes "
            "qubit.flux_half, qubit.flux_int, and qubit.flux_period "
            "(= 2·|flx_int − flx_half|) through the captured role binding. "
            "The picking is user/agent judgement from "
            "the measured map, not simulator truth; apply writeback only after "
            "reviewing the figure."
        ),
        recommended=(
            "Run after lookback and an initial onetone/freq, before any "
            "twotone flux mapping, to find 'flx_int' / period from the "
            "resonator map. Interactive flux-line pick (no fit). Also set the "
            "'flux_dev' field — the flux-bias device name ($flux_dev resolves "
            "from the alias table with no fallback) — "
            "and confirm it points at a connected device. Typical sweep: "
            "flux ~101 points "
            "across one period (driven by flx_half/flx_int when calibrated), "
            "frequency r_f ± one linewidth over ~101 points. Start at a low "
            "readout gain (~0.005) to stay below punch-out so the dip tracks "
            "cleanly; if the dip is hard to see (poor SNR), raise the gain "
            "toward ~0.05 while keeping the frequency window tight. Survey "
            "wide, then narrow around a sweet spot — and if the resonator's "
            "flux dispersion (dispersive shift) looks small, narrow the "
            "frequency window early instead of keeping a broad default span. "
            "For consecutive flux sweeps, reverse the sweep direction (swap "
            "start/stop) on the next run so the current source need not ramp "
            "all the way back across the full range first. Inspect the 2D map "
            "with gui_tab_get_figure (subtab_id='run') to judge window / SNR / shift. "
            "After accepting 'flx_int', move the flux device there and re-run "
            "onetone/freq at that flux before twotone/freq."
        ),
    )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> OneToneFluxDepRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
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

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.res_name}_flux"
