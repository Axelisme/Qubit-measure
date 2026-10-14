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
    experiment_tag: ClassVar[str | None] = FluxDepExp.persistence_tag()
    exp_cls = FluxDepExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FluxDepCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

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
            "Opening a tab or Reset evaluates Form starting expressions once "
            "against captured parameters into direct input. Range.around seeds "
            "101 frequency points at $resonator.freq ± $resonator.kappa, in "
            "MHz. The inline readout reads $resonator.freq, "
            "$resonator.wiring.ch and $resonator.wiring.ro_ch for its tones "
            "and channels. $resonator.ext.default_probe_length - 0.1 us seeds "
            "the ADC window; $resonator.ext.time_of_fly + 0.05 us seeds trigger "
            "timing. Range.between seeds 101 flux points from "
            "1.1 * $qubit.flux_int - 0.1 * $qubit.flux_half to "
            "1.1 * $qubit.flux_half - 0.1 * $qubit.flux_int, in the selected "
            "device unit. Missing or invalid required sources leave the "
            "affected fields empty and Invalid, without a numeric sweep "
            "fallback. For first bring-up without calibration, enter safe, "
            "user-approved frequency and flux bounds. The picker separately "
            "reads captured $qubit.flux_half/$qubit.flux_int; absent/null leaves "
            "use its default line positions. Choose sweet spots from the "
            "measured map."
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
            "(= 2·|qubit.flux_int − qubit.flux_half|) through the captured "
            "role binding. "
            "The picking is user/agent judgement from "
            "the measured map, not simulator truth; apply writeback only after "
            "reviewing the figure."
        ),
        recommended=(
            "Run after lookback and an initial onetone/freq, before any "
            "twotone flux mapping, to find $qubit.flux_int / period from the "
            "resonator map. Interactive flux-line pick (no fit). Review "
            "'flux_dev': opening a tab or Reset resolves $flux_dev once from "
            "the alias table to a direct connected-device name, with no "
            "fallback. A missing/unbound alias leaves the required field empty "
            "and Invalid. Confirm the device and 'flux_unit', which starts as "
            "'A' and does not convert values. The calibrated sweep uses 101 "
            "flux points across the padded interval between $qubit.flux_int "
            "and $qubit.flux_half, and 101 frequency points at $resonator.freq ± "
            "$resonator.kappa MHz. Start at a low "
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
            "After accepting $qubit.flux_int, move the flux device there and "
            "re-run "
            "onetone/freq at that flux before twotone/freq."
        ),
    )

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
