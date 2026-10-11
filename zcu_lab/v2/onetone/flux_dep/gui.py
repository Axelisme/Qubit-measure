from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
    FluxPickState,
    fold_initial_lines,
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
from zcu_tools.gui.cfg import (
    EvalValue,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.gui.interactive import PluginDefinition, Session

from zcu_lab.v2._support.measure.ctx_helpers import optional_parameter_float
from zcu_lab.v2._support.measure.flux_pick_frontend import make_flux_pick_frontend
from zcu_lab.v2._support.measure.flux_pick_plugin import (
    FluxPickPlugin,
    render_flux_pick,
)
from zcu_lab.v2._support.measure.interactive_flux_pick import (
    FluxPickParams,
    FluxPickResult,
)
from zcu_lab.v2._support.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_lab.v2._support.measure.seeds import (
    custom,
    flux_range,
    literal,
    path_seed,
    res_freq_range,
)
from zcu_lab.v2._support.measure.spec_helpers import make_pulse_readout_module_spec
from zcu_lab.v2.onetone.flux_dep.core import FluxDepCfg, FluxDepExp, FluxDepResult

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.ui.interactive_frontend import (
        InteractiveFrontend,
        InteractiveFrontendEnv,
    )
    from zcu_tools.plotting.plots import Plots

OneToneFluxDepRunResult: TypeAlias = RunRecord[FluxDepCfg, FluxDepResult]


def _blank_readout(ctx: SessionEnv) -> ReferenceValue:
    """Build fresh inline readout defaults from the tab's fixed roles."""
    value = make_default_value(make_pulse_readout_module_spec().allowed[0])
    value.with_field("pulse_cfg.freq", 0.0)
    value.with_field(
        "pulse_cfg.ch", path_seed("$resonator.wiring.ch", fallback=0).resolve(ctx)
    )
    value.with_field("pulse_cfg.nqz", 2)
    value.with_field("pulse_cfg.gain", 0.05)
    value.with_field("pulse_cfg.waveform.length", 1.0)
    value.with_field("ro_cfg.ro_freq", 0.0)
    value.with_field(
        "ro_cfg.ro_ch", path_seed("$resonator.wiring.ro_ch", fallback=0).resolve(ctx)
    )
    value.with_field(
        "ro_cfg.ro_length",
        path_seed(
            "$resonator.ext.default_probe_length",
            fallback=0.9,
            expr="$resonator.ext.default_probe_length - 0.1",
        ).resolve(ctx),
    )
    value.with_field(
        "ro_cfg.trig_offset",
        path_seed(
            "$resonator.ext.time_of_fly",
            fallback=0.55,
            expr="$resonator.ext.time_of_fly + 0.05",
        ).resolve(ctx),
    )
    return ReferenceValue(make_custom_reference_key("Pulse Readout"), value)


class OneToneFluxDepAdapter(
    BaseAdapter[FluxDepCfg, OneToneFluxDepRunResult, FluxPickResult, FluxPickParams]
):
    exp_cls = FluxDepExp
    ExpCfg_cls = FluxDepCfg
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

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder()
            .field(
                "modules.readout",
                spec=make_pulse_readout_module_spec()
                .lock_literal("pulse_cfg.freq", 0.0)
                .lock_literal("ro_cfg.ro_freq", 0.0),
                default=custom(_blank_readout, description="inline flux readout"),
            )
            .device(
                "flux_dev",
                label="Flux Device",
                default=literal(EvalValue("$flux_dev")),
            )
            .choice(
                "flux_unit",
                label="Flux unit",
                choices=("A", "V", "native"),
                default="A",
                tooltip="Working unit for the selected flux driver; no conversion.",
            )
            .relax_delay(1.0)
            .sweep(
                "flux",
                label="Flux device value",
                default=flux_range(expts=101),
                decimals=6,
            )
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=res_freq_range(expts=101, span_factor=1.0),
            )
            .reps(1000)
            .rounds(1)
            .build()
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
        half, integer = fold_initial_lines(inputs.dev_values, flux_half, flux_int)
        # One-tone resonator spectra have uninformative phase.
        return FluxPickPlugin(
            inputs,
            FluxPickState(
                flux_half=half, flux_int=integer, conjugate=False, magnitude_only=True
            ),
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
