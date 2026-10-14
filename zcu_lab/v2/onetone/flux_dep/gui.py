from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
)
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
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
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import (
    alias_name_seed,
    custom,
    path_seed,
)
from zcu_tools.gui.app.measure.spec_helpers import make_pulse_readout_module_spec
from zcu_tools.gui.cfg import (
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.gui.interactive import PluginDefinition, Session

from zcu_lab.v2._support.measure.ctx_helpers import optional_parameter_float
from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.seeds import (
    flux_range,
    res_freq_range,
)
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
    experiment_tag: ClassVar[str | None] = FluxDepExp.persistence_tag()
    exp_cls = FluxDepExp
    ExpCfg_cls = FluxDepCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.INTERACTIVE, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
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
                default=alias_name_seed("flux_dev"),
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
