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
    ModuleInit,
)
from zcu_tools.gui.app.measure.seeds import (
    alias_name_seed,
)
from zcu_tools.gui.interactive import PluginDefinition, Session

from zcu_lab.v2._support.measure.ctx_helpers import optional_parameter_float
from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.seeds import (
    flux_range,
    qub_freq_range,
)
from zcu_lab.v2.twotone.fluxdep.core import FreqFluxCfg, FreqFluxExp, FreqFluxResult

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.ui.interactive_frontend import (
        InteractiveFrontend,
        InteractiveFrontendEnv,
    )
    from zcu_tools.plotting.plots import Plots

FluxDepRunResult: TypeAlias = RunRecord[FreqFluxCfg, FreqFluxResult]


class FluxDepAdapter(
    BaseAdapter[FreqFluxCfg, FluxDepRunResult, FluxPickResult, FluxPickParams]
):
    experiment_tag: ClassVar[str | None] = FreqFluxExp.persistence_tag()
    exp_cls = FreqFluxExp
    ExpCfg_cls = FreqFluxCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.INTERACTIVE, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> FluxDepRunResult:
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = FreqFluxExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .reset(optional=True)
            .pulse(
                "qub_pulse",
                role_id="qub_probe",
                init=ModuleInit.INLINE,
                locked={"freq": 0.0},
            )
            .readout()
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
                default=qub_freq_range(expts=1001, span_factor=1.0),
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

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
