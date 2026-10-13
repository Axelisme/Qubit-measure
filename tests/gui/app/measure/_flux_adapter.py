"""Synthetic spectra wired to the public flux-picker plugin and frontend."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs, FluxPickState
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    ExpCfgModel,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    ParameterWriteback,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.flux_pick_frontend import make_flux_pick_frontend
from zcu_tools.gui.app.measure.flux_pick_plugin import (
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
from zcu_tools.gui.app.measure.seeds import alias_name_seed
from zcu_tools.gui.app.measure.ui.interactive_frontend import (
    InteractiveFrontend,
    InteractiveFrontendEnv,
)
from zcu_tools.gui.interactive import PluginDefinition, Session
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg


@dataclass
class SpectrumResult:
    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class SpectrumSweep(ExpCfgModel):
    freq: SweepCfg = Field(
        default_factory=lambda: SweepCfg(start=4.0, stop=5.0, expts=30, step=1.0 / 29)
    )
    flux: SweepCfg = Field(
        default_factory=lambda: SweepCfg(start=-5.0, stop=5.0, expts=60, step=10.0 / 59)
    )


class SpectrumCfg(ExpCfgModel):
    sweep: SpectrumSweep = Field(default_factory=SpectrumSweep)


class SpectrumCore(PersistableExperiment[SpectrumResult, SpectrumCfg]):
    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "freqs", "Hz", scale=1e6), Axis("values", "values", "A")),
        z=ZSpec("signals", "signals", "V"),
        result_type=SpectrumResult,
        cfg_type=SpectrumCfg,
        tag="test/spectrum",
    )


class SpectrumAdapter(
    BaseAdapter[
        SpectrumCfg,
        RunRecord[SpectrumCfg, SpectrumResult],
        FluxPickResult,
        FluxPickParams,
    ]
):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.INTERACTIVE, requires_soc=False, load_data=True
    )
    force_magnitude: ClassVar[bool] = True
    exp_cls = SpectrumCore
    ExpCfg_cls = SpectrumCfg
    experiment_tag: ClassVar[str | None] = "test/spectrum"

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(role_factories={})
            .device(
                "flux_dev", label="Flux Device", default=alias_name_seed("flux_dev")
            )
            .build()
        )

    def make_interactive_plugin(
        self,
        req: AnalyzeRequest[RunRecord[SpectrumCfg, SpectrumResult], FluxPickParams],
        *,
        plots: Plots,
    ) -> PluginDefinition[FluxPickState, FluxPickResult]:
        result = req.run_result.result
        half, integer = None, None
        if req.parameters is not None:
            half = req.parameters.lookup("$qubit.flux_half").value
            integer = req.parameters.lookup("$qubit.flux_int").value
            if half is not None and not isinstance(half, (int, float)):
                raise ValueError("Fixture line must be numeric")
            if integer is not None and not isinstance(integer, (int, float)):
                raise ValueError("Fixture line must be numeric")
        return make_flux_pick_plugin(
            FluxPickInputs(result.signals, result.values, result.freqs),
            flux_half=half,
            flux_int=integer,
            force_magnitude=self.force_magnitude,
            plots=plots,
            result_builder=render_flux_pick,
        )

    def make_interactive_frontend(
        self,
        plugin: PluginDefinition[FluxPickState, FluxPickResult],
        session: Session[FluxPickState],
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
        self,
        req: WritebackRequest[RunRecord[SpectrumCfg, SpectrumResult], FluxPickResult],
    ) -> Sequence[ParameterWriteback]:
        result = req.analyze_result
        return [
            ParameterWriteback("$qubit.flux_half", "First line", result.flx_half),
            ParameterWriteback("$qubit.flux_int", "Second line", result.flx_int),
            ParameterWriteback("$qubit.flux_period", "Distance", result.flx_period),
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return "spectrum"


class PhaseSpectrumAdapter(SpectrumAdapter):
    force_magnitude: ClassVar[bool] = False
