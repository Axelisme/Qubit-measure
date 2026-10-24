"""Synthetic spectra wired to the public flux-picker plugin and frontend."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs, FluxPickState
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    Experiment,
    ZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    AnalyzeRequest,
    Interactive,
    ParameterWriteback,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.flux_pick_frontend import make_flux_pick_frontend
from zcu_tools.gui.app.measure.flux_pick_plugin import (
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_tools.gui.app.measure.interactive_flux_pick import (
    FluxPickParams,
    FluxPickResult,
)
from zcu_tools.gui.app.measure.ui.interactive_frontend import (
    InteractiveFrontend,
    InteractiveFrontendEnv,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.interactive import PluginDefinition, Session
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg


@dataclass
class SpectrumResult:
    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class SpectrumSweep(ConfigBase):
    freq: SweepCfg = Field(
        default_factory=lambda: SweepCfg(start=4.0, stop=5.0, expts=30, step=1.0 / 29)
    )
    flux: SweepCfg = Field(
        default_factory=lambda: SweepCfg(start=-5.0, stop=5.0, expts=60, step=10.0 / 59)
    )


class SpectrumCfg(ConfigBase):
    sweep: SpectrumSweep = Field(default_factory=SpectrumSweep)


class SpectrumCore(Experiment[SpectrumResult, SpectrumCfg]):
    AXES_SPEC: ClassVar[AxesSpec[SpectrumResult, SpectrumCfg]] = AxesSpec(
        axes=(Axis("freqs", "freqs", "Hz", scale=1e6), Axis("values", "values", "A")),
        z=ZSpec("signals", "signals", "V"),
        result_type=SpectrumResult,
        cfg_type=SpectrumCfg,
        tag="test/spectrum",
    )


def SpectrumAdapter_make_form(ctx: FormContext) -> Form[SpectrumCfg]:
    """Return fixed typed spectra input for the analysis fixture."""
    del ctx
    return Form(SpectrumCfg, {})


def SpectrumAdapter_make_interactive_plugin(
    req: AnalyzeRequest[RunRecord[SpectrumCfg, SpectrumResult], FluxPickParams],
    *,
    plots: Plots,
    force_magnitude: bool = True,
) -> PluginDefinition[FluxPickState, FluxPickResult]:
    result = req.run_result.result
    half, integer = (None, None)
    if req.parameters is not None:
        half = req.parameters.lookup("$qubit.flux_half").value
        integer = req.parameters.lookup("$qubit.flux_int").value
        if half is not None and (not isinstance(half, (int, float))):
            raise ValueError("Fixture line must be numeric")
        if integer is not None and (not isinstance(integer, (int, float))):
            raise ValueError("Fixture line must be numeric")
    return make_flux_pick_plugin(
        FluxPickInputs(result.signals, result.values, result.freqs),
        flux_half=half,
        flux_int=integer,
        force_magnitude=force_magnitude,
        plots=plots,
        result_builder=render_flux_pick,
    )


def SpectrumAdapter_make_interactive_frontend(
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


def SpectrumAdapter_get_writeback_items(
    req: WritebackRequest[RunRecord[SpectrumCfg, SpectrumResult], FluxPickResult],
) -> Sequence[ParameterWriteback]:
    result = req.analyze_result
    return [
        ParameterWriteback("$qubit.flux_half", "First line", result.flx_half),
        ParameterWriteback("$qubit.flux_int", "Second line", result.flx_int),
        ParameterWriteback("$qubit.flux_period", "Distance", result.flx_period),
    ]


def SpectrumCore_run(
    cfg: SpectrumCfg, *, context: RunContext
) -> RunRecord[SpectrumCfg, SpectrumResult]:
    return RunRecord(cfg=cfg, result=SpectrumCore().run(cfg, context=context))


SpectrumAdapter = Adapter(
    spec=replace(SpectrumCore.AXES_SPEC, tag="test/spectrum"),
    make_form=SpectrumAdapter_make_form,
    run=SpectrumCore_run,
    guide=None,
    analysis=Interactive(
        plugin=SpectrumAdapter_make_interactive_plugin,
        frontend=SpectrumAdapter_make_interactive_frontend,
        options=lambda source, env: FluxPickParams(),
        writeback=SpectrumAdapter_get_writeback_items,
    ),
    post_analysis=None,
)


PhaseSpectrumAdapter = Adapter(
    spec=SpectrumAdapter.spec,
    make_form=SpectrumAdapter.make_form,
    run=SpectrumAdapter.run,
    guide=None,
    analysis=Interactive(
        plugin=partial(SpectrumAdapter_make_interactive_plugin, force_magnitude=False),
        frontend=SpectrumAdapter_make_interactive_frontend,
        options=lambda source, env: FluxPickParams(),
        writeback=SpectrumAdapter_get_writeback_items,
    ),
    post_analysis=SpectrumAdapter.post_analysis,
)
