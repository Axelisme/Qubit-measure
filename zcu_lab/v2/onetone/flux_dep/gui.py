from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    AnalyzeRequest,
    Interactive,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.onetone.flux_dep.core import FluxDepCfg, FluxDepExp, FluxDepResult
from zcu_lab.v2.onetone.flux_dep.form import make_form
from zcu_lab.v2.support.measure.ctx_helpers import optional_parameter_float

OneToneFluxDepRunResult: TypeAlias = RunRecord[FluxDepCfg, FluxDepResult]


def _make_interactive_plugin(
    req: AnalyzeRequest[OneToneFluxDepRunResult, FluxPickParams],
    *,
    plots: Plots,
) -> FluxPickPlugin[FluxPickResult]:
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


def _get_writeback_items(
    req: WritebackRequest[OneToneFluxDepRunResult, FluxPickResult],
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


def run(
    cfg: FluxDepCfg, *, context: RunContext
) -> RunRecord[FluxDepCfg, FluxDepResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=FluxDepExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=FluxDepExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Interactive(
        plugin=_make_interactive_plugin,
        frontend=make_flux_pick_frontend,
        options=lambda source, env: FluxPickParams(),
        writeback=_get_writeback_items,
    ),
)
