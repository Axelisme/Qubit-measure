from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

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

from zcu_lab.v2._support.measure.ctx_helpers import optional_parameter_float
from zcu_lab.v2.twotone.fluxdep.core import FreqFluxCfg, FreqFluxExp, FreqFluxResult

from .form import make_form

if TYPE_CHECKING:
    from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

FluxDepRunResult: TypeAlias = RunRecord[FreqFluxCfg, FreqFluxResult]


def _FluxDep_make_form(ctx: FormContext) -> Form[FreqFluxCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _FluxDep_make_interactive_plugin(
    req: AnalyzeRequest[FluxDepRunResult, FluxPickParams], *, plots: Plots
) -> FluxPickPlugin[FluxPickResult]:
    """Build a phase-aware picker from the fixed Run/Load result source.

    req.parameters=None or absent/null $qubit.flux_half/flux_int uses the
    line owner's defaults. A supplied capture resolves those paths using
    its fixed aliases; scalar, alias and source failures propagate. No live
    context or legacy md/ml is read. plots receives the pick figure on Done.
    """
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
        force_magnitude=False,
        plots=plots,
        result_builder=render_flux_pick,
    )


def _FluxDep_get_writeback_items(
    req: WritebackRequest[FluxDepRunResult, FluxPickResult],
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


def _run(
    cfg: FreqFluxCfg, *, context: RunContext
) -> RunRecord[FreqFluxCfg, FreqFluxResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=FreqFluxExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=FreqFluxExp.AXES_SPEC,
    make_form=_FluxDep_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Interactive(
        plugin=_FluxDep_make_interactive_plugin,
        frontend=make_flux_pick_frontend,
        options=lambda source, env: FluxPickParams(),
        writeback=_FluxDep_get_writeback_items,
    ),
)
