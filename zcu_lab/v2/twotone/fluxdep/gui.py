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

from zcu_lab.v2.support.measure.ctx_helpers import flux_pick_initial_points
from zcu_lab.v2.support.measure.writeback_helpers import fluxdep_writeback_items
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

    Initial lines use the scanned device's new flux entry and other
    contributors' Run snapshot values. Missing capture/cfg or calibration
    leaves the line owner's defaults. Invalid capture or contributors fail.
    No live context is read. plots receives the pick figure on Done.
    """
    result = req.run_result.result
    inputs = FluxPickInputs(result.signals, result.values, result.freqs * 1e-3)
    cfg, metadata = req.run_result.cfg, req.run_result.metadata
    points = (
        None
        if cfg is None
        else flux_pick_initial_points(
            req.parameters,
            cfg.flux_dev,
            None if metadata is None else metadata.device_snapshot,
        )
    )
    return make_flux_pick_plugin(
        inputs,
        flux_half=None if points is None else points.half,
        flux_int=None if points is None else points.integer,
        force_magnitude=False,
        plots=plots,
        result_builder=render_flux_pick,
    )


def _FluxDep_get_writeback_items(
    req: WritebackRequest[FluxDepRunResult, FluxPickResult],
) -> Sequence[WritebackItem]:
    """Propose reference calibration and a raw-fit record from this Run only."""
    cfg = req.run_result.cfg
    if cfg is None:
        return []
    return fluxdep_writeback_items(req, device_name=cfg.flux_dev)


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
