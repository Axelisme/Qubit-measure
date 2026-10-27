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
from zcu_lab.v2.support.measure.ctx_helpers import flux_pick_initial_points
from zcu_lab.v2.support.measure.writeback_helpers import fluxdep_writeback_items

OneToneFluxDepRunResult: TypeAlias = RunRecord[FluxDepCfg, FluxDepResult]


def _make_interactive_plugin(
    req: AnalyzeRequest[OneToneFluxDepRunResult, FluxPickParams],
    *,
    plots: Plots,
) -> FluxPickPlugin[FluxPickResult]:
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
        force_magnitude=True,
        plots=plots,
        result_builder=render_flux_pick,
    )


def _get_writeback_items(
    req: WritebackRequest[OneToneFluxDepRunResult, FluxPickResult],
) -> Sequence[WritebackItem]:
    """Propose reference calibration and a raw-fit record from this Run only."""
    cfg = req.run_result.cfg
    if cfg is None:
        return []
    return fluxdep_writeback_items(req, device_name=cfg.flux_dev)


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
