from dataclasses import replace
from functools import partial
from pathlib import Path

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Interactive
from zcu_tools.gui.app.measure.flux_pick_frontend import make_flux_pick_frontend

from zcu_lab.v2.support.measure.mux_flux_adapter import (
    get_analyze_params,
    get_writeback_items,
    make_interactive_plugin,
)
from zcu_lab.v2.support.measure.mux_save_names import mux_save_stem
from zcu_lab.v2.support.mux_flux import MuxFluxResult

from .core import OnetoneFluxGlobalCfg, OnetoneFluxGlobalExp
from .form import make_form


def run(
    cfg: OnetoneFluxGlobalCfg, *, context: RunContext
) -> RunRecord[OnetoneFluxGlobalCfg, MuxFluxResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=OnetoneFluxGlobalExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=replace(OnetoneFluxGlobalExp.AXES_SPEC, save_stem=mux_save_stem),
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Interactive(
        plugin=partial(make_interactive_plugin, force_magnitude=True),
        frontend=make_flux_pick_frontend,
        options=get_analyze_params,
        writeback=get_writeback_items,
    ),
)
