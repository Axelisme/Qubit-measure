from functools import partial
from pathlib import Path

from zcu_tools.gui.app.measure.adapter import Adapter, Interactive
from zcu_tools.gui.app.measure.flux_pick_frontend import make_flux_pick_frontend

from zcu_lab.v2._support.measure.mux_flux_adapter import (
    get_analyze_params,
    get_writeback_items,
    make_interactive_plugin,
)

from .core import TwotoneFluxGlobalExp
from .form import make_form

DEFINITION = Adapter(
    spec=TwotoneFluxGlobalExp.AXES_SPEC,
    make_form=make_form,
    run=TwotoneFluxGlobalExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Interactive(
        plugin=partial(make_interactive_plugin, force_magnitude=False),
        frontend=make_flux_pick_frontend,
        options=get_analyze_params,
        writeback=get_writeback_items,
    ),
)
