"""Synthetic parameterized picker for the recipe-to-owner contract."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs, FluxPickState
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AxesSpec, RunRecord, ZSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    AnalyzeRequest,
    Interactive,
    ParameterWriteback,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.flux_pick_plugin import (
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickResult
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.interactive import PluginDefinition
from zcu_tools.plotting.plots import Plots

from tests.gui.app.measure._flux_adapter import (
    SpectrumAdapter_make_interactive_frontend,
)


@dataclass
class PickParams:
    tone: int = 0
    half: float = 0.5
    integer: float = -0.5


class PickCfg(ConfigBase):
    pass


@dataclass
class PickResult:
    value: float = 1.0


def RecipePicker_make_form(ctx: FormContext) -> Form[PickCfg]:
    del ctx
    return Form(PickCfg, {})


def RecipePicker_run(
    cfg: PickCfg, *, context: RunContext
) -> RunRecord[PickCfg, PickResult]:
    return RunRecord(cfg=cfg, result=PickResult())


def RecipePicker_make_interactive_plugin(
    req: AnalyzeRequest[RunRecord[PickCfg, PickResult], PickParams], *, plots: Plots
) -> PluginDefinition[FluxPickState, FluxPickResult]:
    params = req.analyze_params
    if params.tone != 4:
        raise ValueError("Synthetic acquisition contains original tone 4 only")
    return make_flux_pick_plugin(
        FluxPickInputs(
            np.ones((5, 3), dtype=np.complex128),
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
            np.array([4.0, 4.5, 5.0]),
        ),
        flux_half=params.half,
        flux_int=params.integer,
        force_magnitude=True,
        plots=plots,
        result_builder=render_flux_pick,
    )


def RecipePicker_get_writeback_items(
    req: WritebackRequest[RunRecord[PickCfg, PickResult], FluxPickResult],
) -> Sequence[ParameterWriteback]:
    return [
        ParameterWriteback(
            "general.ext.recipe_half", "Accepted half line", req.analyze_result.flx_half
        )
    ]


RecipePicker = Adapter(
    spec=AxesSpec(
        axes=(),
        z=ZSpec("value", "data", ""),
        result_type=PickResult,
        cfg_type=PickCfg,
        tag="test/RecipePicker",
    ),
    make_form=RecipePicker_make_form,
    run=RecipePicker_run,
    guide=None,
    analysis=Interactive(
        plugin=RecipePicker_make_interactive_plugin,
        frontend=SpectrumAdapter_make_interactive_frontend,
        options=lambda source, env: PickParams(),
        writeback=RecipePicker_get_writeback_items,
    ),
    post_analysis=None,
)
