"""Measure flux plugin contract through both production adapters."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.v2_gui.adapters._support import FluxPickParams, FluxPickResult
from zcu_tools.experiment.v2_gui.adapters._support.flux_pick_plugin import (
    FluxPickPlugin,
    make_flux_pick_plugin,
)
from zcu_tools.experiment.v2_gui.adapters.onetone.flux_dep import (
    OneToneFluxDepAdapter,
)
from zcu_tools.experiment.v2_gui.adapters.twotone.flux_dep import (
    FluxDepAdapter as TwoToneFluxDepAdapter,
)
from zcu_tools.gui.app.main.adapter import AnalyzeRequest
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _request(md: MetaDict | None = None):
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.exp(-(devs[:, None] ** 2)) * np.ones((1, 30))
    return AnalyzeRequest(
        run_result=SimpleNamespace(signals=signals, values=devs, freqs=freqs),
        analyze_params=FluxPickParams(),
        md=md if md is not None else MetaDict(),
        ml=ModuleLibrary(),
        predictor=None,
    )


def test_plugin_typed_actions_and_commands_share_committed_state():
    plugin = make_flux_pick_plugin(_request(), force_magnitude=True)
    session = plugin.open(ManualOwnerScheduler())
    start = session.snapshot()
    assert start.magnitude_only is True
    assert start.conjugate is False

    plugin.actions.move.execute(session, ("half", start.flux_half + 0.6))
    after_gui = session.snapshot()
    assert after_gui.flux_half == pytest.approx(start.flux_half + 0.6)
    assert after_gui.flux_int == start.flux_int

    plugin.execute_command(session, "set_conjugate", {"enabled": True})
    plugin.execute_command(
        session, "move_line", {"role": "half", "position": after_gui.flux_half + 0.4}
    )
    after_remote = session.snapshot()
    assert after_remote.flux_half - after_gui.flux_half == pytest.approx(0.4)
    assert after_remote.flux_int - after_gui.flux_int == pytest.approx(0.4)
    with pytest.raises(ValueError, match="role"):
        plugin.execute_command(session, "move_line", {"role": "bad", "position": 1.0})
    assert session.snapshot() == after_remote
    plugin.execute_command(session, "swap_lines", {})
    swapped = session.snapshot()
    assert (swapped.flux_half, swapped.flux_int) == (
        after_remote.flux_int,
        after_remote.flux_half,
    )
    result = plugin.finish(session)
    assert isinstance(result, FluxPickResult)
    assert result.flx_half == swapped.flux_half
    assert result.flx_period == 2 * abs(swapped.flux_int - swapped.flux_half)
    assert result.figure is None


@pytest.mark.parametrize(
    ("adapter_type", "magnitude_only"),
    [(OneToneFluxDepAdapter, True), (TwoToneFluxDepAdapter, False)],
)
def test_both_adapters_seed_projection_and_attach_frontend_figure(
    adapter_type, magnitude_only: bool
):
    md = MetaDict()
    md.flx_half = 0.0
    md.flx_int = 2.0
    plugin = adapter_type().make_interactive_plugin(_request(md))
    assert isinstance(plugin, FluxPickPlugin)
    session = plugin.open(ManualOwnerScheduler())
    state = session.snapshot()
    assert state.magnitude_only is magnitude_only
    assert state.flux_half == pytest.approx(0.0)
    assert state.flux_int == pytest.approx(2.0)
    assert [command.name for command in plugin.commands] == [
        "move_line",
        "set_conjugate",
        "swap_lines",
        "auto_align",
    ]
    figure = Figure()
    result = plugin.finish(session, figure)
    assert result.figure is figure
    assert result.to_summary_dict() == {
        "flx_half": state.flux_half,
        "flx_int": state.flux_int,
        "flx_period": 2 * abs(state.flux_int - state.flux_half),
    }
