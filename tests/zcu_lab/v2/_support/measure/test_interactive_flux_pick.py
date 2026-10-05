"""Measure flux plugin contract through both production adapters."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from zcu_lab.v2._support.measure.flux_pick_plugin import (
    FluxPickPlugin,
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_lab.v2._support.measure.interactive_flux_pick import (
    FluxPickParams,
    FluxPickResult,
)
from zcu_lab.v2.onetone.flux_dep.core import FluxDepResult
from zcu_lab.v2.onetone.flux_dep.gui import OneToneFluxDepAdapter
from zcu_lab.v2.twotone.fluxdep.core import FreqFluxResult
from zcu_lab.v2.twotone.fluxdep.gui import FluxDepAdapter as TwoToneFluxDepAdapter


def _request(md: MetaDict | None = None) -> AnalyzeRequest[Any, FluxPickParams]:
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
    plots = Plots(NonPresentingHost())
    plugin = make_flux_pick_plugin(
        _request(), force_magnitude=True, plots=plots, result_builder=render_flux_pick
    )
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
    with pytest.raises(InvalidInputError, match="role"):
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
    assert tuple(plots) == ("pick",)
    assert isinstance(plots["pick"], Figure)
    plots.finish()
    plots.release()


def test_equal_seed_cannot_finish_until_a_valid_line_is_committed() -> None:
    md = MetaDict()
    md.flx_half = md.flx_int = 0.0
    plots = Plots(NonPresentingHost())
    plugin = make_flux_pick_plugin(
        _request(md), force_magnitude=True, plots=plots, result_builder=render_flux_pick
    )
    session = plugin.open(ManualOwnerScheduler())
    with pytest.raises(FailedPreconditionError, match="separat"):
        plugin.finish(session)
    assert session.snapshot().flux_half == session.snapshot().flux_int
    plugin.execute_command(session, "move_line", {"role": "half", "position": 1.0})
    assert plugin.finish(session).flx_period > 0.0
    assert tuple(plots) == ("pick",)
    plots.finish()
    plots.release()


@pytest.mark.parametrize(
    ("adapter_type", "magnitude_only"),
    [(OneToneFluxDepAdapter, True), (TwoToneFluxDepAdapter, False)],
)
def test_both_adapters_seed_projection_and_publish_named_result_figure(
    adapter_type, magnitude_only: bool
):
    md = MetaDict()
    md.flx_half = 0.0
    md.flx_int = 2.0
    plots = Plots(NonPresentingHost())
    request = _request(md)
    bare = request.run_result
    result_type = (
        FluxDepResult if adapter_type is OneToneFluxDepAdapter else FreqFluxResult
    )
    request = replace(
        request,
        run_result=RunRecord(
            cfg=None,
            result=result_type(
                bare.values,
                bare.freqs,
                np.asarray(bare.signals, dtype=np.complex128),
            ),
        ),
    )
    plugin = adapter_type().make_interactive_plugin(request, plots=plots)
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
    result = plugin.finish(session)
    assert tuple(plots) == ("pick",)
    assert isinstance(plots["pick"], Figure)
    assert result.to_summary_dict() == {
        "flx_half": state.flux_half,
        "flx_int": state.flux_int,
        "flx_period": 2 * abs(state.flux_int - state.flux_half),
    }
    plots.finish()
    plots.release()
