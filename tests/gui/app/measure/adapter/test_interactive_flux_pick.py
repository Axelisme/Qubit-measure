"""Measure flux plugin contracts using caller-owned numeric spectra and seeds."""

from __future__ import annotations

from collections.abc import Generator
from typing import Literal

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs, fold_initial_lines
from zcu_tools.gui.app.measure.flux_pick_plugin import (
    FluxPickPlugin,
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickResult
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.plotting.plots import NonPresentingHost, Plots

FluxEntrypoint = Literal["factory", "twotone"]


@pytest.fixture
def flux_plots() -> Generator[Plots]:
    plots = Plots(NonPresentingHost())
    try:
        yield plots
    finally:
        plots.finish()
        plots.release()


def _inputs() -> FluxPickInputs:
    values = np.linspace(-5.0, 5.0, 60, dtype=np.float64)
    freqs = np.linspace(4.0, 5.0, 30, dtype=np.float64)
    signals = np.asarray(
        np.exp(-(values[:, None] ** 2)) * np.ones((1, 30)), dtype=np.complex128
    )
    return FluxPickInputs(signals, values, freqs)


def _current_plugin(
    entrypoint: FluxEntrypoint, half: float | None, integer: float | None, plots: Plots
) -> FluxPickPlugin:
    # The two retained projection modes are synthetic callers, not lab adapters.
    return make_flux_pick_plugin(
        _inputs(),
        flux_half=half,
        flux_int=integer,
        force_magnitude=entrypoint == "factory",
        plots=plots,
        result_builder=render_flux_pick,
    )


@pytest.mark.parametrize("entrypoint", ["factory", "twotone"])
def test_current_request_without_parameters_builds_and_finishes_picker(
    entrypoint: FluxEntrypoint, flux_plots: Plots
) -> None:
    plugin = _current_plugin(entrypoint, None, None, flux_plots)
    session = plugin.open(ManualOwnerScheduler())
    try:
        state = session.snapshot()
        assert state.flux_half == pytest.approx(0.0)
        assert state.flux_int > state.flux_half
        assert state.magnitude_only is (entrypoint == "factory")
        plugin.execute_command(session, "move_line", {"role": "half", "position": 0.5})
        result = plugin.finish(session)
        assert result.flx_half == pytest.approx(0.5)
        assert result.flx_period > 0.0
        assert tuple(flux_plots) == ("pick",)
    finally:
        session.dispose()


@pytest.mark.parametrize("entrypoint", ["factory", "twotone"])
@pytest.mark.parametrize("half", [None, 0.5])
def test_current_request_uses_fixed_parameter_calibration(
    entrypoint: FluxEntrypoint, half: float | None, flux_plots: Plots
) -> None:
    expected_half, expected_int = fold_initial_lines(_inputs().dev_values, half, 2.0)
    plugin = _current_plugin(entrypoint, half, 2.0, flux_plots)
    session = plugin.open(ManualOwnerScheduler())
    try:
        state = session.snapshot()
        assert state.flux_half == pytest.approx(expected_half)
        assert state.flux_int == pytest.approx(expected_int)
        assert plugin.finish(session).flx_period == pytest.approx(
            2 * abs(expected_int - expected_half)
        )
    finally:
        session.dispose()


@pytest.mark.parametrize("entrypoint", ["factory", "twotone"])
@pytest.mark.parametrize(
    ("half", "error", "message"), [(float("nan"), ValueError, "NaN")]
)
def test_current_request_propagates_invalid_parameter_calibration(
    entrypoint: FluxEntrypoint,
    half: float,
    error: type[Exception],
    message: str,
    flux_plots: Plots,
) -> None:
    with pytest.raises(error, match=message):
        _current_plugin(entrypoint, half, 2.0, flux_plots)


def test_plugin_typed_actions_and_commands_share_committed_state():
    plots = Plots(NonPresentingHost())
    plugin = make_flux_pick_plugin(
        _inputs(), force_magnitude=True, plots=plots, result_builder=render_flux_pick
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
    plots = Plots(NonPresentingHost())
    plugin = make_flux_pick_plugin(
        _inputs(),
        flux_half=0.0,
        flux_int=0.0,
        force_magnitude=True,
        plots=plots,
        result_builder=render_flux_pick,
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


@pytest.mark.parametrize("magnitude_only", [True, False])
def test_both_projections_seed_and_publish_named_result_figure(
    magnitude_only: bool,
) -> None:
    plots = Plots(NonPresentingHost())
    plugin = make_flux_pick_plugin(
        _inputs(),
        flux_half=0.0,
        flux_int=2.0,
        force_magnitude=magnitude_only,
        plots=plots,
        result_builder=render_flux_pick,
    )
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
    assert result.flx_half == state.flux_half
    assert result.flx_int == state.flux_int
    assert result.flx_period == 2 * abs(state.flux_int - state.flux_half)
    plots.finish()
    plots.release()
