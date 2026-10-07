"""Measure-specific capture and terminal binding for shared line picking."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
    FluxPickState,
    analyze_flux_pick,
    fold_initial_lines,
)
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest
from zcu_tools.gui.interactive.flux_pick import SharedFluxPickPlugin
from zcu_tools.plotting.fluxdep.pick import make_flux_pick_figure
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.interactive_flux_pick import FluxPickResult


def render_flux_pick(
    inputs: FluxPickInputs, state: FluxPickState, plots: Plots
) -> FluxPickResult:
    """Build a GUI-owned terminal result from captured inputs and committed state."""
    analysis = analyze_flux_pick(inputs, state)
    plots.adopt("pick", make_flux_pick_figure(inputs, state))
    return FluxPickResult(
        flx_half=analysis.flux_half,
        flx_int=analysis.flux_int,
        flx_period=analysis.flux_period,
    )


class FluxPickPlugin(SharedFluxPickPlugin[FluxPickResult]):
    """Measure result binding; shared actions and worker lifecycle live in lib."""

    def __init__(
        self,
        inputs: FluxPickInputs,
        seed: FluxPickState,
        *,
        plots: Plots,
        result_builder: Callable[
            [FluxPickInputs, FluxPickState, Plots], FluxPickResult
        ],
    ) -> None:
        """Capture inputs and build the measure result/figure after accepted Finish.

        plots receives the named pick figure from result_builder. The builder
        owns measure result policy; exceptions leave session input terminal.
        """
        super().__init__(
            inputs,
            seed,
            build_result=lambda state: result_builder(inputs, state, plots),
        )


def make_flux_pick_plugin(
    req: AnalyzeRequest[Any, Any],
    *,
    force_magnitude: bool,
    plots: Plots,
    result_builder: Callable[[FluxPickInputs, FluxPickState, Plots], FluxPickResult],
) -> FluxPickPlugin:
    """Capture a read-only spectrum, fold the two calibrated seed positions."""
    result = req.run_result
    inputs = FluxPickInputs(result.signals, result.values, result.freqs)
    half, integer = fold_initial_lines(
        inputs.dev_values, req.md.get("flx_half", None), req.md.get("flx_int", None)
    )
    return FluxPickPlugin(
        inputs,
        FluxPickState(flux_half=half, flux_int=integer, magnitude_only=force_magnitude),
        plots=plots,
        result_builder=result_builder,
    )
