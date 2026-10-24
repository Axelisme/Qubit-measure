"""Measure-specific capture and terminal binding for shared line picking."""

from __future__ import annotations

from collections.abc import Callable

from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
    FluxPickState,
    analyze_flux_pick,
    fold_initial_lines,
)
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickResult
from zcu_tools.gui.interactive.flux_pick import SharedFluxPickPlugin
from zcu_tools.plotting.fluxdep.pick import make_flux_pick_figure
from zcu_tools.plotting.plots import Plots


def render_flux_pick(
    inputs: FluxPickInputs, state: FluxPickState, plots: Plots
) -> FluxPickResult:
    """Validate a committed pick, adopt plots["pick"], and return numeric positions.

    inputs contains spectrum/device/frequency axes; state is the committed
    FluxPickState, not a frontend preview. plots is the operation figure owner.
    Insufficient line separation raises ValueError before figure adoption;
    figure conflicts and rendering failures propagate. No context is read.
    """
    analysis = analyze_flux_pick(inputs, state)
    plots.adopt("pick", make_flux_pick_figure(inputs, state))
    return FluxPickResult(
        flx_half=analysis.flux_half,
        flx_int=analysis.flux_int,
        flx_period=analysis.flux_period,
    )


class FluxPickPlugin[R: FluxPickResult](SharedFluxPickPlugin[R]):
    """Measure result binding; shared actions and worker lifecycle live in lib."""

    def __init__(
        self,
        inputs: FluxPickInputs,
        seed: FluxPickState,
        *,
        plots: Plots,
        result_builder: Callable[[FluxPickInputs, FluxPickState, Plots], R],
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


def make_flux_pick_plugin[R: FluxPickResult](
    inputs: FluxPickInputs,
    *,
    flux_half: float | None = None,
    flux_int: float | None = None,
    force_magnitude: bool,
    plots: Plots,
    result_builder: Callable[[FluxPickInputs, FluxPickState, Plots], R],
) -> FluxPickPlugin[R]:
    """Capture a spectrum and initial device-axis positions for measure picking.

    inputs holds complex spectra with device/frequency axes. flux_half and
    flux_int are finite device-axis scalars; None selects the line kernel's
    default. Folding preserves the existing scan-axis policy. force_magnitude
    sets the initial display mode, not the data. plots receives the named pick
    figure via result_builder after Finish. Invalid inputs or non-finite
    positions raise TypeError/ValueError; result-builder failures propagate.
    No experiment result or parameter container is read here.
    """
    half, integer = fold_initial_lines(inputs.dev_values, flux_half, flux_int)
    return FluxPickPlugin(
        inputs,
        FluxPickState(flux_half=half, flux_int=integer, magnitude_only=force_magnitude),
        plots=plots,
        result_builder=result_builder,
    )
