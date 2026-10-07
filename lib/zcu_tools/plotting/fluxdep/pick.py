"""Native terminal flux-pick rendering without presentation or ownership."""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from zcu_tools.analysis.fluxdep.line_picker import TwoLinePicker
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs, FluxPickState


def configure_flux_pick_axes(figure: Figure) -> None:
    """Enable coordinate grids on visible axes of an already-built picker figure.

    Call after constructing the axes, for either a Qt preview or native output.
    Hidden mirror-loss axes stay hidden. Limits, labels, analysis state, canvas
    ownership and Figure registration are unchanged. Matplotlib errors propagate.
    """
    for axes in figure.axes:
        if axes.axison:
            axes.grid(visible=True, color="white", alpha=0.3, linewidth=0.7)


def make_flux_pick_figure(inputs: FluxPickInputs, state: FluxPickState) -> Figure:
    """Return a separate Agg figure showing the accepted device-axis selection."""
    figure = Figure(figsize=(8, 5))
    FigureCanvasAgg(figure)
    picker = TwoLinePicker(
        figure,
        inputs.signals,
        inputs.dev_values,
        inputs.freqs,
        flux_half=state.flux_half,
        flux_int=state.flux_int,
        force_magnitude=state.magnitude_only,
    )
    picker.show_state(state)
    configure_flux_pick_axes(figure)
    return figure
