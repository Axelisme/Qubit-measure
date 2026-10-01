"""Native terminal flux-pick rendering without presentation or ownership."""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from zcu_tools.analysis.fluxdep.line_picker import TwoLinePicker
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs, FluxPickState


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
    return figure
