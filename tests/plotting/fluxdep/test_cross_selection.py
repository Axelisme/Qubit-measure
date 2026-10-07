"""Native builder and shared renderer preserve caller canvas and state."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from zcu_tools.analysis.fluxdep.cross_selection import (
    CrossSelectionInputs,
    make_cross_selection_state,
    project_cross_selection,
    stroke_cross_selection_state,
)
from zcu_tools.analysis.fluxdep.stroke import BrushPoint, BrushStroke
from zcu_tools.plotting.fluxdep.cross_selection import (
    CrossSelectionPlot,
    make_cross_selection_figure,
)


def test_native_builder_and_renderer_leave_state_and_caller_canvas_untouched():
    inputs = CrossSelectionInputs(
        np.array([0.0, 0.5, 1.0]),
        np.array([4.0, 4.5, 5.0]),
        (),
        (0.0, 1.0),
        (4.0, 5.0),
    )
    seed = make_cross_selection_state(inputs)
    state = stroke_cross_selection_state(
        inputs, seed, BrushStroke((BrushPoint(0.5, 4.5),), 0.02, "erase")
    )
    kept = state.selected.copy()
    assert state.last_change is not None
    previous = state.last_change.selected.copy()
    first = make_cross_selection_figure(inputs, state, show_changes=True)
    second = make_cross_selection_figure(
        inputs, seed, previous=state, show_changes=True
    )
    assert first is not second
    assert first.canvas is not second.canvas

    supplied = Figure()
    canvas = FigureCanvasAgg(supplied)
    renderer = CrossSelectionPlot(supplied, inputs)
    renderer.show_state(project_cross_selection(inputs, state), show_changes=True)
    assert supplied.canvas is canvas
    np.testing.assert_array_equal(state.selected, kept)
    np.testing.assert_array_equal(state.last_change.selected, previous)
    assert seed.selected.all()
