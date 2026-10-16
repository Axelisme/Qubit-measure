"""Notebook backend diagnostics through its public presentation entries."""

import inspect
from collections.abc import Callable

import pytest
from matplotlib.figure import Figure
from zcu_tools.plotting.liveplot import instant_plot
from zcu_tools.plotting.liveplot.backend import JupyterBackend, set_liveplot_backend
from zcu_tools.plotting.liveplot.backend.jupyter import (
    instant_plot as notebook_instant_plot,
)


@pytest.mark.parametrize("present", [instant_plot, notebook_instant_plot])
def test_non_widget_warning_identifies_the_presentation_caller(
    present: Callable[[Figure], None],
) -> None:
    # A real non-widget canvas still fails at layout after the diagnostic.
    # Do not replace the canvas or accidentally add a new fallback policy.
    figure = Figure()
    with (
        set_liveplot_backend(JupyterBackend()),
        pytest.warns(
            UserWarning,
            match="^Warning: The matplotlib backend should be set to 'widget' for live plotting\\.$",
        ) as recorded,
    ):
        call_line = inspect.stack()[0].lineno + 2
        with pytest.raises(AttributeError, match="layout"):
            present(figure)
    assert len(recorded) == 1
    assert recorded[0].filename == __file__
    assert recorded[0].lineno == call_line
