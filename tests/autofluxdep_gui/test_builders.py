"""Per-experiment Builder structural tests — registry + Plotter layout.

The Nodes' real produce path (set flux device -> acquire -> fit) is covered
end-to-end against the flux-aware MockSoc by the ``test_*_acquire.py`` integration
tests. These tests stay at the structural level the integration tests do not
touch: the registry exposes every experiment type, and each Builder's Plotter
embeds the runner module's subplot layout (the matching number of matplotlib
axes).
"""

from __future__ import annotations

import pytest

# --- registry exposes all migrated measurement types ---


def test_registry_exposes_all_experiments():
    from zcu_tools.gui.app.autofluxdep.registry import available_node_types

    types = set(available_node_types())
    assert types == {
        "qubit_freq",
        "lenrabi",
        "ro_optimize",
        "t1",
        "t2ramsey",
        "t2echo",
        "mist",
    }


# --- liveplot alignment: each Builder builds the runner module's subplot layout ---


@pytest.mark.parametrize(
    ("type_name", "n_axes"),
    [
        ("qubit_freq", 3),  # fit_freq (1) + detune 2DwithLine (2)
        ("lenrabi", 2),  # rabi_curve 2DwithLine (2D + line)
        ("ro_optimize", 1),  # snr 2D landscape
        ("t1", 2),  # scalar scatter + current-point curve
        ("t2ramsey", 2),
        ("t2echo", 2),
        ("mist", 2),  # flux×gain 2DwithLine
    ],
)
def test_make_plotter_builds_aligned_subplots(type_name, n_axes):
    # each experiment's Plotter embeds the same LivePlot panels the runner module
    # draws, so the figure has the matching number of axes.
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.autofluxdep.registry import create_placement

    builder = create_placement(type_name).builder
    figure = Figure()
    plotter = builder.make_plotter(figure)
    assert plotter is not None
    assert len(figure.axes) == n_axes
