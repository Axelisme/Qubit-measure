"""Tests for measure-gui app-layer ambient scopes."""

from __future__ import annotations

from qtpy.QtWidgets import QLabel, QStackedWidget
from zcu_tools.gui.app.main.services.scopes import figure_ambient
from zcu_tools.gui.plotting import (
    FigureContainer,
    attach_existing_figure_to_container,
    get_figure_container,
)
from zcu_tools.liveplot.backend import close_figure


def test_figure_ambient_preserves_container_owned_figure_on_liveplot_close(qapp):
    del qapp
    import matplotlib.pyplot as plt

    stack = QStackedWidget()
    placeholder = QLabel("(placeholder)")
    stack.addWidget(placeholder)
    container = FigureContainer(stack, placeholder)

    fig = plt.figure()
    attach_existing_figure_to_container(fig, container)
    with figure_ambient(container):
        assert get_figure_container(fig) is container

        close_figure(fig)

        assert get_figure_container(fig) is container
        assert stack.currentWidget() is fig.canvas

    plt.close(fig)
