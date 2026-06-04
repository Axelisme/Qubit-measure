"""Tests for the fluxdep embedded-matplotlib substrate (plot_host + backend setup).

Headless: a FigureContainer over a real QStackedWidget, exercised on the main
(test) thread. The worker-thread marshalling path is covered by the higher-level
search smoke; here we pin the container attach/clear behaviour and the
backend-setup invariant.
"""

from __future__ import annotations

import matplotlib
import pytest
from matplotlib.figure import Figure
from qtpy.QtWidgets import QLabel, QStackedWidget  # type: ignore[attr-defined]
from zcu_tools.fluxdep_gui.ui.plot_host import (
    FigureContainer,
    ensure_bridge,
    get_figure_container,
    require_current_container,
    set_current_container,
    use_container,
)


@pytest.fixture
def container(qapp):
    ensure_bridge()
    stack = QStackedWidget()
    placeholder = QLabel("placeholder")
    stack.addWidget(placeholder)
    cont = FigureContainer(stack, placeholder)
    yield cont, stack, placeholder
    set_current_container(None)
    stack.deleteLater()


def test_attach_canvas_shows_it(container):
    cont, stack, placeholder = container
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    fig = Figure()
    canvas = FigureCanvasQTAgg(fig)
    cont.attach_canvas(canvas)
    assert stack.count() == 2
    assert stack.currentWidget() is canvas


def test_clear_back_to_placeholder(container):
    cont, stack, placeholder = container
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    cont.attach_canvas(FigureCanvasQTAgg(Figure()))
    cont.clear()
    assert stack.count() == 1
    assert stack.currentWidget() is placeholder


def test_use_container_sets_and_restores():
    assert _current_is_none()
    stack = QStackedWidget()
    ph = QLabel()
    stack.addWidget(ph)
    cont = FigureContainer(stack, ph)
    with use_container(cont):
        assert require_current_container() is cont
    assert _current_is_none()


def test_set_current_container_explicit():
    stack = QStackedWidget()
    ph = QLabel()
    stack.addWidget(ph)
    cont = FigureContainer(stack, ph)
    set_current_container(cont)
    assert require_current_container() is cont
    set_current_container(None)
    assert _current_is_none()


def test_require_current_container_fast_fails_when_unset():
    set_current_container(None)
    with pytest.raises(RuntimeError, match="no current FigureContainer"):
        require_current_container()


def test_get_figure_container_none_for_unattached():
    assert get_figure_container(Figure()) is None


def _current_is_none() -> bool:
    try:
        require_current_container()
        return False
    except RuntimeError:
        return True


# --- backend setup invariant ----------------------------------------------


def test_backend_name_matches_module():
    from zcu_tools.fluxdep_gui.ui.mpl_backend_setup import BACKEND_NAME

    assert BACKEND_NAME == "module://zcu_tools.fluxdep_gui.ui.mpl_backend"


def test_backend_is_selectable():
    # The custom backend must be importable + registerable by matplotlib.
    # (It is already active in this test process if run after the GUI, but
    # selecting it again must not raise.)
    matplotlib.use("module://zcu_tools.fluxdep_gui.ui.mpl_backend", force=True)
    assert matplotlib.get_backend() == "module://zcu_tools.fluxdep_gui.ui.mpl_backend"
