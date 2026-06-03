from __future__ import annotations

from qtpy.QtWidgets import QLabel, QStackedWidget
from zcu_tools.gui.plot_host import (
    FigureContainer,
    assert_plot_invariants,
    attach_existing_figure_to_container,
    close_figure,
    dump_plot_state,
    is_main_thread,
    set_shutting_down,
)
from zcu_tools.gui.plot_routing import (
    get_current_container,
    has_current_container,
    routing_scope,
)


def _make_container() -> FigureContainer:
    stack = QStackedWidget()
    placeholder = QLabel("(placeholder)")
    stack.addWidget(placeholder)
    return FigureContainer(stack, placeholder)


def test_routing_scope_round_trip(qapp):
    del qapp
    container = _make_container()

    assert has_current_container() is False
    with routing_scope(container):
        assert has_current_container() is True
        assert get_current_container() is container

    assert has_current_container() is False
    assert get_current_container() is None


def test_attach_existing_figure_to_container(qapp):
    del qapp

    import matplotlib.pyplot as plt

    container = _make_container()
    fig = plt.figure()

    canvas = attach_existing_figure_to_container(fig, container)

    assert container._stack.count() == 2
    assert container._stack.currentWidget() is canvas

    plt.close(fig)


def test_plot_state_snapshot_and_invariants(qapp):
    del qapp

    import matplotlib.pyplot as plt

    container = _make_container()
    fig = plt.figure()
    attach_existing_figure_to_container(fig, container)

    state = dump_plot_state()

    assert state.active_figure_count >= 1
    assert id(fig) in state.attached_figure_ids
    assert_plot_invariants()

    plt.close(fig)


def test_is_main_thread_true_on_gui_thread(qapp):
    del qapp
    assert is_main_thread() is True


def test_gui_canvas_draw_idle_inline_on_main_thread(qapp):
    """A GuiFigureCanvas.draw_idle on the main thread draws inline (super path),
    not via the bridge — verified by completing without hanging.

    (The pyplot→GuiFigureCanvas routing is covered by test_mpl_backend.py in a
    subprocess; here we exercise the overridden draw_idle directly.)
    """
    del qapp

    from matplotlib.figure import Figure

    from zcu_tools.gui.mpl_backend import GuiFigureCanvas

    fig = Figure()
    canvas = GuiFigureCanvas(fig)
    # Main-thread call: takes the inline super() path, must not raise / hang.
    canvas.draw_idle()


def test_close_figure_noop_when_shutting_down(qapp):
    del qapp

    import matplotlib.pyplot as plt

    fig = plt.figure()
    try:
        set_shutting_down(True)
        container = _make_container()
        attach_existing_figure_to_container(fig, container)
        close_figure(fig)
    finally:
        set_shutting_down(False)
        plt.close(fig)
