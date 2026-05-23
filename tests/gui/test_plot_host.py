from __future__ import annotations

import pytest
from qtpy.QtWidgets import QLabel, QStackedWidget
from zcu_tools.gui.plot_host import (
    FigureContainer,
    attach_existing_figure_to_container,
    create_figure_in_active_container,
    has_container,
    pop_container,
    push_container,
)


def _make_container() -> FigureContainer:
    stack = QStackedWidget()
    placeholder = QLabel("(placeholder)")
    stack.addWidget(placeholder)
    return FigureContainer(stack, placeholder)


def _clear_container_stack() -> None:
    while has_container():
        pop_container()


def test_push_pop_container_round_trip(qapp):
    del qapp
    _clear_container_stack()
    container = _make_container()

    push_container(container)
    assert has_container() is True
    popped = pop_container(container)

    assert popped is container
    assert has_container() is False


def test_pop_non_top_container_raises(qapp):
    del qapp
    _clear_container_stack()
    container1 = _make_container()
    container2 = _make_container()

    push_container(container1)
    push_container(container2)

    with pytest.raises(RuntimeError, match="non-top pop"):
        pop_container(container1)

    pop_container(container2)
    pop_container(container1)


def test_create_figure_requires_active_container():
    _clear_container_stack()

    with pytest.raises(RuntimeError, match="No active FigureContainer"):
        create_figure_in_active_container(1, 1)


def test_attach_existing_figure_to_container(qapp):
    del qapp
    _clear_container_stack()

    import matplotlib.pyplot as plt

    container = _make_container()
    fig = plt.figure()

    canvas = attach_existing_figure_to_container(fig, container)

    assert container._stack.count() == 2
    assert container._stack.currentWidget() is canvas

    plt.close(fig)


def test_create_figure_in_active_container(qapp):
    del qapp
    _clear_container_stack()

    push_container(_make_container())
    fig, axs = create_figure_in_active_container(1, 1)

    assert fig is not None
    assert len(axs) == 1
    assert len(axs[0]) == 1

    pop_container()


def test_auto_select_backend_prefers_qt_when_container_is_active(qapp):
    del qapp
    _clear_container_stack()

    from zcu_tools.liveplot.backend import auto_select_backend

    push_container(_make_container())

    backend = auto_select_backend()

    assert backend.__name__.endswith(".qt")

    pop_container()
