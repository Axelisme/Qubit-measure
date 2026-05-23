from __future__ import annotations

import pytest
from qtpy.QtWidgets import QLabel, QStackedWidget
from zcu_tools.gui.plot_host import (
    FigureContainer,
    assert_plot_invariants,
    attach_existing_figure_to_container,
    create_figure_in_current_container,
    dump_plot_state,
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


def test_create_figure_requires_current_container():
    with pytest.raises(RuntimeError, match="No active FigureContainer"):
        create_figure_in_current_container(1, 1)


def test_attach_existing_figure_to_container(qapp):
    del qapp

    import matplotlib.pyplot as plt

    container = _make_container()
    fig = plt.figure()

    canvas = attach_existing_figure_to_container(fig, container)

    assert container._stack.count() == 2
    assert container._stack.currentWidget() is canvas

    plt.close(fig)


def test_create_figure_in_current_container(qapp):
    del qapp

    with routing_scope(_make_container()):
        fig, axs = create_figure_in_current_container(1, 1)

    assert fig is not None
    assert len(axs) == 1
    assert len(axs[0]) == 1


def test_auto_select_backend_prefers_qt_when_container_is_active(qapp):
    del qapp

    from zcu_tools.liveplot.backend import auto_select_backend

    with routing_scope(_make_container()):
        backend = auto_select_backend()

    assert backend.__name__.endswith(".qt")


def test_plot_state_snapshot_and_invariants(qapp):
    del qapp

    container = _make_container()
    with routing_scope(container):
        fig, _ = create_figure_in_current_container(1, 1)

    state = dump_plot_state()

    assert state.active_figure_count >= 1
    assert id(fig) in state.attached_figure_ids
    assert_plot_invariants()
