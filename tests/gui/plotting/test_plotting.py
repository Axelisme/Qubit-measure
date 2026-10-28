"""Explicit figure attachment, registry and container lifecycle contracts."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator

import pytest
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_qt import FigureManagerQT
from matplotlib.figure import Figure
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication, QLabel, QStackedWidget
from zcu_tools.gui.plotting import (
    FigureContainer,
    assert_plot_invariants,
    attach_existing_figure_to_container,
    dump_plot_state,
    get_figure_container,
)


def _make_container() -> tuple[FigureContainer, QStackedWidget]:
    stack = QStackedWidget()
    placeholder = QLabel("(placeholder)")
    stack.addWidget(placeholder)
    return FigureContainer(stack, placeholder), stack


@pytest.fixture
def make_figure(qapp: QApplication) -> Iterator[Callable[[], Figure]]:
    """Delete the Qt manager windows created by this test's pyplot figures."""
    import matplotlib.pyplot as plt

    owned: list[tuple[Figure, FigureManagerBase | None]] = []

    def make() -> Figure:
        figure = plt.figure()
        owned.append((figure, figure.canvas.manager))
        return figure

    yield make

    for figure, manager in reversed(owned):
        # pyplot.close hides the manager window but does not delete its QObject.
        try:
            plt.close(figure)
        finally:
            if isinstance(manager, FigureManagerQT):
                manager.window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    qapp.processEvents()


@pytest.fixture
def make_container(
    qapp: QApplication,
) -> Iterator[Callable[[], tuple[FigureContainer, QStackedWidget]]]:
    """Own presentation cleanup even when an attachment assertion fails."""
    owned: list[tuple[FigureContainer, QStackedWidget]] = []

    def make() -> tuple[FigureContainer, QStackedWidget]:
        presentation = _make_container()
        owned.append(presentation)
        return presentation

    yield make

    for container, stack in reversed(owned):
        try:
            container.clear_dynamic_canvases()
        finally:
            stack.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    qapp.processEvents()


def test_attach_existing_figure_to_container(make_container, make_figure):
    import matplotlib.pyplot as plt

    container, stack = make_container()
    fig = make_figure()

    canvas = attach_existing_figure_to_container(fig, container)

    assert stack.count() == 2
    assert stack.currentWidget() is canvas
    assert get_figure_container(fig) is container

    plt.close(fig)


def test_plot_state_snapshot_and_invariants(make_container, make_figure):
    import matplotlib.pyplot as plt

    # This test observes the public snapshot, not the stack presentation.
    container, _ = make_container()
    fig = make_figure()
    attach_existing_figure_to_container(fig, container)

    state = dump_plot_state()

    assert state.active_figure_count >= 1
    assert id(fig) in state.attached_figure_ids
    assert_plot_invariants()

    plt.close(fig)


def test_snapshot_identifies_live_container_and_forgets_deleted_owner(qapp):
    """Diagnostics name the live owner, not a figure's pending Python GC cycle."""
    from matplotlib.figure import Figure

    container, stack = _make_container()
    stack.setObjectName("snapshot-owner")
    figure = Figure()
    try:
        attach_existing_figure_to_container(figure, container)
        attachments = dump_plot_state().attached_figures
        matching = [entry for entry in attachments if entry.figure_id == id(figure)]
        assert len(matching) == 1
        assert matching[0].container_object_name == "snapshot-owner"

    finally:
        stack.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    assert id(figure) not in dump_plot_state().attached_figure_ids


def test_two_figures_coexist_in_one_container(make_container, make_figure):
    """Regression: a run/analyze figure and a post-analysis figure share one
    container's stack. Alternating attaches (A, B, A) must NOT delete the other
    figure's canvas — both stay alive, and the last-attached is current.

    This is the exact failure from the post-analysis shared-container bug: the
    old single-slot ``_canvas_widget`` evicted the other figure's canvas, whose
    dead wrapper was then reused on the next attach and crashed.
    """
    import matplotlib.pyplot as plt
    from qtpy import sip

    container, stack = make_container()
    fig_a = make_figure()  # run/analyze figure
    fig_b = make_figure()  # post-analysis figure
    try:
        canvas_a = attach_existing_figure_to_container(fig_a, container)
        canvas_b = attach_existing_figure_to_container(fig_b, container)
        # Re-attaching A simulates the per-content-change re-render order
        # (analyze figure rendered, then post figure) repeating.
        canvas_a_again = attach_existing_figure_to_container(fig_a, container)

        # Same figure -> same (live) canvas reused, not a fresh dead wrapper.
        assert canvas_a_again is canvas_a
        assert not sip.isdeleted(canvas_a)
        assert not sip.isdeleted(canvas_b)

        # Both canvases coexist in the stack (placeholder + 2 canvases).
        assert stack.count() == 3
        # Last attached (A) is the visible one.
        assert stack.currentWidget() is canvas_a
        assert get_figure_container(fig_a) is container
        assert get_figure_container(fig_b) is container
    finally:
        plt.close(fig_a)
        plt.close(fig_b)


def test_attach_self_heals_dead_canvas_wrapper(make_container, make_figure):
    """Defense: if a figure's canvas widget is force-deleted out from under it,
    re-attaching the same figure builds a fresh canvas instead of crashing on
    the dead wrapper."""
    import matplotlib.pyplot as plt
    from qtpy import sip

    container, stack = make_container()
    fig = make_figure()
    try:
        canvas = attach_existing_figure_to_container(fig, container)

        # Force-delete the canvas widget at the C++ level (simulating a path that
        # deleted it while matplotlib still holds fig.canvas). ``deleteLater`` is
        # not enough here: matplotlib keeps a strong reference so the DeferredDelete
        # never collects the C++ object — ``sip.delete`` is the deterministic kill.
        container.detach_canvas(canvas)
        sip.delete(canvas)
        app = QApplication.instance()
        assert isinstance(app, QApplication)
        app.processEvents()
        assert sip.isdeleted(canvas)

        # Re-attach must not raise; it creates a fresh, live canvas.
        fresh = attach_existing_figure_to_container(fig, container)
        assert fresh is not canvas
        assert not sip.isdeleted(fresh)
        assert stack.currentWidget() is fresh
    finally:
        plt.close(fig)


def test_registry_evicts_gc_collected_figure(qapp):
    """Root fix: the registry is weak-keyed, so once a figure is GC'd its entry
    vanishes automatically — no stale ``id(fig)`` entry survives to be aliased by
    a later figure that happens to reuse the collected id.

    Without weak keys, the entry lingered forever (purge only ran in diagnostics)
    and CPython id-reuse let a NEW figure hit the stale entry pointing at a
    different container — the intermittent "analyze figure not displayed" bug.

    Public attachment registers the figure. Detaching the canvas and clearing
    its Qt parent releases presentation ownership without popping the mapping.
    Dropping both locals must let GC collect the figure/canvas cycle. The weakref
    assertion precedes diagnostics, so a strong registry cannot pass by purging
    the detached entry in dump_plot_state().
    """
    import gc

    from matplotlib.figure import Figure

    container, stack = _make_container()
    fig = Figure()
    try:
        canvas = attach_existing_figure_to_container(fig, container)
        assert get_figure_container(fig) is container

        container.detach_canvas(canvas)
        canvas.setParent(None)
        figure_id = id(fig)
        fig_ref = weakref.ref(fig)
        del canvas
        del fig
        gc.collect()

        assert fig_ref() is None, "figure was not GC'd; test cannot prove weak eviction"
        # Other weak entries may also expire during GC; only this figure is ours.
        assert figure_id not in dump_plot_state().attached_figure_ids
    finally:
        try:
            container.clear_dynamic_canvases()
        finally:
            stack.deleteLater()
            qapp.processEvents()


def test_new_figure_does_not_detach_other_container(qapp, make_figure):
    """Root fix: attaching a new figure to container B never detaches the canvas
    of an unrelated container A. A ghost figure is publicly attached to A, then
    detached and released for GC without popping its registry mapping. After
    restoring A's current canvas and collecting the ghost, B's attach must leave
    A's current widget untouched (no placeholder flip on the wrong container).
    """
    import gc

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    container_a, stack_a = _make_container()
    try:
        container_b, stack_b = _make_container()
        try:
            # A keeps a real, current canvas of its own.
            fig_a = make_figure()
            try:
                canvas_a = attach_existing_figure_to_container(fig_a, container_a)
                assert stack_a.currentWidget() is canvas_a

                # Public attachment registers the ghost. Detaching and releasing
                # its Qt parent lets GC collect the figure/canvas cycle without
                # purging the mapping. Restore A after the ghost's selection.
                ghost = Figure()
                canvas_ghost = attach_existing_figure_to_container(ghost, container_a)
                container_a.detach_canvas(canvas_ghost)
                canvas_ghost.setParent(None)
                container_a.set_current_canvas(canvas_a)
                ghost_ref = weakref.ref(ghost)
                del canvas_ghost, ghost
                gc.collect()

                assert ghost_ref() is None, (
                    "ghost was not GC'd; test requires weak eviction"
                )

                fig_b = make_figure()
                try:
                    canvas_b = attach_existing_figure_to_container(fig_b, container_b)
                    # B got its own canvas; A is untouched (still showing A).
                    assert stack_b.currentWidget() is canvas_b
                    assert stack_a.currentWidget() is canvas_a
                finally:
                    plt.close(fig_b)
            finally:
                plt.close(fig_a)
        finally:
            try:
                container_b.clear_dynamic_canvases()
            finally:
                stack_b.deleteLater()
    finally:
        try:
            container_a.clear_dynamic_canvases()
        finally:
            try:
                stack_a.deleteLater()
            finally:
                qapp.processEvents()


def test_attach_ignores_stale_previous_container_entry(make_container, make_figure):
    """Stale-entry defense: if the registry maps a figure to a container that no
    longer hosts its canvas, attaching to a new container must NOT call
    detach_canvas on the stale one (which would flip it to its placeholder)."""
    import matplotlib.pyplot as plt

    stale_container, stale_stack = make_container()
    target_container, target_stack = make_container()

    other_fig = make_figure()
    fig = make_figure()
    try:
        # Detach presentation without dropping the figure's container mapping.
        stale_canvas = attach_existing_figure_to_container(fig, stale_container)
        stale_container.detach_canvas(stale_canvas)
        assert get_figure_container(fig) is stale_container
        assert stale_stack.indexOf(stale_canvas) == -1

        # Give the stale container a current canvas to detect a placeholder flip.
        other_canvas = attach_existing_figure_to_container(other_fig, stale_container)
        assert stale_stack.currentWidget() is other_canvas

        canvas = attach_existing_figure_to_container(fig, target_container)

        # The stale entry was dropped without detaching the unrelated container.
        assert target_stack.currentWidget() is canvas
        assert stale_stack.currentWidget() is other_canvas
        assert get_figure_container(fig) is target_container
    finally:
        plt.close(fig)
        plt.close(other_fig)
