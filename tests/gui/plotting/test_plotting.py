"""Explicit figure attachment, registry and container lifecycle contracts."""

from __future__ import annotations

import weakref

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


def test_attach_existing_figure_to_container(qapp):
    del qapp
    import matplotlib.pyplot as plt

    container, stack = _make_container()
    fig = plt.figure()

    canvas = attach_existing_figure_to_container(fig, container)

    assert stack.count() == 2
    assert stack.currentWidget() is canvas
    assert get_figure_container(fig) is container

    plt.close(fig)


def test_plot_state_snapshot_and_invariants(qapp):
    del qapp
    import matplotlib.pyplot as plt

    # This test observes the public snapshot, not the stack presentation.
    container, _ = _make_container()
    fig = plt.figure()
    attach_existing_figure_to_container(fig, container)

    state = dump_plot_state()

    assert state.active_figure_count >= 1
    assert id(fig) in state.attached_figure_ids
    assert_plot_invariants()

    plt.close(fig)


def test_two_figures_coexist_in_one_container(qapp):
    """Regression: a run/analyze figure and a post-analysis figure share one
    container's stack. Alternating attaches (A, B, A) must NOT delete the other
    figure's canvas — both stay alive, and the last-attached is current.

    This is the exact failure from the post-analysis shared-container bug: the
    old single-slot ``_canvas_widget`` evicted the other figure's canvas, whose
    dead wrapper was then reused on the next attach and crashed.
    """
    del qapp
    import matplotlib.pyplot as plt
    from qtpy import sip

    container, stack = _make_container()
    fig_a = plt.figure()  # run/analyze figure
    fig_b = plt.figure()  # post-analysis figure
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


def test_attach_self_heals_dead_canvas_wrapper(qapp):
    """Defense: if a figure's canvas widget is force-deleted out from under it,
    re-attaching the same figure builds a fresh canvas instead of crashing on
    the dead wrapper."""
    del qapp
    import matplotlib.pyplot as plt
    from qtpy import sip

    container, stack = _make_container()
    fig = plt.figure()
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

    The figure is registered directly (a bare ``Figure``, no canvas in any stack)
    so the only strong reference is the local ``fig``; dropping it must let the
    weak key evict the entry. This isolates weak-eviction from the explicit
    pop paths (remove_canvas / clear_dynamic_canvases), which are the normal —
    but not the only — way an entry leaves the registry.
    """
    del qapp
    import gc

    from matplotlib.figure import Figure
    from zcu_tools.gui.plotting.host import _fig_container_registry

    # Bare-figure eviction observes the registry snapshot, not a canvas stack.
    container, _ = _make_container()
    fig = Figure()
    _fig_container_registry[fig] = container
    assert get_figure_container(fig) is container

    figure_id = id(fig)
    fig_ref = weakref.ref(fig)
    del fig
    gc.collect()

    assert fig_ref() is None, "figure was not GC'd; test cannot prove weak eviction"
    # Other weak entries may also expire during GC; only this figure is ours.
    assert figure_id not in dump_plot_state().attached_figure_ids


def test_new_figure_does_not_detach_other_container(qapp):
    """Root fix: attaching a new figure to container B never detaches the canvas
    of an unrelated container A. After A's figure entry is gone from the registry
    (weak-evicted), B's attach must leave A's current widget untouched (no
    placeholder flip from a detach on the wrong container)."""
    del qapp
    import gc

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from zcu_tools.gui.plotting.host import _fig_container_registry

    container_a, stack_a = _make_container()
    container_b, stack_b = _make_container()

    # A keeps a real, current canvas of its own.
    fig_a = plt.figure()
    canvas_a = attach_existing_figure_to_container(fig_a, container_a)
    assert stack_a.currentWidget() is canvas_a

    # A second figure was once mapped to container_a but its entry then got
    # weak-evicted (the figure is GC'd). With an id-keyed dict this entry would
    # linger and a new figure could alias it; weak keys make it vanish.
    ghost = Figure()
    _fig_container_registry[ghost] = container_a
    del ghost
    gc.collect()

    fig_b = plt.figure()
    try:
        canvas_b = attach_existing_figure_to_container(fig_b, container_b)
        # B got its own canvas; A's container is untouched (still showing A).
        assert stack_b.currentWidget() is canvas_b
        assert stack_a.currentWidget() is canvas_a
    finally:
        plt.close(fig_b)
        plt.close(fig_a)


def test_attach_ignores_stale_previous_container_entry(qapp):
    """Stale-entry defense: if the registry maps a figure to a container that no
    longer hosts its canvas, attaching to a new container must NOT call
    detach_canvas on the stale one (which would flip it to its placeholder)."""
    del qapp
    import matplotlib.pyplot as plt
    from zcu_tools.gui.plotting.host import _fig_container_registry

    stale_container, stale_stack = _make_container()
    target_container, target_stack = _make_container()

    # Give the stale container a real, current canvas of its own so we can detect
    # an erroneous placeholder flip.
    other_fig = plt.figure()
    other_canvas = attach_existing_figure_to_container(other_fig, stale_container)
    assert stale_stack.currentWidget() is other_canvas

    fig = plt.figure()
    try:
        # Craft the stale state: registry says ``fig`` lives in stale_container,
        # but stale_container never hosted fig's canvas.
        _fig_container_registry[fig] = stale_container

        canvas = attach_existing_figure_to_container(fig, target_container)

        # The stale entry was dropped without detaching the unrelated container.
        assert target_stack.currentWidget() is canvas
        assert stale_stack.currentWidget() is other_canvas
        assert get_figure_container(fig) is target_container
    finally:
        plt.close(fig)
        plt.close(other_fig)
