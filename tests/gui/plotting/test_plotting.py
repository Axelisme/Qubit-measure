"""Tests for the shared plotting substrate (``zcu_tools.gui.plotting``).

Mirrors the original per-app plot_host tests against the extracted shared
package, plus a QThreadPool-worker routing test (R4): the headline fluxdep
migration risk is that a ``ContextVar`` set on the main thread is NOT visible to
a ``QRunnable`` pool worker, so the worker must enter its own ``routing_scope``.
This test proves the shared mechanism supports that pattern.
"""

from __future__ import annotations

from qtpy.QtWidgets import QApplication, QLabel, QStackedWidget
from zcu_tools.gui.plotting import (
    FigureContainer,
    assert_plot_invariants,
    attach_existing_figure_to_container,
    close_figure,
    dump_plot_state,
    get_figure_container,
    is_main_thread,
    routing_scope,
    set_shutting_down,
)
from zcu_tools.gui.plotting.routing import (
    get_current_container,
    has_current_container,
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
    assert get_figure_container(fig) is container

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
    """A GuiFigureCanvas.draw_idle on the main thread draws inline (super path)."""
    del qapp
    from matplotlib.figure import Figure
    from zcu_tools.gui.plotting.backend import GuiFigureCanvas

    fig = Figure()
    canvas = GuiFigureCanvas(fig)
    canvas.draw_idle()  # main-thread: inline super() path, must not hang


def test_show_raises_when_figure_not_attached(qapp):
    """DECISION 1: show() / activate Fast-Fails if the figure has no container."""
    import pytest
    from matplotlib.figure import Figure
    from zcu_tools.gui.plotting.backend import GuiFigureCanvas, GuiFigureManager

    del qapp
    fig = Figure()
    canvas = GuiFigureCanvas(fig)
    manager = GuiFigureManager(canvas, 1)
    with pytest.raises(RuntimeError, match="not attached to any FigureContainer"):
        manager.show()


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
    from qtpy import sip  # type: ignore[attr-defined]

    container = _make_container()
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
        assert not sip.isdeleted(canvas_a)  # type: ignore[attr-defined]
        assert not sip.isdeleted(canvas_b)  # type: ignore[attr-defined]

        # Both canvases coexist in the stack (placeholder + 2 canvases).
        assert container._stack.count() == 3
        # Last attached (A) is the visible one.
        assert container._stack.currentWidget() is canvas_a
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
    from qtpy import sip  # type: ignore[attr-defined]

    container = _make_container()
    fig = plt.figure()
    try:
        canvas = attach_existing_figure_to_container(fig, container)

        # Force-delete the canvas widget at the C++ level (simulating a path that
        # deleted it while matplotlib still holds fig.canvas). ``deleteLater`` is
        # not enough here: matplotlib keeps a strong reference so the DeferredDelete
        # never collects the C++ object — ``sip.delete`` is the deterministic kill.
        container.detach_canvas(canvas)
        sip.delete(canvas)  # type: ignore[attr-defined]
        QApplication.instance().processEvents()  # type: ignore[union-attr]
        assert sip.isdeleted(canvas)  # type: ignore[attr-defined]

        # Re-attach must not raise; it creates a fresh, live canvas.
        fresh = attach_existing_figure_to_container(fig, container)
        assert fresh is not canvas
        assert not sip.isdeleted(fresh)  # type: ignore[attr-defined]
        assert container._stack.currentWidget() is fresh
    finally:
        plt.close(fig)


# --- R4: a QThreadPool worker entering its own routing_scope routes correctly --
#
# Must run in a subprocess: the pytest process can't `matplotlib.use(BACKEND_NAME)`
# (pyplot is already imported), so plt.figure() there would not hit our backend.
# The subprocess configures the backend, runs the fluxdep pattern (a QThreadPool
# worker that enters routing_scope ITSELF and calls plt.figure()), and pumps the
# Qt loop so the worker→main-thread attach emit is serviced. This is the headline
# fluxdep migration check (R4: a main-thread ContextVar is invisible to a pool
# worker, so the worker enters the scope).


def test_pool_worker_routes_via_own_scope():
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent(
        """
        import threading
        from zcu_tools.gui.plotting.setup import configure_matplotlib_backend
        configure_matplotlib_backend()

        from qtpy.QtCore import QCoreApplication, QRunnable, QThreadPool
        from qtpy.QtWidgets import QApplication, QLabel, QStackedWidget
        from zcu_tools.gui.plotting import (
            FigureContainer, routing_scope, get_figure_container,
        )
        import matplotlib.pyplot as plt

        app = QApplication.instance() or QApplication([])
        stack = QStackedWidget()
        ph = QLabel("(placeholder)")
        stack.addWidget(ph)
        container = FigureContainer(stack, ph)

        done = threading.Event()
        out = {}

        class W(QRunnable):
            def run(self):
                try:
                    # The worker enters its OWN routing_scope (the fluxdep pattern);
                    # it does NOT inherit a main-thread ContextVar.
                    with routing_scope(container):
                        fig = plt.figure()
                        out["routed"] = get_figure_container(fig) is container
                        out["manager"] = type(fig.canvas.manager).__name__
                        plt.close(fig)
                except BaseException as exc:
                    out["err"] = repr(exc)
                finally:
                    done.set()

        QThreadPool.globalInstance().start(W())
        waited = 0.0
        while not done.is_set() and waited < 8.0:
            app.processEvents()
            done.wait(0.02)
            waited += 0.02

        assert done.is_set(), "worker did not finish"
        assert out.get("err") is None, out.get("err")
        assert out.get("manager") == "GuiFigureManager", out
        assert out.get("routed") is True, "worker plt.figure() did not route to its container"
        print("ok")
        """
    )
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
