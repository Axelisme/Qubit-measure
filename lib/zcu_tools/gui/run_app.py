"""Shared Qt-app bootstrap for the GUI apps.

``run_qt_app`` owns the parts every app's ``run_app`` repeats verbatim — the
``QApplication`` get-or-create, the ``ensure_host`` / ``set_shutting_down`` plot
host lifecycle, building the window, optionally starting a remote-control
adapter, and entering the Qt event loop. The app-specific *construction* (State
+ Controller, which window class, which adapter type) stays in each app's
``run_app`` and is injected as factories.

The plot host must be created on the Qt main thread before any worker plots, so
``ensure_host`` runs here right after the ``QApplication`` exists; ``aboutToQuit
→ set_shutting_down(True)`` marks the host down before Qt deletes its widgets so
the matplotlib atexit hook becomes a no-op instead of touching deleted widgets.

Import-clean enough to live at the ``gui/`` top level: the only heavy imports
(qtpy, the plot host) are deferred into the function body, exactly as the apps
do them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

Ctrl = TypeVar("Ctrl")
Window = TypeVar("Window")


def run_qt_app(
    *,
    controller_factory: Callable[[], Ctrl],
    window_factory: Callable[[Ctrl], Window],
    control: Any | None = None,
    adapter_factory: Callable[[Ctrl, Any], Any] | None = None,
) -> int:
    """Build and run a GUI app on the Qt event loop. Returns the exit code.

    ``controller_factory()`` builds the app's State + Controller (and anything
    else it needs) and returns the Controller; ``window_factory(ctrl)`` builds
    the MainWindow, which is shown before the event loop starts.

    When ``control`` is given AND ``adapter_factory`` is provided,
    ``adapter_factory(ctrl, control)`` builds a remote-control adapter; it is
    started and stopped on the Qt main thread (``stop`` unsubscribes the
    EventBus synchronously, so it must run there — ``aboutToQuit`` fires on the
    main thread).
    """
    import sys

    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    from zcu_tools.gui.plotting import ensure_host, set_shutting_down

    app = QApplication.instance() or QApplication(sys.argv)

    # Create the plot host now, on the Qt main thread: the shared embedded
    # matplotlib backend marshals worker-thread figure work through it, and it
    # must be built on the GUI thread before any worker plots.
    ensure_host()
    # On teardown, mark the plot host down BEFORE Qt deletes its widgets, so the
    # matplotlib atexit hook (Gcf.destroy_all → backend destroy → remove_canvas)
    # becomes a no-op instead of touching a deleted container.
    app.aboutToQuit.connect(lambda: set_shutting_down(True))

    ctrl = controller_factory()
    window = window_factory(ctrl)
    window.show()  # type: ignore[attr-defined]

    if control is not None and adapter_factory is not None:
        adapter = adapter_factory(ctrl, control)
        try:
            adapter.start()
        except RuntimeError as exc:
            # With the default port, a busy port already fell back to an ephemeral
            # one inside start(); reaching here means either an explicitly-pinned
            # port is taken or even the ephemeral bind failed.
            port = getattr(control, "port", "?")
            print(
                f"\nERROR: cannot open control socket on port {port}.\n"
                f"  {exc}\n\n"
                f"  That port is pinned and already in use.\n"
                f"  Pass a different --control-port <N>, omit it to auto-pick a\n"
                f"  free port, or --no-control to disable the remote-control socket.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        # stop() unsubscribes EventBus synchronously, so it must run on the Qt
        # main thread; aboutToQuit fires there.
        app.aboutToQuit.connect(adapter.stop)  # type: ignore[attr-defined]

    return app.exec()
