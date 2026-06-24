"""InteractiveAnalysisWidget — the measure-gui host for an INTERACTIVE analysis.

A self-contained Qt widget (a matplotlib canvas + a generic controls column) that
implements the ``InteractiveHost`` port: the adapter's ``InteractiveSession`` draws
on ``figure``, repaints via ``redraw``, and offloads a heavy step via
``run_background``. The widget renders one button per ``session.actions()`` entry
(it never learns what an action does) plus a Done button, forwards canvas mouse
events to the session, and shows ``session.info_text()`` verbatim.

All interaction is on the Qt main thread (Case B of ADR-0017); only the compute
passed to ``run_background`` runs off-main, delegated to the app's
``BackgroundRunner`` (pool strategy) through the injected ``InteractiveHostEnv``
port (ADR-0019), which marshals the result back to the main thread.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.main.adapter import InteractiveSession


class InteractiveHostEnv(Protocol):
    """The narrow capability the host widget needs from the app to run a heavy
    interactive step off the main thread (ADR-0019). The Controller satisfies it
    (delegating to ``BackgroundRunner``'s pool); tests inject a fake. The widget
    is a passive host that issues no commands — this one capability is all it
    pulls from the app, so it is injected as this port rather than the whole
    Controller."""

    def run_background(
        self, compute: Callable[[], object], on_done: Callable[[object], None]
    ) -> None: ...


class InteractiveAnalysisWidget(QWidget):
    """Qt host for an interactive analysis session (implements InteractiveHost)."""

    def __init__(self, env: InteractiveHostEnv, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._env = env
        self._figure = Figure(figsize=(8, 5))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._session: InteractiveSession | None = None
        self._on_done: Callable[[], None] | None = None

        controls = QWidget()
        self._controls_layout = QVBoxLayout(controls)
        controls.setFixedWidth(200)

        root = QHBoxLayout(self)
        root.addWidget(self._canvas, stretch=1)
        root.addWidget(controls)

        self._info = QLabel("")
        self._info.setWordWrap(True)
        self._action_buttons: list[QPushButton] = []
        self._done_btn = QPushButton("Done")
        self._done_btn.clicked.connect(self._handle_done)

        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_move)
        self._canvas.mpl_connect("button_release_event", self._on_release)

    # --- InteractiveHost port -------------------------------------------

    @property
    def figure(self) -> Figure:
        return self._figure

    def redraw(self) -> None:
        if self._session is not None:
            self._info.setText(self._session.info_text())
        self._canvas.draw_idle()

    def run_background(
        self, compute: Callable[[], object], on_done: Callable[[object], None]
    ) -> None:
        # Delegate to the app's BackgroundRunner (pool strategy) via the injected
        # env port; the widget no longer owns a thread pool or the marshal.
        self._env.run_background(compute, on_done)

    # --- binding + lifecycle --------------------------------------------

    def bind(self, session: InteractiveSession, on_done: Callable[[], None]) -> None:
        """Attach the adapter's session: render its action buttons, show its info,
        and remember the Done callback. Call once, right after the session is
        created with this widget as its host."""
        self._session = session
        self._on_done = on_done
        # One generic button per action — we render the label and dispatch the id;
        # we never learn what the action does.
        for action_id, label in session.actions():
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked=False, a=action_id: self._invoke(a))
            self._controls_layout.addWidget(btn)
            self._action_buttons.append(btn)
        self._info.setText(session.info_text())
        self._controls_layout.addWidget(self._info)
        self._controls_layout.addStretch(1)
        self._controls_layout.addWidget(self._done_btn)
        self.redraw()

    def _invoke(self, action_id: str) -> None:
        if self._session is not None:
            self._session.invoke_action(action_id)

    def _handle_done(self) -> None:
        if self._on_done is None:
            return
        # Disable further input — the result is being committed.
        self._done_btn.setEnabled(False)
        for btn in self._action_buttons:
            btn.setEnabled(False)
        self._on_done()

    # --- canvas events --> session --------------------------------------

    def _on_press(self, event) -> None:  # noqa: ANN001 - mpl MouseEvent
        if self._session is not None and event.inaxes is not None:
            self._session.on_press(event.xdata)

    def _on_move(self, event) -> None:  # noqa: ANN001 - mpl MouseEvent
        if self._session is not None and event.inaxes is not None:
            self._session.on_move(event.xdata)

    def _on_release(self, event) -> None:  # noqa: ANN001 - mpl MouseEvent
        if self._session is not None and event.inaxes is not None:
            self._session.on_release(event.xdata, event.ydata)
