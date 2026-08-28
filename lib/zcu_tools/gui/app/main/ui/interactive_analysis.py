"""InteractiveAnalysisWidget — the measure-gui host for an INTERACTIVE analysis.

A self-contained Qt widget (a matplotlib canvas + a generic controls column) that
implements the ``InteractiveHost`` port: the adapter's ``InteractiveSession`` draws
on ``figure``, repaints via ``redraw``, and offloads a heavy step via
``run_background``. The widget renders one widget per ``session.controls()``
entry (``ButtonControl`` → ``QPushButton``, ``ToggleControl`` → ``QCheckBox``)
plus a Done button, forwards canvas mouse events to the session, and shows
``session.info_text()`` verbatim. It owns only validation / lowering — it never
compares domain keys.

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
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.main.adapter.types import (
    ButtonControl,
    InteractiveSession,
    ToggleControl,
)


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
        self._bound = False
        self._done = False

        controls = QWidget()
        self._controls_layout = QVBoxLayout(controls)
        controls.setFixedWidth(200)

        root = QHBoxLayout(self)
        root.addWidget(self._canvas, stretch=1)
        root.addWidget(controls)

        self._info = QLabel("")
        self._info.setWordWrap(True)
        self._action_buttons: list[QPushButton] = []
        self._checkboxes: list[QCheckBox] = []
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
        """Attach the adapter's session: render its control widgets, show its info,
        and remember the Done callback. Call once, right after the session is
        created with this widget as its host."""
        if self._bound or self._session is not None:
            raise RuntimeError(
                "InteractiveAnalysisWidget.bind() may only be called once"
            )
        if not callable(on_done):
            raise TypeError("on_done must be callable")

        controls = session.controls()
        if not isinstance(controls, tuple):
            raise TypeError("InteractiveSession.controls() must return a tuple")
        # --- validation (no partial mount on failure) ------------------
        seen_keys: set[str] = set()
        for ctrl in controls:
            if isinstance(ctrl, ButtonControl):
                # ButtonControl.__post_init__ already validates key/label/callable,
                # but re-validate for the host seam to ensure Fast Fail even if
                # the declaration bypassed the dataclass constructor.
                if not isinstance(ctrl.key, str) or not ctrl.key.strip():
                    raise ValueError("ButtonControl key must be a non-empty string")
                if not isinstance(ctrl.label, str) or not ctrl.label.strip():
                    raise ValueError("ButtonControl label must be a non-empty string")
                if not callable(ctrl.on_trigger):
                    raise TypeError("ButtonControl on_trigger must be callable")
                if ctrl.key in seen_keys:
                    raise ValueError(f"duplicate control key {ctrl.key!r}")
                seen_keys.add(ctrl.key)
            elif isinstance(ctrl, ToggleControl):
                if not isinstance(ctrl.key, str) or not ctrl.key.strip():
                    raise ValueError("ToggleControl key must be a non-empty string")
                if not isinstance(ctrl.label, str) or not ctrl.label.strip():
                    raise ValueError("ToggleControl label must be a non-empty string")
                if type(ctrl.initial) is not bool:  # noqa: E721
                    raise TypeError("ToggleControl initial must be bool")
                if not callable(ctrl.on_change):
                    raise TypeError("ToggleControl on_change must be callable")
                if ctrl.key in seen_keys:
                    raise ValueError(f"duplicate control key {ctrl.key!r}")
                seen_keys.add(ctrl.key)
            else:
                raise TypeError(
                    f"unsupported InteractiveControl variant {type(ctrl).__name__}"
                )

        # --- lowering (exhaustive, typed callbacks, no domain key compare) ---
        self._session = session
        self._on_done = on_done
        self._bound = True
        try:
            for ctrl in controls:
                if isinstance(ctrl, ButtonControl):

                    def _make_trigger(cb: Callable[[], None]) -> Callable[[bool], None]:
                        def _on_clicked(_checked: bool = False) -> None:
                            if self._done:
                                return
                            cb()

                        return _on_clicked

                    btn = QPushButton(ctrl.label)
                    btn.clicked.connect(_make_trigger(ctrl.on_trigger))
                    self._controls_layout.addWidget(btn)
                    self._action_buttons.append(btn)
                elif isinstance(ctrl, ToggleControl):

                    def _make_toggle(
                        cb: Callable[[bool], None],
                    ) -> Callable[[bool], None]:
                        def _on_toggled(checked: bool) -> None:
                            if self._done:
                                return
                            cb(bool(checked))

                        return _on_toggled

                    cb_widget = QCheckBox(ctrl.label)
                    # Apply initial before connecting so construction never fires the callback.
                    cb_widget.setChecked(bool(ctrl.initial))
                    cb_widget.toggled.connect(_make_toggle(ctrl.on_change))
                    self._controls_layout.addWidget(cb_widget)
                    self._checkboxes.append(cb_widget)
                else:  # pragma: no cover — validated above, exhaustive guard
                    raise TypeError(
                        f"unsupported InteractiveControl variant {type(ctrl).__name__}"
                    )
        except Exception:
            # Ensure no partial mount remains visible on failure after widgets were added.
            for btn in self._action_buttons:
                self._controls_layout.removeWidget(btn)
                btn.deleteLater()
            for chk in self._checkboxes:
                self._controls_layout.removeWidget(chk)
                chk.deleteLater()
            self._action_buttons.clear()
            self._checkboxes.clear()
            self._session = None
            self._on_done = None
            self._bound = False
            raise
        self._info.setText(session.info_text())
        self._controls_layout.addWidget(self._info)
        self._controls_layout.addStretch(1)
        self._controls_layout.addWidget(self._done_btn)
        self.redraw()

    def _handle_done(self) -> None:
        if self._on_done is None or self._done:
            return
        # Close the input gate before submitting — disable all controls, Done,
        # and canvas forwarding in one step; submission then runs exactly once.
        self._done = True
        self._done_btn.setEnabled(False)
        for btn in self._action_buttons:
            btn.setEnabled(False)
        for chk in self._checkboxes:
            chk.setEnabled(False)
        self._on_done()

    # --- canvas events --> session (gated after Done) -------------------

    def _on_press(self, event) -> None:  # noqa: ANN001 - mpl MouseEvent
        if self._done:
            return
        if self._session is not None and event.inaxes is not None:
            self._session.on_press(event.xdata)

    def _on_move(self, event) -> None:  # noqa: ANN001 - mpl MouseEvent
        if self._done:
            return
        if self._session is not None and event.inaxes is not None:
            self._session.on_move(event.xdata)

    def _on_release(self, event) -> None:  # noqa: ANN001 - mpl MouseEvent
        if self._done:
            return
        if self._session is not None and event.inaxes is not None:
            self._session.on_release(event.xdata, event.ydata)
