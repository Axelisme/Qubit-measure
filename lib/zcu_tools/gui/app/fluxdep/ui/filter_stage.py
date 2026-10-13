"""FilterStage — the Fit page's Filter stage: published masks or the live draft.

Without a live Filter context the stage draws the published point cloud and
per-spectrum masks read-only (``Controller.render_selection``), for the holder
and every viewer alike. With a live context, such as an agent's, it hosts a
SelectorWidget on that context. Edit begins a context, Discard cancels it, and
the Selector's Finish publishes and closes it; all three follow control
authority. Leaving the stage only detaches the Selector view: the context
stays until Finish or Discard (or an upstream change retires it).

Public Qt seams (objectName): ``fluxdep.filter.edit``, ``fluxdep.filter.discard``
and ``fluxdep.filter.published`` (the read-only canvas).
"""

from __future__ import annotations

import logging

from matplotlib.backend_bases import Event
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import (
    FitChangedPayload,
    InteractiveChangedPayload,
    SelectionChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.gui.app.fluxdep.interactive import CrossSelectionContext
from zcu_tools.gui.app.fluxdep.ui.interactive.selector import SelectorWidget
from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.expected_error import FailedPreconditionError

logger = logging.getLogger(__name__)

_NO_CLOUD = "Select points on a spectrum first; Edit opens the Filter then."


class FilterStage(QWidget):
    """The Filter stage of the Fit page, built hidden and detached.

    attach() shows the current content and keeps it current while shown;
    detach() hides it without cancelling a live context. Construct and use on
    the State-owner Qt thread; dispose() releases the event subscriptions.
    """

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._writable = True
        # True between attach() and detach(): the stage's content is on screen.
        self._shown = False
        self._selector: SelectorWidget | None = None
        self._selector_context_id: int | None = None
        self._subs = EventSubscriptions()
        self._build_ui()
        self._subs.subscribe(
            ctrl.bus, InteractiveChangedPayload, self._on_interactive_changed
        )
        for published in (
            SpectrumAddedPayload,
            SpectrumRemovedPayload,
            SpectrumChangedPayload,
            SelectionChangedPayload,
            FitChangedPayload,
        ):
            self._subs.subscribe(ctrl.bus, published, self._on_published_changed)
        self._refresh_buttons()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._edit = QPushButton("Edit")
        self._edit.setObjectName("fluxdep.filter.edit")
        self._edit.setToolTip("Open a Filter draft from the published masks")
        self._edit.clicked.connect(self._on_edit)
        self._discard = QPushButton("Discard")
        self._discard.setObjectName("fluxdep.filter.discard")
        self._discard.setToolTip("Drop the Filter draft; the published masks stay")
        self._discard.clicked.connect(self._on_discard)
        self._status = QLabel("")
        self._status.setAccessibleName("Filter status")
        self._status.setWordWrap(True)
        self._status.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        row = QHBoxLayout()
        row.addWidget(self._edit)
        row.addWidget(self._discard)
        row.addWidget(self._status, stretch=1)
        root.addLayout(row)

        self._views = QStackedWidget()
        self._published_figure = Figure(figsize=(6, 4))
        self._published = FigureCanvasQTAgg(self._published_figure)
        self._published.setObjectName("fluxdep.filter.published")
        self._published.mpl_connect("resize_event", self._fit_published_layout)
        self._draft_holder = QWidget()
        self._draft_layout = QVBoxLayout(self._draft_holder)
        self._draft_layout.setContentsMargins(0, 0, 0, 0)
        self._views.addWidget(self._published)
        self._views.addWidget(self._draft_holder)
        root.addWidget(self._views, stretch=1)

    # --- public ------------------------------------------------------------

    def attach(self) -> None:
        """Show the live Filter context's Selector, else the published masks.

        Never begins a context. A Selector already attached to the same live
        context is kept, with its controls and preview. Raises RuntimeError if
        the live context changes while its view is being attached.
        """
        self._shown = True
        self._refresh()

    def detach(self) -> None:
        """Hide the stage and retire its Selector view, keeping the context open."""
        self._shown = False
        self._retire_selector()

    def set_writable(self, *, writable: bool) -> None:
        """Gate Edit, Discard and the Selector controls; viewing stays available."""
        self._writable = writable
        if self._selector is not None:
            self._selector.set_editable(editable=writable)
        self._refresh_buttons()

    def quiesce(self) -> None:
        """Drain the attached Selector's preview worker, if any."""
        if self._selector is not None:
            self._selector.quiesce()

    def dispose(self) -> None:
        """Release event subscriptions; call after quiesce() when closing."""
        self._subs.unsubscribe_all()

    # --- content -------------------------------------------------------------

    def _refresh(self) -> None:
        live = self._ctrl.interactive.current_cross_selection()
        if live is None:
            self._retire_selector()
            self._show_published()
        else:
            active = self._ctrl.interactive.inspect_selection()
            if active is None or active.context is not live:
                raise RuntimeError("selection context changed while attaching its view")
            if active.context_id != self._selector_context_id:
                self._retire_selector()
                self._mount_selector(live, active.context_id)
            self._views.setCurrentWidget(self._draft_holder)
        self._refresh_buttons()

    def _mount_selector(self, live: CrossSelectionContext, context_id: int) -> None:
        selector = SelectorWidget(
            live, on_apply=self._ctrl.interactive.apply_cross_selection
        )
        selector.set_editable(editable=self._writable)
        selector.finished.connect(self._on_finish)
        self._draft_layout.addWidget(selector)
        self._selector = selector
        self._selector_context_id = context_id
        self._status.setText("")

    def _retire_selector(self) -> None:
        if self._selector is not None:
            self._selector.teardown()
            self._draft_layout.removeWidget(self._selector)
            self._selector.deleteLater()
            self._selector = None
        self._selector_context_id = None

    def _show_published(self) -> None:
        self._views.setCurrentWidget(self._published)
        try:
            self._ctrl.render_selection(self._published_figure)
        except Exception as exc:
            # A drawing failure leaves the publication intact; report it here.
            logger.exception("render_selection failed")
            self._status.setText(f"Could not draw the published masks: {exc}")
        else:
            self._status.setText("" if self._has_cloud() else _NO_CLOUD)
        self._published.draw_idle()

    def _fit_published_layout(self, _event: Event) -> None:
        self._published_figure.tight_layout()

    def _has_cloud(self) -> bool:
        return any(e.point_count > 0 for e in self._ctrl.state.spectrums.values())

    def _refresh_buttons(self) -> None:
        live = self._ctrl.interactive.current_cross_selection() is not None
        self._edit.setEnabled(self._writable and not live and self._has_cloud())
        self._discard.setEnabled(self._writable and live)

    # --- actions and facts -----------------------------------------------------

    def _on_edit(self) -> None:
        # Holder-only: the button is disabled for viewers and without a cloud.
        try:
            self._ctrl.interactive.begin_cross_selection()
        except FailedPreconditionError as exc:
            # The published view stays; say why no draft opened.
            self._status.setText(f"Could not open the Filter: {exc}")
            return
        self.attach()

    def _on_discard(self) -> None:
        self._ctrl.interactive.cancel_selection()

    def _on_finish(self) -> None:
        try:
            self._ctrl.interactive.finish_cross_selection()
        except Exception as exc:
            # The draft stays open and editable; report why Finish failed.
            logger.exception("Filter finish failed")
            self._status.setText(f"Finish failed: {exc}")

    def _on_interactive_changed(self, payload: InteractiveChangedPayload) -> None:
        if payload.kind != "selection" or payload.phase == "updated":
            return
        if (
            payload.phase == "closed"
            and payload.context_id == self._selector_context_id
        ):
            self._retire_selector()
        if self._shown:
            self._refresh()
        else:
            self._refresh_buttons()

    def _on_published_changed(self, payload: object) -> None:
        del payload
        # A live draft's Selector follows its own Session; only the published
        # view reads these publications.
        if self._shown and self._selector is None:
            self._show_published()
        self._refresh_buttons()
