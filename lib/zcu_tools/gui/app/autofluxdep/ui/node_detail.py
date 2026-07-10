"""NodeDetailPane — the right side: the selected Node's edit / run sub-tabs.

Shows ONE Node at a time (whichever the left list selects). Inner QTabWidget:
- "編輯" (Edit): the typed ``NodeCfgForm``. Locked
  read-only during a run so the user can still see "what this run used".
- "執行" (Run): the Node's liveplot — a bare matplotlib ``FigureCanvasQTAgg``
  embedded directly (NOT via gui/plotting's backend; the worker never draws, so
  the plain Qt canvas suffices — see ADR-0017). The MainWindow owns the per-Node
  Figure + Plotter (built at Run start, lifetime = sweep) and hands this pane the
  canvas to show for the selected Node via ``show_run_canvas``.

``show_node`` updates the selected node and can defer edit form materialization
for run auto-follow navigation; during a run, materialized edit forms are cached
per Node so manual selection does not rebuild the same hidden form repeatedly.
``set_running`` flips the edit→run lock and can switch tabs when auto-follow is
enabled;
``show_run_canvas`` swaps in the selected Node's live canvas (or a placeholder
when there is none).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode

from .node_cfg_form import NodeCfgForm

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.controller import Controller

_EDIT_TAB = 0
_RUN_TAB = 1


class NodeDetailPane(QWidget):
    """Right pane: edit/run sub-tabs for the currently selected Node."""

    user_tab_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._node: PlacedNode | None = None
        self._form: NodeCfgForm | None = None
        self._form_node: PlacedNode | None = None
        self._form_index: int | None = None
        self._pending_form: tuple[Controller, PlacedNode, int] | None = None
        self._form_cache: dict[tuple[int, int], NodeCfgForm] = {}
        self._running = False
        self._canvas: QWidget | None = None
        self._programmatic_tab_switch = False
        # Where a de-selected canvas is parked instead of being left parentless.
        # A parentless QWidget becomes a top-level window the moment it draws —
        # and every Node's Plotter redraws each run point, even off-screen ones —
        # so a de-selected canvas would flash as a stray window. Parking it under
        # a hidden widget keeps it parented (never a window) while it keeps
        # drawing into its Result. Set by the MainWindow that owns the canvases.
        self._canvas_park: QWidget | None = None

        root = QVBoxLayout(self)
        self._title = QLabel("(no node selected)")
        root.addWidget(self._title)

        self._tabs = QTabWidget()
        self._tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tabs, 1)

        # Edit tab host (form swapped in per node)
        self._edit_host = QWidget()
        self._edit_layout = QVBoxLayout(self._edit_host)
        self._tabs.addTab(self._edit_host, "編輯")

        # Run tab: a host whose single child is either the live canvas or a
        # placeholder label (swapped by show_run_canvas).
        self._run_host = QWidget()
        self._run_layout = QVBoxLayout(self._run_host)
        self._run_placeholder = QLabel("(liveplot appears here during a run)")
        self._run_layout.addWidget(self._run_placeholder)
        self._tabs.addTab(self._run_host, "執行")

    # --- selection ---

    def show_node(
        self,
        controller: Controller | None,
        node: PlacedNode | None,
        index: int,
        *,
        build_edit_form: bool = True,
    ) -> None:
        """Show ``node`` at workflow ``index`` and optionally build its edit form.

        ``controller`` is the typed-form's commit target + LiveModel env (None only
        when clearing the selection). ``index`` is the placement's list position
        that receives the form's complete cfg value tree on edit.

        Run auto-follow uses ``build_edit_form=False`` so navigation can keep the
        run canvas/title current without rebuilding a hidden edit form on every
        active-node transition. Opening the edit tab materializes the pending form.
        """
        self._node = node
        if node is None or controller is None:
            self._pending_form = None
            self._clear_form()
            self._title.setText("(no node selected)")
            return
        self._title.setText(node.name)
        self._pending_form = (controller, node, index)
        if build_edit_form or self.current_tab == _EDIT_TAB:
            self._ensure_edit_form_current()

    def _clear_form(self) -> None:
        """Destroy the current edit form and every hidden cached form."""
        current = self._form
        for form in set(self._form_cache.values()):
            if form is current:
                continue
            self._destroy_form(form)
        self._form_cache.clear()
        self._destroy_current_form()

    def _destroy_current_form(self) -> None:
        """Destroy only the currently displayed edit form."""
        if self._form is not None:
            self._destroy_form(self._form)
            self._form = None
        self._form_node = None
        self._form_index = None

    def _destroy_form(self, form: NodeCfgForm) -> None:
        self._edit_layout.removeWidget(form)
        form.teardown()
        form.setParent(None)
        form.deleteLater()

    def _cache_key(self, node: PlacedNode, index: int) -> tuple[int, int]:
        return (id(node), index)

    def _cache_current_form(self) -> None:
        if self._form is None or self._form_node is None or self._form_index is None:
            return
        self._form_cache[self._cache_key(self._form_node, self._form_index)] = (
            self._form
        )

    def _detach_current_form(self) -> None:
        """Hide the current form without tearing it down."""
        if self._form is None:
            return
        self._edit_layout.removeWidget(self._form)
        self._form.hide()
        self._form.setParent(self._edit_host)
        self._form = None
        self._form_node = None
        self._form_index = None

    def _show_form(self, form: NodeCfgForm, node: PlacedNode, index: int) -> None:
        if form.parentWidget() is not self._edit_host:
            form.setParent(self._edit_host)
        self._form = form
        self._form_node = node
        self._form_index = index
        form.set_read_only(self._running)
        self._edit_layout.addWidget(form)
        form.show()

    def _ensure_edit_form_current(self) -> None:
        """Materialize the pending edit form when the edit tab needs it."""
        pending = self._pending_form
        if pending is None:
            return

        controller, node, index = pending
        if (
            self._form is not None
            and self._form_node is node
            and self._form_index == index
        ):
            self._form.set_read_only(self._running)
            self._pending_form = None
            return

        if self._running:
            self._cache_current_form()
            self._detach_current_form()
            key = self._cache_key(node, index)
            form = self._form_cache.get(key)
            if form is None:
                form = NodeCfgForm(controller, node, index)
                self._form_cache[key] = form
            self._show_form(form, node, index)
        else:
            self._destroy_current_form()
            form = NodeCfgForm(controller, node, index)
            self._show_form(form, node, index)
        self._pending_form = None

    def _drop_hidden_form_cache(self) -> None:
        current = self._form
        for key, form in list(self._form_cache.items()):
            if form is current:
                continue
            self._destroy_form(form)
            del self._form_cache[key]
        self._form_cache.clear()

    def set_canvas_park(self, park: QWidget) -> None:
        """Inject the hidden widget de-selected canvases are parked under.

        The MainWindow owns the canvases and this park; this pane only moves a
        canvas between the run tab and the park (never leaves one parentless).
        """
        self._canvas_park = park

    def refresh_cfg_editor(self, event: object) -> None:
        """Refresh every materialized cfg form against a session-context event."""
        seen: set[int] = set()
        for form in (self._form, *self._form_cache.values()):
            if form is None:
                continue
            marker = id(form)
            if marker in seen:
                continue
            seen.add(marker)
            form.refresh_external(event)

    def show_run_canvas(self, canvas: QWidget | None) -> None:
        """Swap the run tab's content to ``canvas`` (or the placeholder if None).

        The MainWindow owns the canvases; this pane only displays the selected
        Node's. A de-selected canvas is re-parented to the hidden park (NOT left
        parentless — a parentless canvas flashes as a stray window when its
        Plotter redraws), so it keeps drawing into its Result and re-selecting
        shows the latest state.
        """
        if self._canvas is not None:
            self._canvas.setParent(self._canvas_park)  # park, don't detach
            self._canvas = None
        if canvas is None:
            self._run_placeholder.show()
            return
        self._run_placeholder.hide()
        self._canvas = canvas
        self._run_layout.addWidget(canvas)
        canvas.show()

    # --- run state ---

    def set_running(self, running: bool, *, switch_tab: bool = True) -> None:
        if running:
            self._running = True
            self._cache_current_form()
            for form in set(self._form_cache.values()):
                form.set_read_only(True)
            if self._form is not None:
                self._form.set_read_only(True)
        else:
            if switch_tab:
                self._ensure_edit_form_current()
            self._running = False
            if self._form is not None:
                self._form.set_read_only(False)
            self._drop_hidden_form_cache()
        self._tabs.setTabText(_EDIT_TAB, "編輯·唯讀" if running else "編輯")
        if not switch_tab:
            return
        if running:
            self._set_current_tab(_RUN_TAB)
        else:
            self._set_current_tab(_EDIT_TAB)

    def focus_run(self) -> None:
        """Auto-follow: switch to the run tab (the running Node is selected)."""
        self._set_current_tab(_RUN_TAB)

    def _set_current_tab(self, index: int) -> None:
        self._programmatic_tab_switch = True
        try:
            self._tabs.setCurrentIndex(index)
        finally:
            self._programmatic_tab_switch = False

    def _on_tab_changed(self, index: int) -> None:
        if index == _EDIT_TAB:
            self._ensure_edit_form_current()
        if not self._programmatic_tab_switch:
            self.user_tab_changed.emit(index)

    def teardown(self) -> None:
        """Tear down every edit form owned by this pane."""
        self._pending_form = None
        self._clear_form()

    # --- testing accessors ---

    @property
    def current_form(self) -> NodeCfgForm | None:
        return self._form

    @property
    def current_tab(self) -> int:
        return self._tabs.currentIndex()
