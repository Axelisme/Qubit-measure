"""NodeDetailPane — the right side: the selected Node's edit / run sub-tabs.

Shows ONE Node at a time (whichever the left list selects). Inner QTabWidget:
- "編輯" (Edit): the ParamForm (settings + dep summary). Locked read-only during a
  run so the user can still see "what this run used".
- "執行" (Run): the Node's liveplot — a bare matplotlib ``FigureCanvasQTAgg``
  embedded directly (NOT via gui/plotting's backend; the worker never draws, so
  the plain Qt canvas suffices — see ADR-0018). The MainWindow owns the per-Node
  Figure + Plotter (built at Run start, lifetime = sweep) and hands this pane the
  canvas to show for the selected Node via ``show_run_canvas``.

``show_node`` rebuilds the edit form for a new selection; ``set_running`` flips
the edit→run lock and switches to the run tab; ``show_run_canvas`` swaps in the
selected Node's live canvas (or a placeholder when there is none).
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode

from .param_form import ParamForm

_EDIT_TAB = 0
_RUN_TAB = 1


class NodeDetailPane(QWidget):
    """Right pane: edit/run sub-tabs for the currently selected Node."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._node: Optional[PlacedNode] = None
        self._form: Optional[ParamForm] = None
        self._running = False
        self._canvas: Optional[QWidget] = None
        # Where a de-selected canvas is parked instead of being left parentless.
        # A parentless QWidget becomes a top-level window the moment it draws —
        # and every Node's Plotter redraws each run point, even off-screen ones —
        # so a de-selected canvas would flash as a stray window. Parking it under
        # a hidden widget keeps it parented (never a window) while it keeps
        # drawing into its Result. Set by the MainWindow that owns the canvases.
        self._canvas_park: Optional[QWidget] = None

        root = QVBoxLayout(self)
        self._title = QLabel("(no node selected)")
        root.addWidget(self._title)

        self._tabs = QTabWidget()
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

    def show_node(self, node: Optional[PlacedNode]) -> None:
        self._node = node
        # clear old form
        if self._form is not None:
            self._form.setParent(None)
            self._form = None
        if node is None:
            self._title.setText("(no node selected)")
            return
        self._title.setText(node.name)
        self._form = ParamForm(node)
        self._form.set_read_only(self._running)
        self._edit_layout.addWidget(self._form)

    def set_canvas_park(self, park: QWidget) -> None:
        """Inject the hidden widget de-selected canvases are parked under.

        The MainWindow owns the canvases and this park; this pane only moves a
        canvas between the run tab and the park (never leaves one parentless).
        """
        self._canvas_park = park

    def show_run_canvas(self, canvas: Optional[QWidget]) -> None:
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

    def set_running(self, running: bool) -> None:
        self._running = running
        if self._form is not None:
            self._form.set_read_only(running)
        self._tabs.setTabText(_EDIT_TAB, "編輯·唯讀" if running else "編輯")
        if running:
            self._tabs.setCurrentIndex(_RUN_TAB)
        else:
            self._tabs.setCurrentIndex(_EDIT_TAB)

    def focus_run(self) -> None:
        """Auto-follow: switch to the run tab (the running Node is selected)."""
        self._tabs.setCurrentIndex(_RUN_TAB)

    # --- testing accessors ---

    @property
    def current_form(self) -> Optional[ParamForm]:
        return self._form

    @property
    def current_tab(self) -> int:
        return self._tabs.currentIndex()
