"""NodeDetailPane — the right side: the selected Node's edit / run sub-tabs.

Shows ONE Node at a time (whichever the left list selects). Inner QTabWidget:
- "編輯" (Edit): the ParamForm (settings + dep summary). Locked read-only during a
  run so the user can still see "what this run used".
- "執行" (Run): the Node's liveplot. Prototype: a placeholder label (real
  matplotlib canvas + worker marshal is Phase C). Shows a "started / waiting"
  status the auto-follow updates.

``show_node`` rebuilds for a new selection; ``set_running`` flips edit→run lock
and switches to the run tab; ``focus_run`` is called by auto-follow when the Node
starts executing.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.autofluxdep.nodes.spec import NodeInstance

from .param_form import ParamForm

_EDIT_TAB = 0
_RUN_TAB = 1


class NodeDetailPane(QWidget):
    """Right pane: edit/run sub-tabs for the currently selected Node."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._node: Optional[NodeInstance] = None
        self._form: Optional[ParamForm] = None
        self._running = False

        root = QVBoxLayout(self)
        self._title = QLabel("(no node selected)")
        root.addWidget(self._title)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        # Edit tab host (form swapped in per node)
        self._edit_host = QWidget()
        self._edit_layout = QVBoxLayout(self._edit_host)
        self._tabs.addTab(self._edit_host, "編輯")

        # Run tab: liveplot placeholder
        self._run_host = QWidget()
        run_layout = QVBoxLayout(self._run_host)
        self._plot_label = QLabel("(liveplot appears here during a run)")
        run_layout.addWidget(self._plot_label)
        self._tabs.addTab(self._run_host, "執行")

    # --- selection ---

    def show_node(self, node: Optional[NodeInstance]) -> None:
        self._node = node
        # clear old form
        if self._form is not None:
            self._form.setParent(None)
            self._form = None
        if node is None:
            self._title.setText("(no node selected)")
            self._plot_label.setText("(no node selected)")
            return
        self._title.setText(node.name)
        self._form = ParamForm(node)
        self._form.set_read_only(self._running)
        self._edit_layout.addWidget(self._form)
        self._plot_label.setText(f"{node.name} — (liveplot appears here during a run)")

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

    def focus_run(self, node_name: str, flux_idx: int) -> None:
        """Auto-follow: a Node started executing → show its run tab + status."""
        self._tabs.setCurrentIndex(_RUN_TAB)
        self._plot_label.setText(f"{node_name} — running (flux point {flux_idx})")

    # --- testing accessors ---

    @property
    def current_form(self) -> Optional[ParamForm]:
        return self._form

    @property
    def current_tab(self) -> int:
        return self._tabs.currentIndex()
