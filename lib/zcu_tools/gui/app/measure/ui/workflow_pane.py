"""Driving workflow pane: render owner drafts and forward commands to the facet."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.experiment.workflows import Actor
from zcu_tools.gui.app.measure.services.workflow_control import (
    WorkflowTabSnapshot,
)
from zcu_tools.gui.app.measure.services.workflow_tab import FormKind
from zcu_tools.gui.form import FormEnv, FormErrors, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

from .workflow_figures import WorkflowFigures

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.controller import Controller

logger = logging.getLogger(__name__)


class WorkflowPane(QWidget):
    """Show Plan, Tunables and Ask for one facet-owned tab.

    The facet owns capture, lowering, tunables binding and run/lease lifetime.
    controller supplies the shared RUN gate and owner scheduler.
    Call refresh on owner-loop polls and dispose after accepted close, before
    destroying Qt children. Rejected closes must leave the pane intact.
    """

    def __init__(
        self,
        controller: Controller,
        tab_id: str,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self._control = controller.workflow_control()
        self._tab_id = tab_id
        self._writable = True
        self._disposed = False
        self._editors: dict[FormKind, FormEditor[FormModel]] = {}
        self._states: dict[FormKind, FormState[FormModel]] = {}
        self._envs: dict[FormKind, FormEnv] = {}
        self._editing: dict[FormKind, bool] = {}
        self._pages: dict[FormKind, QVBoxLayout] = {}
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        left = QWidget(splitter)
        column = QVBoxLayout(left)
        self._subtabs = QTabWidget(left)
        self._add_form_page("plan", "Plan")
        self._add_form_page("tunables", "Tunables")
        self._add_ask_page()
        column.addWidget(self._subtabs, 1)
        self._message = QLabel(left)
        self._message.setObjectName("workflow.message")
        self._message.setWordWrap(True)
        make_label_selectable(self._message)
        column.addWidget(self._message)
        self._status = QLabel("idle", left)
        self._status.setObjectName("workflow.status")
        make_label_selectable(self._status)
        column.addWidget(self._status)
        actions = QHBoxLayout()
        self._buttons: dict[str, QPushButton] = {}
        for caption in ("Refresh", "Start", "Pause", "Resume", "Stop"):
            button = QPushButton(caption, left)
            button.clicked.connect(
                lambda _checked=False, command=caption: self._command(command)
            )
            self._buttons[caption] = button
            actions.addWidget(button)
        column.addLayout(actions)
        self._figures = WorkflowFigures(
            controller.owner_scheduler, lambda: self._control.plots(tab_id), splitter
        )
        splitter.addWidget(left)
        splitter.addWidget(self._figures)
        splitter.setSizes([420, 780])
        layout.addWidget(splitter)
        self._control.set_plot_host(tab_id, self._figures)
        self._unsubscribe = self._control.subscribe(self.refresh)
        self.refresh()

    def _add_ask_page(self) -> None:
        page = QWidget(self._subtabs)
        layout = QVBoxLayout(page)
        label = QLabel("No question pending.", page)
        label.setObjectName("workflow.ask")
        label.setWordWrap(True)
        make_label_selectable(label)
        layout.addWidget(label)
        layout.addStretch()
        self._subtabs.addTab(page, "Ask")

    def _add_form_page(self, kind: FormKind, caption: str) -> None:
        page = QWidget(self._subtabs)
        column = QVBoxLayout(page)
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        body = QWidget(scroll)
        self._pages[kind] = QVBoxLayout(body)
        scroll.setWidget(body)
        column.addWidget(scroll)
        if kind == "tunables":
            button = QPushButton("Apply", page)
            button.clicked.connect(lambda: self._command("Apply"))
            self._apply = button
            column.addWidget(button)
        self._subtabs.addTab(page, caption)

    def _bind_form(self, kind: FormKind, *, editable: bool) -> None:
        state = self._control.form_state(self._tab_id, kind)
        env = self._control.form_env(self._tab_id, kind)
        editor = self._editors.get(kind)
        if editor is None or self._states.get(kind) is not state:
            if editor is not None:
                editor.detach()
                self._pages[kind].removeWidget(editor)
                editor.hide()
                editor.deleteLater()
            editor = FormEditor(state, env, self)
            self._pages[kind].addWidget(editor)
            self._editors[kind] = editor
            self._states[kind] = state
            self._envs[kind] = env
            self._editing.pop(kind, None)
        elif self._envs[kind] is not env:
            editor.refresh(env)
            self._envs[kind] = env
        if self._editing.get(kind) != editable:
            editor.set_editing_enabled(enabled=editable)
            self._editing[kind] = editable

    def set_writable(self, *, writable: bool) -> None:
        """Set local command availability; the next refresh applies it to controls."""
        self._writable = writable

    def refresh(self) -> None:
        """Read native status and current draft identity; never execute engine work."""
        if self._disposed:
            return
        snapshot = next(
            (s for s in self._control.tabs() if s.tab_id == self._tab_id), None
        )
        if snapshot is None:
            return
        self._project(snapshot)

    def _project(self, snapshot: WorkflowTabSnapshot) -> None:
        status = snapshot.status
        lifecycle = "pending" if snapshot.busy else "idle"
        if status is not None:
            lifecycle = status.lifecycle
        self._status.setText(lifecycle)
        writable = self._writable
        lease_active = any(
            p.kind == "run" for p in self._ctrl.get_hardware_gate_presence()
        )
        idle = writable and not lease_active and not snapshot.busy
        accepted = status is not None and snapshot.run_id is not None
        self._bind_form("plan", editable=idle)
        self._bind_form("tunables", editable=writable and (idle or snapshot.busy))
        self._buttons["Refresh"].setEnabled(idle)
        self._buttons["Start"].setEnabled(idle)
        self._buttons["Pause"].setEnabled(
            writable and accepted and lifecycle == "running"
        )
        self._buttons["Resume"].setEnabled(
            writable and accepted and lifecycle == "paused"
        )
        self._buttons["Stop"].setEnabled(writable and snapshot.busy)
        self._apply.setEnabled(
            writable and accepted and lifecycle in {"running", "paused"}
        )
        if snapshot.error is not None:
            self._message.setText(snapshot.error)
        # Native Ask is not available yet. In particular, no UI-thread engine
        # continuation or fake question/answer semantics belong in this adapter.
        if snapshot.run_id is not None:
            self._control.pending_question(snapshot.run_id)

    def _command(self, command: str) -> None:
        try:
            self._submit(command)
        except Exception as error:
            logger.exception("Workflow UI command failed: %s", command)
            self._message.setText(str(error))
        self.refresh()

    def _submit(self, command: str) -> None:
        snapshot = next(s for s in self._control.tabs() if s.tab_id == self._tab_id)
        run_id = snapshot.run_id
        if command == "Refresh":
            self._control.prepare_forms(self._tab_id)
        elif command == "Start":
            result = self._control.start(
                self._tab_id,
                uuid4().hex,
                plan_errors=self._editors["plan"].local_parse_errors(),
                tunables_errors=self._editors["tunables"].local_parse_errors(),
            )
            self._message.setText(
                "; ".join(result.messages.values())
                if isinstance(result, FormErrors)
                else ""
            )
        elif run_id is None:
            raise RuntimeError("Workflow has not started")
        elif command == "Pause":
            self._control.pause(run_id)
        elif command == "Resume":
            self._control.resume(run_id)
        elif command == "Stop":
            self._control.stop(run_id)
        elif command == "Apply":
            result = self._control.apply_tunables(
                run_id,
                actor=Actor("user", "gui"),
                local_errors=self._editors["tunables"].local_parse_errors(),
            )
            self._message.setText(
                "; ".join(result.messages.values())
                if isinstance(result, FormErrors)
                else f"Applied revision {result.revision}"
            )
        else:
            raise ValueError(f"Unknown workflow command: {command}")

    def dispose(self) -> None:
        """Detach owner/editor subscriptions and redraw timers before Qt deletion.

        The caller must first close the facet tab or drain its producer. This
        does not stop work, close hardware or drain the shared session.
        """
        if self._disposed:
            return
        self._disposed = True
        self._unsubscribe()
        for editor in self._editors.values():
            editor.detach()
        self._figures.dispose()
