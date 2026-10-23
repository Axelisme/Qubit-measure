"""Driving workflow pane: render owner drafts and forward commands to the facet.

Visible run controls (availability still follows writable/lease/accepted gates):

============================  ==============
Run state                     Controls
============================  ==============
Idle or terminal              Refresh, Start
Running                       Pause, Stop
Paused without a question     Resume, Stop
Waiting for an Ask answer     Stop
Pending, pausing or stopping  Stop
============================  ==============

Answer stays in Ask. Ask appears for a pending question or the latest retained
exchange, selects new questions, and keeps accepted/cancelled exchanges visible.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.experiment.workflows import Actor
from zcu_tools.gui.app.measure.services.workflow_borrower import WorkflowBorrower
from zcu_tools.gui.app.measure.services.workflow_control import (
    WorkflowTabSnapshot,
)
from zcu_tools.gui.app.measure.services.workflow_run import QuestionExchange
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
        self._retiring = False
        self._borrower = WorkflowBorrower()
        self._editor_borrowers: dict[FormKind, WorkflowBorrower] = {}
        self._editors: dict[FormKind, FormEditor[FormModel]] = {}
        self._states: dict[FormKind, FormState[FormModel]] = {}
        self._envs: dict[FormKind, FormEnv] = {}
        self._editing: dict[FormKind, bool] = {}
        self._ask_pending = False
        self._question_request_id: str | None = None
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
        self._add_feedback(left, column)
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
        self._borrower.add_cleanup(self._figures.dispose)
        self._control.set_plot_host(tab_id, self._figures)
        self._borrower.add_cleanup(self._control.subscribe(self.refresh))
        self._borrower.add_cleanup(self._dispose_editors)
        self.refresh()

    def _add_feedback(self, parent: QWidget, column: QVBoxLayout) -> None:
        self._owner_error: str | None = None
        self._message = QLabel(parent)
        self._message.setObjectName("workflow.message")
        self._message.setWordWrap(True)
        make_label_selectable(self._message)
        column.addWidget(self._message)
        self._status = QLabel("idle", parent)
        self._status.setObjectName("workflow.status")
        make_label_selectable(self._status)
        column.addWidget(self._status)
        self._scope_labels: dict[str, QLabel] = {}
        for name, tooltip in (
            ("active_path", "Active chain through the first fork"),
            ("live_paths", "Live scope leaves, one full path per line"),
        ):
            label = QLabel(parent)
            label.setObjectName(f"workflow.{name}")
            label.setToolTip(tooltip)
            label.setWordWrap(True)
            make_label_selectable(label)
            column.addWidget(label)
            self._scope_labels[name] = label

    def _add_ask_page(self) -> None:
        page = QWidget(self._subtabs)
        layout = QVBoxLayout(page)
        self._ask_page = page
        self._question_label = QLabel("No question pending.", page)
        self._question_label.setObjectName("workflow.ask")
        self._question_label.setWordWrap(True)
        make_label_selectable(self._question_label)
        layout.addWidget(self._question_label)
        self._answer_schema = QLabel(page)
        self._answer_schema.setObjectName("workflow.answer_schema")
        self._answer_schema.setWordWrap(True)
        make_label_selectable(self._answer_schema)
        layout.addWidget(self._answer_schema)
        layout.addWidget(QLabel("Answer JSON", page))
        self._answer_input = QLineEdit(page)
        self._answer_input.setObjectName("workflow.answer")
        self._answer_input.setPlaceholderText(
            "Enter an object matching the answer schema"
        )
        layout.addWidget(self._answer_input)
        self._answer_button = QPushButton("Answer", page)
        self._answer_button.clicked.connect(lambda: self._command("Answer"))
        layout.addWidget(self._answer_button)
        self._answer_status = QLabel(page)
        self._answer_status.setObjectName("workflow.answer_status")
        self._answer_status.setWordWrap(True)
        make_label_selectable(self._answer_status)
        layout.addWidget(self._answer_status)
        layout.addStretch()
        page.hide()

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
            self._pause_hint = QLabel(
                "Resume 會重做未提交的點；可先 Apply 新的 tunables", page
            )
            self._pause_hint.setObjectName("workflow.pause_hint")
            self._pause_hint.setWordWrap(True)
            make_label_selectable(self._pause_hint)
            column.addWidget(self._pause_hint)
            button = QPushButton("Apply", page)
            button.clicked.connect(lambda: self._command("Apply"))
            self._apply = button
            column.addWidget(button)
            self._discard = QPushButton("Discard", page)
            self._discard.clicked.connect(lambda: self._command("Discard"))
            column.addWidget(self._discard)
        self._subtabs.addTab(page, caption)

    def _bind_form(self, kind: FormKind, *, editable: bool) -> None:
        state = self._control.form_state(self._tab_id, kind)
        env = self._control.form_env(self._tab_id, kind)
        editor = self._editors.get(kind)
        if editor is None or self._states.get(kind) is not state:
            if editor is not None:
                self._editor_borrowers[kind].dispose()
                self._pages[kind].removeWidget(editor)
                editor.hide()
                editor.deleteLater()
            editor = FormEditor(state, env, self)
            lifetime = WorkflowBorrower()
            lifetime.add_cleanup(editor.detach)
            self._editor_borrowers[kind] = lifetime
            self._pages[kind].addWidget(editor)
            self._editors[kind] = editor
            self._states[kind] = state
            self._envs[kind] = env
            self._editing.pop(kind, None)
            if kind == "tunables":
                lifetime.add_cleanup(state.subscribe(self._refresh_discard))
                editor.errors_changed.connect(self._refresh_discard)
        elif self._envs[kind] is not env:
            editor.refresh(env)
            self._envs[kind] = env
        if self._editing.get(kind) != editable:
            editor.set_editing_enabled(enabled=editable)
            self._editing[kind] = editable

    def _refresh_discard(self) -> None:
        state = self._states["tunables"]
        self._discard.setEnabled(
            self._editing.get("tunables", False)
            and (
                state.is_dirty
                or self._editors["tunables"].local_parse_errors() is not None
            )
        )

    def set_writable(self, *, writable: bool) -> None:
        """Set local command availability; the next refresh applies it to controls."""
        self._writable = writable

    def refresh(self) -> None:
        """Read native status and current draft identity; never execute engine work."""
        if self._retiring:
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
        if lifecycle == "paused" and self._status.text() != "paused":
            self._subtabs.setCurrentIndex(1)
        self._pause_hint.setVisible(lifecycle == "paused")
        self._status.setText(lifecycle)
        self._scope_labels["active_path"].setText(
            "" if status is None else status.active_path
        )
        self._scope_labels["live_paths"].setText(
            "" if status is None else "\n".join(status.live_paths)
        )
        self._scope_labels["active_path"].setVisible(status is not None)
        self._scope_labels["live_paths"].setVisible(
            status is not None and status.live_paths != [status.active_path]
        )
        writable = self._writable
        lease_active = any(
            p.kind == "run" for p in self._ctrl.get_hardware_gate_presence()
        )
        idle = writable and not lease_active and not snapshot.busy
        self._bind_form("plan", editable=idle)
        self._bind_form(
            "tunables",
            editable=writable
            and (idle or lifecycle in {"running", "pausing", "paused"}),
        )
        self._project_controls(snapshot, lifecycle, idle=idle)
        self._refresh_discard()
        if snapshot.error is not None:
            self._message.setText(snapshot.error)
        elif (
            self._owner_error is not None and self._message.text() == self._owner_error
        ):
            self._message.clear()
        self._owner_error = snapshot.error
        self._project_question(snapshot)

    def _project_controls(
        self, snapshot: WorkflowTabSnapshot, lifecycle: str, *, idle: bool
    ) -> None:
        writable = self._writable
        accepted = snapshot.status is not None and snapshot.run_id is not None
        for caption in ("Refresh", "Start"):
            self._buttons[caption].setVisible(not snapshot.busy)
        self._buttons["Pause"].setVisible(snapshot.busy and lifecycle == "running")
        self._buttons["Resume"].setVisible(
            snapshot.busy
            and lifecycle == "paused"
            and snapshot.pending_question is None
        )
        self._buttons["Stop"].setVisible(snapshot.busy)
        self._buttons["Refresh"].setEnabled(idle)
        self._buttons["Start"].setEnabled(idle)
        self._buttons["Pause"].setEnabled(
            writable and accepted and lifecycle == "running"
        )
        self._buttons["Resume"].setEnabled(
            writable
            and accepted
            and lifecycle == "paused"
            and snapshot.pending_question is None
        )
        self._buttons["Stop"].setEnabled(snapshot.busy)
        self._apply.setEnabled(
            writable and accepted and lifecycle in {"running", "pausing", "paused"}
        )

    def _project_question(self, snapshot: WorkflowTabSnapshot) -> None:
        question = snapshot.pending_question
        pending = question is not None
        exchange = snapshot.last_exchange
        ask_index = self._subtabs.indexOf(self._ask_page)
        if pending or exchange is not None:
            if ask_index < 0:
                self._subtabs.addTab(self._ask_page, "Ask")
        elif ask_index >= 0:
            self._subtabs.removeTab(ask_index)
            self._ask_page.hide()
        if question is not None and self._question_request_id != question["request_id"]:
            self._question_request_id = question["request_id"]
            self._answer_input.setText("{}")
            self._subtabs.setCurrentWidget(self._ask_page)
        if exchange is not None:
            self._show_exchange(exchange)
        self._ask_pending = pending
        enabled = self._writable and pending
        self._answer_input.setEnabled(enabled)
        self._answer_button.setEnabled(enabled)

    def _show_exchange(self, exchange: QuestionExchange) -> None:
        question = exchange.question
        self._question_label.setText(
            f"{question['question_type']}\n"
            + json.dumps(question["question"], ensure_ascii=False, indent=2)
        )
        self._answer_schema.setText(
            "Answer schema: "
            + json.dumps(question["answer_schema"], ensure_ascii=False)
        )
        if exchange.answer is not None:
            self._answer_input.setText(json.dumps(exchange.answer, ensure_ascii=False))
        self._answer_status.setText(
            {
                "waiting": "等待回答",
                "submitted": "回答已提交",
                "answered": "已回答",
                "cancelled": "提問已取消",
            }[exchange.state]
        )

    def focus_question(self) -> None:
        """Show the pending Ask after badge navigation, without submitting anything."""
        if self._ask_pending:
            self._subtabs.setCurrentWidget(self._ask_page)

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
        elif command == "Discard":
            self._control.discard_tunables(self._tab_id)
            self._editors["tunables"].discard_local_text()
            self._message.clear()
        elif run_id is None:
            raise RuntimeError("Workflow has not started")
        elif command == "Pause":
            self._control.pause(run_id)
        elif command == "Resume":
            self._control.resume(run_id)
        elif command == "Stop":
            if self._ctrl.control_identity.status().holder != "gui":
                self._ctrl.control_identity.reclaim_local()
            self._control.stop(run_id)
        elif command == "Answer":
            answer = TypeAdapter(JsonValue).validate_json(self._answer_input.text())
            if self._question_request_id is None:
                raise RuntimeError("No rendered question to answer")
            self._control.answer(run_id, answer, request_id=self._question_request_id)
            self._message.setText("")
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

    def _dispose_editors(self) -> None:
        for lifetime in self._editor_borrowers.values():
            lifetime.dispose()

    def dispose(self) -> None:
        """Detach owner/editor subscriptions and redraw timers before Qt deletion.

        The caller must first close the facet tab or drain its producer. This
        does not stop work, close hardware or drain the shared session.
        Failure propagates and retains pending cleanup for a later dispose call;
        the caller must keep Qt children alive until disposal succeeds.
        """
        self._retiring = True
        self._borrower.dispose()
