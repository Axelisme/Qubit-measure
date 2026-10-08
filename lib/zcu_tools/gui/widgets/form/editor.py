"""Qt rendering and local text drafts for the caller-owned form state."""

from collections.abc import Callable
from weakref import ref

from pydantic import JsonValue
from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import QThread
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormModel,
    FormPath,
    FormState,
    FormWritePath,
)

from ._controls import FieldRow, LocalText
from ._schema_view import FieldView, flatten_fields, project_fields


class FormEditor[F: FormModel](QWidget):
    """Edit a caller-owned FormState on the Qt GUI thread, without submitting it.

    The editor borrows state and the host-captured env. Parsed input writes the
    state immediately; unparseable text remains local and blocks editor validity.
    Evaluation, choices, branch defaults and model validation belong to state.
    No Engine update, revision check, source capture, persistence or I/O occurs.

    Public child object names use model-field paths joined by dots:
    form.input.<path> is a QLineEdit for text, numeric, complex, expression or
    whole-container input; a QCheckBox for bool; or a QComboBox for choices and
    discriminators. Combo item data is the ordinary input value.
    form.mode.<path> is a numeric/complex QComboBox with data "direct" and "expr".
    Discriminators offer only ordinary typed tags and have no expression mode.
    Empty choices keep the current branch visible without changing its input.
    form.enabled.<path> is a nullable-model activation QCheckBox.
    form.value.<path> is a QLabel with the current evaluated value.
    form.error.<path> is a QLabel with current field errors; form.error displays
    model-wide errors. Mapping-key/index errors appear at their container field.
    Only current active fields have input controls. Sequences and typed mappings
    use JSON text for the whole leaf; they do not expose key/index write paths.
    Complex direct text uses Python complex-number notation, not an expression.

    Build finishes before subscribing. detach or close removes the subscription,
    disables this view and drops local text without changing caller state.
    QObject destruction also removes the subscription without touching state.
    refresh and error queries after detach raise RuntimeError.
    validity_changed emits a bool only when combined validity changes; it is not
    an Apply request. GUI-thread use and a live QApplication are required.
    """

    validity_changed = pyqtSignal(bool)

    def __init__(
        self, state: FormState[F], env: FormEnv, parent: QWidget | None = None
    ) -> None:
        """Build controls for state using env, then subscribe to state changes.

        State/source validation errors are displayed, not replaced by defaults.
        Structural/schema or source failures propagate; failed construction
        leaves no state subscription. parent controls the ordinary Qt lifetime.
        """
        app = QApplication.instance()
        if not isinstance(app, QApplication) or app.thread() != QThread.currentThread():
            raise RuntimeError("FormEditor requires the QApplication GUI thread")
        super().__init__(parent)
        self._state: FormState[F] | None = state
        self._env: FormEnv | None = env
        self._unsubscribe: Callable[[], None] | None = None
        self._editing_enabled = True
        self._local: dict[FormWritePath, LocalText] = {}
        self._rows: dict[FormWritePath, FieldRow] = {}
        self._state_errors: FormErrors | None = None
        self._valid: bool | None = None
        self._layout = QVBoxLayout(self)
        self._root_error = QLabel(self)
        self._root_error.setObjectName("form.error")
        self._root_error.setWordWrap(True)
        self._layout.addWidget(self._root_error)
        self._content = QWidget(self)
        self._layout.addWidget(self._content)
        self._publish()
        # The state callback must not keep the QWidget alive. QObject destruction
        # invokes the standalone unsubscribe even when closeEvent was skipped.
        weak_view = ref(self)

        def published() -> None:
            view = weak_view()
            if view is not None and view._state is not None:
                view._publish()

        unsubscribe = state.subscribe(published)
        self._unsubscribe = unsubscribe
        self.destroyed.connect(lambda _object: unsubscribe())

    def refresh(self, env: FormEnv) -> None:
        """Re-evaluate with env and refresh controls, choices, values and errors.

        Keep still-active local unparseable text. Never reset state, run domain
        defaults, poll sources or echo a write. External state publications
        refresh parsed controls through the subscription. Inactive branch text
        is discarded. Raises RuntimeError after detach.
        """
        self._require_state()
        self._env = env
        self._publish()

    def discard_local_text(self) -> None:
        """Discard unparseable view text and render the current state.

        Call on the GUI thread before detach. This does not reset/write state,
        evaluate defaults, submit, or change the fixed env. Parsed draft changes
        remain unless the host resets state separately. Detached views raise
        RuntimeError.
        """
        self._require_state()
        self._local.clear()
        self._publish()

    def set_editing_enabled(self, *, enabled: bool) -> None:
        """Enable or disable input controls without changing state or local text.

        External state changes remain visible while inputs are disabled.
        Raises RuntimeError after detach.
        """
        self._require_state()
        self._editing_enabled = enabled
        self._publish()

    def local_parse_errors(self) -> FormErrors | None:
        """Return detached local text errors, or None when every input parsed.

        Paths identify active controls. This query does not parse, write, or
        imply that the model is valid. Raises RuntimeError after detach.
        """
        self._require_state()
        messages: dict[FormPath, str] = {
            path: local.error for path, local in self._local.items()
        }
        return FormErrors(messages) if messages else None

    def errors(self) -> FormErrors | None:
        """Return merged local parse and current state errors, or None if valid.

        Local errors at a path take precedence over state errors at that path.
        Querying performs no state write. Raises RuntimeError after detach.
        """
        self._require_state()
        messages = dict(self._state_errors.messages) if self._state_errors else {}
        messages.update({path: local.error for path, local in self._local.items()})
        return FormErrors(messages) if messages else None

    def detach(self) -> None:
        """Idempotently unsubscribe, disable the view and discard local text.

        Do not reset, close or submit caller-owned state. Later publications do
        not update this view. The QWidget may still be destroyed normally.
        """
        if self._state is None:
            return
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._state = None
        self._env = None
        self._local.clear()
        self.setEnabled(False)

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Detach and accept Qt's close event a0, preserving caller state."""
        self.detach()
        if a0 is not None:
            a0.accept()

    def _require_state(self) -> FormState[F]:
        if self._state is None:
            raise RuntimeError("FormEditor is detached")
        return self._state

    def _require_env(self) -> FormEnv:
        if self._env is None:
            raise RuntimeError("FormEditor is detached")
        return self._env

    def _publish(self) -> None:
        state = self._require_state()
        env = self._require_env()
        schema = state.input_schema(env)
        preview = state.preview(env)
        fields = project_fields(schema, state.input.data, preview)
        flat = flatten_fields(fields)
        self._state_errors = preview.errors
        self._local = {
            field.path: local
            for field in flat
            if (local := self._local.get(field.path)) is not None
            and field.kind != "model"
        }
        shape = tuple(
            (field.path, field.kind, field.options is not None, field.expression)
            for field in flat
        )
        old_shape = tuple(
            (
                row.field.path,
                row.field.kind,
                row.field.options is not None,
                row.field.expression,
            )
            for row in self._rows.values()
        )
        if shape != old_shape:
            self._rebuild(fields)
        for field in flat:
            self._rows[field.path].display(
                field, self._local.get(field.path), editable=self._editing_enabled
            )
        self._show_errors()

    def _rebuild(self, fields: tuple[FieldView, ...]) -> None:
        old = self._content
        self._content = QWidget(self)
        layout = QVBoxLayout(self._content)
        self._rows = {}
        self._add_rows(fields, self._content, layout)
        self._layout.replaceWidget(old, self._content)
        # Retire names immediately; deleteLater waits until the active Qt signal
        # finishes, so rebuilding a branch cannot delete its emitting selector.
        old.setParent(None)
        old.deleteLater()

    def _add_rows(
        self, fields: tuple[FieldView, ...], parent: QWidget, layout: QVBoxLayout
    ) -> None:
        for field in fields:
            row = FieldRow(field, parent, self._write, self._unparsed)
            self._rows[field.path] = row
            layout.addWidget(row.widget)
            if row.group is not None:
                child_layout = row.group.layout()
                if isinstance(child_layout, QVBoxLayout):
                    self._add_rows(field.children, row.group, child_layout)
        layout.addStretch()

    def _write(self, path: FormWritePath, value: JsonValue) -> None:
        state = self._require_state()
        self._local.pop(path, None)
        state.write({path: value}, self._require_env())
        # Unchanged writes do not publish but can clear a local parse error.
        self._publish()

    def _unparsed(self, path: FormWritePath, local: LocalText) -> None:
        self._local[path] = local
        self._show_errors()

    def _show_errors(self) -> None:
        errors = self.errors()
        messages = dict(errors.messages) if errors else {}
        self._root_error.setText(messages.pop((), ""))
        by_row: dict[FormWritePath, list[str]] = {}
        for path, message in messages.items():
            target = self._error_target(path)
            if target is not None:
                by_row.setdefault(target, []).append(message)
            else:
                root = self._root_error.text()
                self._root_error.setText("\n".join(filter(None, (root, message))))
        for path, row in self._rows.items():
            row.error.setText("\n".join(by_row.get(path, ())))
            if path in self._local:
                row.value.clear()
        valid = errors is None
        if valid != self._valid:
            self._valid = valid
            self.validity_changed.emit(valid)

    def _error_target(self, path: FormPath) -> FormWritePath | None:
        for length in range(len(path), 0, -1):
            prefix = path[:length]
            for candidate in self._rows:
                if candidate == prefix:
                    return candidate
        return None
