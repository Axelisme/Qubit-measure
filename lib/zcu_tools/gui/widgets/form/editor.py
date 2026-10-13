"""Qt rendering and local text drafts for the caller-owned form state."""

from collections.abc import Callable, Mapping
from weakref import ref

from pydantic import JsonValue
from PyQt6 import sip
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
    WriteRejected,
)

from ._controls import ROW_SPACING, FieldRow, GroupBlock, LocalText
from ._schema_view import FieldView, flatten_fields, project_fields


class FormEditor[F: FormModel](QWidget):
    """Edit a caller-owned FormState on the Qt GUI thread, without submitting it.

    The editor borrows state and the host-captured env. Parsed input writes the
    state immediately, or goes to the host's write callable (see __init__);
    unparseable text and host-rejected input remain local and block editor
    validity. Evaluation, choices, branch defaults and model validation belong
    to state.
    No Engine update, revision check, source capture, persistence or I/O occurs.

    Public child object names use model-field paths joined by dots:
    form.input.<path> is a QLineEdit for text, numeric, complex, expression or
    whole-container input; a QCheckBox for bool; or a QComboBox for choices and
    discriminators. Combo item data is the ordinary input value.
    form.mode.<path> is a numeric/complex/string QComboBox with data "direct" and "expr".
    On a choice leaf (Literal, Enum or choices) it is hidden in Direct mode, so
    the row shows one dropdown: the choices, then a last "Expression…" item
    without data that switches the mode to "expr". Returning to Direct selects
    the resolved value when it is a current choice, else the last Direct one.
    Discriminators offer only ordinary typed tags and have no expression mode.
    Empty choices keep the current branch visible without changing its input.
    A one-value Literal has form.input.<path> as read-only QLineEdit text of
    its value, without form.mode or a dropdown. A float with Display decimals
    shows its Direct and resolved values rounded to that many digits; text
    typed with more digits stays as typed, and the state keeps every digit.
    Fields the schema marks hidden (Display(hidden=True), or a one-value
    structural tag outside a union) have no controls. Their input is kept,
    and their errors appear at the nearest drawn ancestor row or form.error.
    An untagged str-or-model field (a library key or an inline model) has
    form.mode.<path> with data "key" and "inline" instead: Key mode shows
    form.input.<path> as a QLineEdit, or an editable QComboBox listing the
    captured library keys whose Spec matches; Inline mode shows the model's
    child rows, a tagged union with its selector. Switching writes the new
    mode's default ("" or {}); the other mode's value is never converted.
    form.enabled.<path> is a nullable-model activation QCheckBox.
    form.group.<path>.<group> (form.group.<group> at the root level) is the
    block of one Display group of the model level at <path>. It holds the
    level's fields of that group, in declaration order, under a header with
    the group name in the level's label column and an Expand/Collapse
    QPushButton. Each level shows its ungrouped rows first, then its group
    blocks in the order of each group's first field. A group whose fields
    are all hidden has no block. While any of its rows (or their children)
    shows an error, the header shows a "Contains errors" note; a folded
    group stays folded.
    Groups start folded; model rows start as Display(collapsed=...) says.
    Folds the user sets on groups and model rows survive rebuilds of this
    editor (branch, Key/Inline or activation changes), remembered by path.
    form.value.<path> is a QLabel with the resolved value of an Expression-mode
    input; Direct input is not echoed, and the label is hidden while empty.
    form.error.<path> is a QLabel under its field with current field errors;
    form.error displays model-wide errors. Both are hidden while empty.
    Mapping-key/index errors appear at their container field.
    Only current active fields have input controls. Sequences and typed mappings
    use JSON text for the whole leaf; they do not expose key/index write paths.
    Complex direct text uses Python complex-number notation, not an expression.

    Build finishes before subscribing. detach or close removes the subscription,
    disables this view and drops local text without changing caller state.
    After QObject destruction or garbage collection of this view, the next
    state publication removes the subscription without touching state.
    refresh and error queries after detach raise RuntimeError.
    validity_changed emits a bool only when combined validity changes; it is not
    an Apply request. GUI-thread use and a live QApplication are required.
    """

    validity_changed = pyqtSignal(bool)

    def __init__(
        self,
        state: FormState[F],
        env: FormEnv,
        parent: QWidget | None = None,
        *,
        write: Callable[[Mapping[FormWritePath, JsonValue]], None] | None = None,
    ) -> None:
        """Build controls for state using env, then subscribe to state changes.

        State/source validation errors are displayed, not replaced by defaults.
        Structural/schema or source failures propagate; failed construction
        leaves no state subscription. parent controls the ordinary Qt lifetime.

        write decides where parsed input goes. When it is None (the default),
        each parsed batch is installed with state.write(batch, env) using the
        current env. When it is a callable, it receives every parsed batch, a
        one-entry mapping from the edited field's write path to its JSON
        input, and the editor never writes state itself. The callable installs
        accepted input into this same state (for example after its own checks
        or substitutions); the editor still reads and subscribes to state, so
        input the callable does not install is not shown.
        To refuse a batch, the callable raises WriteRejected(message) before
        writing state. That field then keeps the user's input as local view
        input with message as its error: text and key inputs keep the typed
        text and mode; a checkbox, dropdown, Enabled box or Key/Inline switch
        shows the state's value again. Like unparseable text, it appears in
        local_parse_errors and errors, blocks validity, survives refresh and
        publications, and clears on that field's next accepted write, on
        discard_local_text, or when the field leaves the form. Any other
        exception from the callable propagates.
        """
        app = QApplication.instance()
        if not isinstance(app, QApplication) or app.thread() != QThread.currentThread():
            raise RuntimeError("FormEditor requires the QApplication GUI thread")
        super().__init__(parent)
        self._state: FormState[F] | None = state
        self._env: FormEnv | None = env
        self._host_write = write
        self._unsubscribe: Callable[[], None] | None = None
        self._editing_enabled = True
        self._local: dict[FormWritePath, LocalText] = {}
        self._rows: dict[FormWritePath, FieldRow] = {}
        self._groups: list[GroupBlock] = []
        # User folds by model path, and by (model level path, group name).
        self._model_expanded: dict[FormWritePath, bool] = {}
        self._group_expanded: dict[tuple[FormWritePath, str], bool] = {}
        self._state_errors: FormErrors | None = None
        self._valid: bool | None = None
        # Hosts own outer margins; rows keep ROW_SPACING apart.
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(ROW_SPACING)
        self._root_error = QLabel(self)
        self._root_error.setObjectName("form.error")
        self._root_error.setWordWrap(True)
        self._root_error.hide()
        self._layout.addWidget(self._root_error)
        self._content = QWidget(self)
        self._layout.addWidget(self._content)
        self._publish()
        # The state callback must not keep the QWidget alive. No Python slot is
        # connected to destroyed: cyclic GC may clear such a slot before Qt
        # calls it while deleting the widget, which crashes the process. A
        # publication that finds this view collected or deleted unsubscribes.
        weak_view = ref(self)

        def published() -> None:
            view = weak_view()
            if view is None or sip.isdeleted(view):
                # Bound below, before state can publish to this callback.
                unsubscribe()
            elif view._state is not None:
                view._publish()

        unsubscribe = state.subscribe(published)
        self._unsubscribe = unsubscribe

    def refresh(self, env: FormEnv) -> None:
        """Re-evaluate with env and refresh controls, choices, values and errors.

        Keep still-active local unparseable or rejected input. Never reset
        state, run domain defaults, poll sources or echo a write. External
        state publications refresh parsed controls through the subscription.
        Inactive branch text is discarded. Raises RuntimeError after detach.
        """
        self._require_state()
        self._env = env
        self._publish()

    def discard_local_text(self) -> None:
        """Discard local unparseable or rejected input and render the state.

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
        """Return detached local input errors, or None when state holds all input.

        Local input is unparseable text or input the host's write rejected;
        its message is the parse error or the WriteRejected message. Paths
        identify active controls. This query does not parse, write, or imply
        that the model is valid. Raises RuntimeError after detach.
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
        # Text does not survive its path becoming a model section, nor does a
        # rejected section input (which has no text) its path becoming a leaf.
        self._local = {
            field.path: local
            for field in flat
            if (local := self._local.get(field.path)) is not None
            and local.section == (field.kind == "model")
        }
        shape = tuple(
            (
                field.path,
                field.kind,
                field.options is not None,
                field.expression,
                field.fixed,
                field.group,
            )
            for field in flat
        )
        old_shape = tuple(
            (
                row.field.path,
                row.field.kind,
                row.field.options is not None,
                row.field.expression,
                row.field.fixed,
                row.field.group,
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(ROW_SPACING)
        self._rows = {}
        self._groups = []
        self._add_level((), fields, self._content, layout)
        # One stretch below the last row takes spare height; a stretch inside a
        # group would make the group grow and spread its rows apart.
        layout.addStretch()
        self._layout.replaceWidget(old, self._content)
        # Retire names immediately; deleteLater waits until the active Qt signal
        # finishes, so rebuilding a branch cannot delete its emitting selector.
        old.setParent(None)
        old.deleteLater()

    def _add_level(
        self,
        path: FormWritePath,
        fields: tuple[FieldView, ...],
        parent: QWidget,
        layout: QVBoxLayout,
    ) -> None:
        """Add the rows of the model level at path to layout under parent.

        Ungrouped fields come first, then one GroupBlock per Display group in
        the order of each group's first field, holding that group's fields.
        """
        groups: dict[str, list[FieldView]] = {}
        for field in fields:
            if field.group is not None:
                groups.setdefault(field.group, []).append(field)
        labels = [
            self._add_row(field, parent, layout)
            for field in fields
            if field.group is None
        ]
        for name, members in groups.items():
            block = GroupBlock(
                path,
                name,
                tuple(member.path for member in members),
                parent,
                self._group_expanded,
            )
            self._groups.append(block)
            layout.addWidget(block.widget)
            labels.append(block.label)
            # Rows in a group block are indented, so they get their own column.
            _align_labels(
                [self._add_row(member, block.body, block.rows) for member in members]
            )
        _align_labels(labels)

    def _add_row(
        self, field: FieldView, parent: QWidget, layout: QVBoxLayout
    ) -> QLabel:
        """Add field's row, and a model's child level, and return its label."""
        row = FieldRow(
            field,
            parent,
            self._write,
            self._unparsed,
            self._model_expanded,
        )
        self._rows[field.path] = row
        layout.addWidget(row.widget)
        if row.group is not None:
            child_layout = row.group.layout()
            if isinstance(child_layout, QVBoxLayout):
                # Recursion back into _add_level: a model's children are a level.
                self._add_level(field.path, field.children, row.group, child_layout)
        return row.label

    def _write(self, path: FormWritePath, value: JsonValue) -> None:
        state = self._require_state()
        self._local.pop(path, None)
        batch = {path: value}
        if self._host_write is None:
            state.write(batch, self._require_env())
        else:
            try:
                self._host_write(batch)
            except WriteRejected as exc:
                # The host installed nothing: keep this input local, like
                # unparseable text; the publish below redraws state around it.
                self._local[path] = self._rows[path].rejected_input(exc.message)
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
        self._root_error.setVisible(bool(self._root_error.text()))
        for path, row in self._rows.items():
            row.set_error("\n".join(by_row.get(path, ())))
            if path in self._local:
                row.clear_value()
        for block in self._groups:
            block.set_has_errors(
                has_errors=any(
                    path[: len(member)] == member
                    for path in by_row
                    for member in block.members
                )
            )
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


def _align_labels(labels: list[QLabel]) -> None:
    """Give labels one fixed width, the widest label's, as one label column.

    Every control beside these labels then starts at the same left edge.
    """
    width = max((label.sizeHint().width() for label in labels), default=0)
    for label in labels:
        label.setFixedWidth(width)
