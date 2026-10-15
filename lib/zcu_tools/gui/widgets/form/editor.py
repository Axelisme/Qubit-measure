"""Qt rendering and local text drafts for the caller-owned form state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from weakref import ref

from pydantic import JsonValue
from PyQt6 import sip
from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import QSignalBlocker, Qt, QThread
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormFieldDescription,
    FormModel,
    FormPath,
    FormPreview,
    FormState,
    FormWritePath,
    WriteRejected,
)
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

from ._controls import FieldRow, GroupBlock, LocalText
from ._fold import ROW_SPACING
from ._view import FieldView, project_field


@dataclass
class _Level:
    """Qt ownership for one instantiated model or display-group body."""

    path: FormWritePath
    widget: QWidget
    layout: QVBoxLayout
    rows: dict[FormWritePath, FieldRow] = field(default_factory=dict)
    children: dict[FormWritePath, _Level] = field(default_factory=dict)
    groups: dict[str, tuple[GroupBlock, _Level]] = field(default_factory=dict)


def _shape(field: FieldView) -> tuple[object, ...]:
    """Control construction inputs, excluding current input and selection."""
    return (
        field.kind,
        field.options is not None,
        field.expression,
        field.fixed,
        field.label,
        field.unit,
        field.description,
        field.placeholder,
        field.nullable,
        field.can_expand,
    )


class FormEditor[F: FormModel](QWidget):
    """Edit a caller-owned FormState on the Qt GUI thread, without submitting it.

    The editor borrows state and the host-captured env. A text field commits
    its text when the user presses Enter or the field loses focus, not on each
    keystroke; text that already shows the state's input writes nothing, and
    read-only rows commit nothing. Checkboxes, dropdowns, Enabled boxes and the
    Expression action write at once. Committed parsed input writes the state,
    or goes to the host's write callable (see __init__); committed unparseable
    text and host-rejected input remain local and block editor validity.
    Uncommitted text changes neither state nor validity, and an external state
    publication redraws it with the state's input. Evaluation, choices, branch
    defaults and model validation belong to state.
    No Engine update, revision check, source capture, persistence or I/O occurs.

    Public child object names use model-field paths joined by dots:
    form.input.<path> is a QLineEdit for text, numeric, complex, expression or
    whole-container input; a QCheckBox for bool; or a non-editable QComboBox
    for a field whose core description lists options.
    form.mode.<path> is the checkable QAction "Expression" of a field whose
    description permits expression input: the core offers it on text fields
    (int, float, complex and str values typed into a QLineEdit). It is
    checked in Expression mode. form.input.<path> lists it in actions() and
    in its right-click menu, after the text field's standard items; the row
    tooltip says so. Triggering it switches the mode and writes the input
    again through the same write route as a commit. In Expression mode it
    writes {"expr": <current text>}. In Direct mode it writes the value the
    state's public preview computed for the shown text, when that text is
    the state's committed, accepted Expression input. A computed value the
    model rejects is still written, and validation shows its errors.
    Uncommitted or host-rejected text never takes an older preview value.
    Without a usable computed value, Direct mode parses the shown text as a
    commit does: a numeric field keeps unparseable text as local input with
    its parse error, and a str field writes the text as a literal.
    An Expression-mode input has a light tint of the palette's
    highlight colour. Dropdown and checkbox fields have no form.mode.<path>.
    A one-value Literal has form.input.<path> as read-only QLineEdit text of
    its value, without form.mode or a dropdown. A float with Display decimals
    shows its Direct and resolved values rounded to that many digits; text
    typed with more digits stays as typed, and the state keeps every digit.
    Fields the description marks hidden (Display(hidden=True), or a one-value
    structural tag outside a union) have no controls. Their input is kept,
    and their errors appear at the nearest drawn ancestor row or form.error.
    Every field whose description lists options (Literal, Enum, choices,
    tagged unions, and the module and waveform selections a Form generator
    declares) shows one dropdown, form.input.<path>, and nothing else
    decides its items: each option's label in declaration order, with the
    option's value as item data. The first option matching the input by the
    core's Option rule is selected. A present input no option matches is
    kept: a last item shows it, a nonempty string as itself and other JSON
    as its JSON text, plus " (missing)", with the input as data; it is
    selected, the state is not changed and the field's errors still show.
    An unfilled field selects nothing. Choosing an option writes its value
    at the field as one batch; the core starts a new branch from its own
    defaults. While the selected option has fields, their rows show below
    the dropdown (tags its value fixes have no rows); such a row has no
    Enabled box, since a nullable field lists None as an option. The
    dropdown's right-click menu lists the selected option's commands as
    QActions form.command.<path>.<name> with the declared label as text, also
    in form.input.<path>.actions(); triggering one runs that command (see
    __init__). An option without commands opens no menu.
    The editor has no fold buttons: group blocks and model sections fold by
    a left-button click on their fold title, a bold QLabel whose text is
    "\u25b8 " (folded) or "\u25be " (expanded) followed by the label.
    Tab stops on every fold title, while a click does not focus it. A title
    with keyboard focus shows the style's focus frame around it, and Space,
    Return or Enter folds or unfolds it; no other key does, and keys typed
    into an input never reach a title, so Enter there still commits text.
    form.fold.<path> is the fold title of the model row at <path>: a nested
    model, a tagged union, an option row whose selected option draws fields
    and an enabled nullable model. It is that row's name label, in its
    level's label column, and folds the row's child rows; a unit follows
    it, after the row's dropdown or Enabled box if any. Other rows show
    their name in normal weight without a mark, and a click on it does
    nothing. form.enabled.<path> is a nullable-model activation QCheckBox
    beside the title. A disabled nullable model draws only its plain label
    (no fold mark, no form.fold.<path>) and form.enabled.<path>, without an
    empty box.
    form.group.<path>.<group> (form.group.<group> at the root level) is the
    block of one Display group of the model level at <path>. It holds the
    level's fields of that group, in declaration order, under a header whose
    fold title form.fold.group.<path>.<group> (form.fold.group.<group> at the
    root) shows the group name in the level's label column. Each level shows
    its ungrouped rows first, then its group blocks in the order of each
    group's first field. A group whose fields are all hidden has no block.
    While any of its rows (or their children) shows an error, the header
    shows a "Contains errors" note; a folded group stays folded.
    Groups start folded; model rows start as Display(collapsed=...) says.
    Folds the user sets on groups and model rows survive rebuilds of this
    editor (option or activation changes), remembered by path.
    form.value.<path> is a QLabel with the resolved value of an Expression-mode
    input; Direct input is not echoed, and the label is hidden while empty.
    Field errors lead the row tooltip, followed by its description and hint.
    Direct text inputs and dropdowns get a light-red Base tint; checkboxes and
    model sections get a red name label. An invalid Expression keeps its mode
    tint and gets a red border. The coloured widget has formError=True while
    invalid, False otherwise. Errors add no row height. form.error displays
    model-wide errors and is hidden while empty.
    Mapping-key/index errors appear at their container field.
    Only current active, instantiated fields have input controls. Model and
    group descendants are described and built on first expansion, or during
    initial construction when expanded. Collapsing retains the same widgets
    and their uncommitted text without writing state. Changing an option or
    nullable activation replaces only that subtree, not unrelated rows.
    Sequences and typed mappings
    use JSON text for the whole leaf; they do not expose key/index write paths.
    Complex direct text uses Python complex-number notation, not an expression.

    set_editing_enabled(enabled=False) makes the form read-only but readable;
    no control is disabled or greyed out. Every form.input QLineEdit is
    read-only (isReadOnly() is True): its text can be selected and copied,
    and its right-click menu keeps only the copying items, with
    form.mode.<path> listed but disabled. Dropdowns and checkboxes
    (form.input and form.enabled) show the current value in normal colours
    and ignore mouse clicks, wheel turns and keys other than Tab and Backtab;
    a wheel turn over them scrolls an enclosing scroll area instead. Option
    command actions stay listed but disabled. Fold titles still take Tab
    focus and fold and unfold on a click, Space, Return or Enter, without
    writing state.

    Build finishes before subscribing. detach or close removes the subscription,
    disables this view and drops local text without changing caller state.
    After QObject destruction or garbage collection of this view, the next
    state publication removes the subscription without touching state.
    refresh and error queries after detach raise RuntimeError.
    validity_changed emits a bool only when combined validity changes; it is not
    an Apply request. errors_changed emits the new errors() result, FormErrors
    or None when valid, only when it differs from the previous result. Local
    parse/rejection changes and state error publications can trigger it, even
    while validity stays False. It never emits after detach.
    GUI-thread use and a live QApplication are required.
    """

    validity_changed = pyqtSignal(bool)
    errors_changed = pyqtSignal(object)

    def __init__(
        self,
        state: FormState[F],
        env: FormEnv,
        parent: QWidget | None = None,
        *,
        write: Callable[[Mapping[FormWritePath, JsonValue]], None] | None = None,
        run_command: Callable[[FormWritePath, str], None] | None = None,
    ) -> None:
        """Build controls for state using env, then subscribe to state changes.

        State/source validation errors are displayed, not replaced by defaults.
        Structural description or source failures propagate; failed construction
        leaves no state subscription. parent controls the ordinary Qt lifetime.

        write decides where parsed input goes. When it is None (the default),
        each parsed batch is installed with state.write(batch, env) using the
        current env. When it is a callable, it receives every parsed batch, a
        mapping from write paths to JSON input with one entry, the edited
        field's path. The editor never writes state itself. The callable installs
        accepted input into this same state (for example after its own checks
        or substitutions); the editor still reads and subscribes to state, so
        input the callable does not install is not shown.
        To refuse a batch, the callable raises WriteRejected(message) before
        writing state. That field then keeps the user's input as local view
        input with message as its error: text inputs keep the typed text and
        mode; a checkbox, dropdown (including a key-or-inline dropdown) or
        Enabled box shows the state's value again. Like unparseable text, it appears in
        local_parse_errors and errors, blocks validity, survives refresh and
        publications, and clears on that field's next accepted write, on
        discard_local_text, or when the field leaves the form. Any other
        exception from the callable propagates.

        run_command decides where a triggered option command goes. When it is
        None (the default), the editor calls state.run_command(path, name,
        env) with the current env. When it is a callable, it receives the
        option field's path and the command name, for example
        (("modules", "qub_pulse"), "to_inline"), and the editor never runs
        the command on state; the callable runs it on this same state, for
        example through the host's guarded entry. It refuses with
        WriteRejected(message) before changing state, which the field shows
        like a rejected write; any other exception propagates. write and
        run_command are passed together or not at all: a host that guards
        writes must also route commands, so passing only one raises
        ValueError before any subscription.
        """
        if (write is None) != (run_command is None):
            raise ValueError("FormEditor needs both write and run_command, or neither")
        app = QApplication.instance()
        if not isinstance(app, QApplication) or app.thread() != QThread.currentThread():
            raise RuntimeError("FormEditor requires the QApplication GUI thread")
        super().__init__(parent)
        self._state: FormState[F] | None = state
        self._env: FormEnv | None = env
        self._host_write = write
        self._host_command = run_command
        self._unsubscribe: Callable[[], None] | None = None
        self._editing_enabled = True
        self._local: dict[FormWritePath, LocalText] = {}
        self._rows: dict[FormWritePath, FieldRow] = {}
        self._groups: list[GroupBlock] = []
        # User folds by model path, and by (model level path, group name).
        self._model_expanded: dict[FormWritePath, bool] = {}
        self._group_expanded: dict[tuple[FormWritePath, str], bool] = {}
        self._state_errors: FormErrors | None = None
        self._last_errors: FormErrors | None = None
        self._valid: bool | None = None
        # Hosts own outer margins; rows keep ROW_SPACING apart.
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(ROW_SPACING)
        self._root_error = QLabel(self)
        make_label_selectable(self._root_error)
        self._root_error.setObjectName("form.error")
        self._root_error.setWordWrap(True)
        self._root_error.hide()
        self._layout.addWidget(self._root_error)
        self._content = QWidget(self)
        self._layout.addWidget(self._content)
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(ROW_SPACING)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._level = _Level((), self._content, content_layout)
        self._publication_count = 0
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

        Read current descriptions for instantiated levels with the new env.
        Never reuse choices or branch metadata from an earlier capture.
        Keep still-active local unparseable or rejected
        input. Never reset state, run domain defaults, poll sources or echo a
        write. External
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
        """Allow user edits (enabled=True, the initial mode) or make the form read-only.

        Read-only (enabled=False) keeps every control enabled and readable:
        text can be selected and copied, fold titles still fold, and no user input
        changes state or local text (see the class docstring for each
        control). Neither mode writes state or drops local text. External
        state changes remain visible while read-only. Raises RuntimeError
        after detach.
        """
        self._require_state()
        if self._editing_enabled == enabled:
            return
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

    def _publish(self, *, redraw: bool = True) -> None:
        state = self._require_state()
        preview = state.preview(self._require_env())
        data = state.input.data
        fields = state.describe((), self._require_env()).children
        self._sync_level(self._level, fields, data, preview, redraw=redraw)
        self._rows = {}
        self._groups = []
        self._index_level(self._level)
        self._local = {
            path: local
            for path, local in self._local.items()
            if path in self._rows
            and local.section == (self._rows[path].field.kind == "model")
        }
        self._state_errors = preview.errors
        self._show_errors()
        self._publication_count += 1

    def _fold_changed(self) -> None:
        # Folding only builds missing rows. It must not redraw existing text
        # or commit it merely because the user hides its body.
        self._publish(redraw=False)

    def _clear_level(self, level: _Level) -> None:
        for child in level.children.values():
            self._clear_level(child)
        for _, body in level.groups.values():
            self._clear_level(body)
        for path, row in level.rows.items():
            self._local.pop(path, None)
            self._retire(row.widget)
        for block, _ in level.groups.values():
            self._retire(block.widget)
        level.rows.clear()
        level.children.clear()
        level.groups.clear()

    def _retire(self, widget: QWidget) -> None:
        # Retire names immediately, but let an emitting Qt control finish.
        focused = QApplication.focusWidget()
        if focused is not None and widget.isAncestorOf(focused):
            # Replacing a branch must not commit text from the retiring row.
            with QSignalBlocker(focused):
                widget.setParent(None)
        else:
            widget.setParent(None)
        widget.deleteLater()

    def _sync_level(
        self,
        level: _Level,
        metadata: tuple[FormFieldDescription, ...],
        data: dict[str, JsonValue],
        preview: FormPreview,
        *,
        redraw: bool,
    ) -> None:
        shown = tuple(item for item in metadata if not item.hidden)
        ungrouped = tuple(item for item in shown if item.group is None)
        self._sync_rows(level, ungrouped, data, preview, redraw=redraw)
        grouped: dict[str, list[FormFieldDescription]] = {}
        for item in shown:
            if item.group is not None:
                grouped.setdefault(item.group, []).append(item)
        for name in tuple(level.groups):
            if name not in grouped:
                block, body = level.groups.pop(name)
                self._clear_level(body)
                self._retire(block.widget)
        for name, members in grouped.items():
            if name not in level.groups:
                block = GroupBlock(
                    level.path,
                    name,
                    tuple(item.path for item in members),
                    level.widget,
                    self._group_expanded,
                    self._fold_changed,
                )
                body = _Level(level.path, block.body, block.rows)
                level.groups[name] = block, body
                level.layout.addWidget(block.widget)
                block.widget.show()
            block, body = level.groups[name]
            block.members = tuple(item.path for item in members)
            if not block.body.isHidden() or body.rows or body.groups:
                # Group membership is already in this layer's metadata. No
                # model description is queried until its own body is expanded.
                self._sync_rows(body, tuple(members), data, preview, redraw=redraw)
        _align_labels(
            [row.label for row in level.rows.values()]
            + [block.label for block, _ in level.groups.values()]
        )

    def _sync_rows(
        self,
        level: _Level,
        metadata: tuple[FormFieldDescription, ...],
        data: dict[str, JsonValue],
        preview: FormPreview,
        *,
        redraw: bool,
    ) -> None:
        wanted = {item.path for item in metadata}
        for path in tuple(level.rows):
            if path not in wanted:
                child = level.children.pop(path, None)
                if child is not None:
                    self._clear_level(child)
                self._local.pop(path, None)
                self._retire(level.rows.pop(path).widget)
        for index, item in enumerate(metadata):
            field = project_field(item, data, preview)
            row = level.rows.get(field.path)
            fresh = row is None or _shape(row.field) != _shape(field)
            if fresh:
                if row is not None:
                    child = level.children.pop(field.path, None)
                    if child is not None:
                        self._clear_level(child)
                    self._retire(row.widget)
                row = FieldRow(
                    field,
                    level.widget,
                    self._write,
                    self._unparsed,
                    self._model_expanded,
                    self._run_command,
                )
                row.connect_fold(self._fold_changed)
                level.rows[field.path] = row
                level.layout.insertWidget(index, row.widget)
                row.widget.show()
            # The new-row branch always assigns row.
            assert row is not None
            if row.field.selected != field.selected:
                child = level.children.pop(field.path, None)
                if child is not None:
                    self._clear_level(child)
                # An earlier empty selection hid the body and fold marker.
                # Restore the structural hint before considering the new layer.
                row.set_children_available(available=field.can_expand)
            local = self._local.get(field.path)
            if local is not None and local.section != (field.kind == "model"):
                self._local.pop(field.path)
                local = None
            if redraw or fresh:
                row.display(field, local, editable=self._editing_enabled)
            self._sync_children(level, row, data, preview, redraw=redraw)
        _align_labels([row.label for row in level.rows.values()])

    def _sync_children(
        self,
        level: _Level,
        row: FieldRow,
        data: dict[str, JsonValue],
        preview: FormPreview,
        *,
        redraw: bool,
    ) -> None:
        if row.group is None or row.rows is None:
            return
        path = row.field.path
        child = level.children.get(path)
        if child is None:
            if row.group.isHidden():
                return
            child = _Level(path, row.group, row.rows)
            level.children[path] = child
        fields = self._require_state().describe(path, self._require_env()).children
        row.set_children_available(available=any(not field.hidden for field in fields))
        self._sync_level(child, fields, data, preview, redraw=redraw)

    def _index_level(self, level: _Level) -> None:
        self._rows.update(level.rows)
        for child in level.children.values():
            self._index_level(child)
        for block, body in level.groups.values():
            self._groups.append(block)
            self._index_level(body)

    def _write(
        self, path: FormWritePath, batch: Mapping[FormWritePath, JsonValue]
    ) -> None:
        """Install batch from the row at path, or keep it local if rejected."""
        state = self._require_state()
        publication = self._publication_count
        self._local.pop(path, None)
        if self._host_write is None:
            state.write(batch, self._require_env())
        else:
            try:
                self._host_write(batch)
            except WriteRejected as exc:
                # The host installed nothing: keep this input local, like
                # unparseable text; the publish below redraws state around it.
                self._local[path] = self._rows[path].rejected_input(exc.message)
        # A changed write already rendered synchronously in the subscriber.
        # Unchanged/rejected writes still need to update view-local errors.
        if self._publication_count == publication:
            self._publish()

    def _run_command(self, path: FormWritePath, name: str) -> None:
        """Run option command name at path, or keep its rejection local."""
        state = self._require_state()
        publication = self._publication_count
        self._local.pop(path, None)
        if self._host_command is None:
            state.run_command(path, name, self._require_env())
        else:
            try:
                self._host_command(path, name)
            except WriteRejected as exc:
                # The host changed nothing: show its message at the field.
                self._local[path] = self._rows[path].rejected_input(exc.message)
        if self._publication_count == publication:
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
                    for path in (errors.messages if errors else {})
                    for member in block.members
                )
            )
        valid = errors is None
        if valid != self._valid:
            self._valid = valid
            self.validity_changed.emit(valid)
        if self._state is not None and errors != self._last_errors:
            self._last_errors = errors
            self.errors_changed.emit(errors)

    def _error_target(self, path: FormPath) -> FormWritePath | None:
        for length in range(len(path), 0, -1):
            prefix = path[:length]
            for candidate in self._rows:
                if candidate == prefix and self._rows[candidate].widget.isVisibleTo(
                    self._content
                ):
                    return candidate
        return None


def _align_labels(labels: list[QLabel]) -> None:
    """Give labels one fixed width, the widest label's, as one label column.

    Every control beside these labels then starts at the same left edge.
    """
    width = max((label.sizeHint().width() for label in labels), default=0)
    for label in labels:
        label.setFixedWidth(width)
