"""Qt rendering and local text drafts for the caller-owned form state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from weakref import ref

from pydantic import JsonValue
from pydantic_core import PydanticCustomError
from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import QSignalBlocker, Qt, QThread
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from zcu_tools.gui.form import (
    FormCommandError,
    FormConversionError,
    FormEnv,
    FormErrors,
    FormFieldDescription,
    FormInputRejected,
    FormModel,
    FormPath,
    FormPreview,
    FormState,
    FormWritePath,
    WriteRejected,
)
from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.session.context_types import ContextSourceUnavailableError
from zcu_tools.gui.widgets import error_colour, set_foreground

from ._controls import FieldRow, GroupBlock, LocalText, RowActions, make_error_label
from ._fold import ROW_SPACING
from ._view import FieldView, present_fields, project_field


@dataclass
class _Level:
    """Qt ownership for one instantiated model or display-group body."""

    path: FormWritePath
    widget: QWidget
    layout: QVBoxLayout
    elements: bool = False  # Direct children of a list with an element picker.
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
        field.section,
    )


def _order_widgets(level: _Level, fields: tuple[FormFieldDescription, ...]) -> None:
    """Place retained rows and groups at their visible declaration positions."""
    placed_groups: set[str] = set()
    index = 0
    for item in fields:
        if item.group is None:
            widget = level.rows[item.path].widget
        else:
            if item.group in placed_groups:
                continue
            placed_groups.add(item.group)
            widget = level.groups[item.group][0].widget
        # A presentation change can move a retained row into or out of a group.
        level.layout.insertWidget(index, widget)
        index += 1


def _completion_candidates(env: FormEnv) -> tuple[str, ...]:
    """Project sorted alias names and one component-field level once per capture.

    Device/unbound aliases have names only. Source faults disable suggestions;
    the ordinary form preview still reports errors at consuming fields.
    """
    context = env.parameters
    if context is None:
        return ()
    try:
        aliases = context.resolved_aliases()
        point = context.document()
    except ContextSourceUnavailableError:
        return ()
    candidates: list[str] = []
    for row in aliases:
        name = f"${row.alias}"
        candidates.append(name)
        component = (
            point.components.get(row.resolved)
            if isinstance(row.resolved, str)
            else None
        )
        if component is not None:
            candidates.extend(
                f"{name}.{field}" for field in type(component).model_fields
            )
    return tuple(sorted(candidates))


class FormEditor[F: FormModel](QWidget):
    """Edit a caller-owned FormState on the Qt GUI thread, without submitting it.

    The editor borrows state and the host-captured env. A text field commits
    its text after 500 ms without typing, on Enter or on ordinary focus loss,
    not on each keystroke. Text that already shows state's input writes nothing, and
    read-only rows commit nothing. Checkboxes, dropdowns, Enabled boxes and the
    Expression action write at once. Committed parsed input writes the state,
    or goes to the host's write callable (see __init__); committed unparseable
    text and host-rejected input remain local and block editor validity.
    Uncommitted text changes neither state nor validity, and an external state
    publication redraws it with the state's input. Evaluation, choices, branch
    defaults and model validation belong to state.
    WriteRejected, FormConversionError, FormInputRejected and source errors
    expressed as PydanticCustomError become field-local errors, retaining the
    previous committed state; other failures propagate.
    No Engine update, revision check, source capture, persistence or I/O occurs.
    Construction and refresh project completion names once from the supplied
    capture, using resolved_aliases() and document().components. Expression
    inputs offer case-sensitive, lexicographically sorted $alias or one-level
    $alias.field completions with no selection and only whitespace or end of
    text immediately after the cursor. A following identifier character,
    operator or closing parenthesis blocks hints. Device aliases offer only names.
    A grey suffix is painted inline, shifting any displayed right-hand text
    without changing the draft. Its accessible description announces the same
    suffix and keyboard controls. First Tab accepts without committing or moving
    focus; consecutive Tab presses cycle the original candidates, wrapping.
    Esc dismisses until another edit/cursor move. Focus loss ends a cycle.
    Direct/read-only inputs, selections and deeper paths have no suggestions.

    Public child object names use model-field paths and indexes joined by dots:
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
    option's value as item data. The core's selected index reflects an explicit
    selection override, or raw input when there is no override. An unmatched
    selection is kept: a last item shows it, a nonempty string as itself and
    other JSON as its JSON text, plus " (missing)", with the selection as data.
    This item is selected, state is unchanged and the field's errors still show.
    An unfilled field selects nothing. Choosing an option writes its value
    at the field as one batch; the core's converter, when declared, decides
    the new child input. Rows follow the actual branch independently of the
    selection and show below the dropdown (tags its value fixes have no rows).
    Such a row has no
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
    header at its natural width, and folds the row's child rows. Section
    dropdowns fill the remaining width independently of scalar labels. Units
    appear only beside scalar inputs, in an equal-width column per level,
    with an empty slot for unitless inputs. Other rows show
    their name in normal weight without a mark, and a click on it does
    nothing. form.enabled.<path> is a nullable-model activation QCheckBox
    beside the title. A disabled nullable model draws only its plain label
    (no fold mark, no form.fold.<path>) and form.enabled.<path>, without an
    empty box.
    form.group.<path>.<group> (form.group.<group> at the root level) is the
    block of one Display group of the model level at <path>. It holds the
    level's fields of that group, in declaration order, under a header whose
    fold title form.fold.group.<path>.<group> (form.fold.group.<group> at the
    root) shows the group name at its natural width. Rows retain their
    declaration order, with each group block at its first visible member's
    position. A group whose fields are all hidden has no block.
    While any of its rows (or their children) shows an error, the header
    shows a "Contains errors" note naming invalid child paths; folds stay unchanged.
    Groups start folded except Sweep, whose member axes also start open. A
    sole nonnullable expandable Sweep member uses a combined title and keeps
    its canonical path, selector and commands. These hints use Display.group,
    not field names. Model rows start folded unless their subtree has errors on the editor's first publication. Saved user
    folds take precedence, and later refreshes keep the established folds.
    Folds the user sets on groups and model rows survive rebuilds of this
    editor (option or activation changes), remembered by path.
    form.value.<path> is a QLabel with the resolved value of an Expression-mode
    input; Direct input is not echoed, and the label is hidden while empty.
    Field errors lead the row tooltip, followed by its description and hint.
    Direct text inputs and dropdowns get a light-red Base tint; checkboxes and
    model sections get a red name label. An invalid Expression keeps its mode
    tint and gets a red border. The coloured widget has formError=True while
    invalid, False otherwise. form.error.<path> shows selectable inline error text
    below that row's control. form.error.summary shows the first error and its
    canonical path at the bottom, including folded or hidden children. form.error
    displays model-wide or otherwise unlocated errors. Empty labels are hidden.
    Mapping-key/index errors appear at their container field, except within a
    FixedList where the original index addresses each model element.
    Only current active, instantiated fields have input controls. Model and
    group descendants are described and built on first expansion, or during
    initial construction when expanded. Collapsing retains the same widgets
    and their uncommitted text without writing state. Changing an option or
    nullable activation replaces only that subtree, not unrelated rows.
    A FixedList has form.picker.<path>, a navigation-only QComboBox listing
    "index: name". It draws the picked element's fields directly below the list
    row, without an element title or extra indentation. Field labels and full
    indexed paths stay unchanged. Other elements remain mounted but hidden. Switching never writes state or commits
    hidden text, and preserves drafts and parse errors. The list row marks
    hidden errors with their element name in its tooltip. Participants remain
    a separate JSON list input. Other sequences and typed mappings use JSON
    text for the whole leaf; they do not expose key/index write paths.
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
    writing state. Model-list pickers also remain usable for navigation.

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
        select: Callable[[FormWritePath, JsonValue], None] | None = None,
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
        like a rejected write.

        select receives (path, option.value) for dropdown changes. Without host
        callbacks, the editor calls state.select(path, selection, env); it never
        infers selection from a value write. A host callback must run selection
        through its same busy/owner guard before publishing to this state.
        write, select and run_command are all supplied or all omitted; a partial
        set raises ValueError before subscribing. FormConversionError and
        FormInputRejected show recoverable row errors and preserve state.
        Other exceptions propagate.
        """
        if len({write is None, run_command is None, select is None}) != 1:
            raise ValueError("FormEditor needs write, select and run_command together")
        app = QApplication.instance()
        if not isinstance(app, QApplication) or app.thread() != QThread.currentThread():
            raise RuntimeError("FormEditor requires the QApplication GUI thread")
        super().__init__(parent)
        self._state: FormState[F] | None = state
        self._env: FormEnv | None = env
        self._completion_names = _completion_candidates(env)
        self._host_write = write
        self._host_command = run_command
        self._host_select = select
        self._scope = LifetimeScope(self)
        self._close_control: dict[QWidget, Callable[[], None]] = {}
        self._scope.add_cleanup(self._discard_view)
        self._scope.add_cleanup(self._close_controls)
        self._editing_enabled = self._idle_commit = True
        self._local: dict[FormWritePath, LocalText] = {}
        self._rows: dict[FormWritePath, FieldRow] = {}
        self._groups: list[GroupBlock] = []
        # Established model folds by path; user group folds by (level path, name).
        self._model_expanded: dict[FormWritePath, bool] = {}
        self._group_expanded: dict[tuple[FormWritePath, str], bool] = {}
        self._state_errors: FormErrors | None = None
        self._last_errors: FormErrors | None = None
        self._valid: bool | None = None
        # Hosts own outer margins; rows keep ROW_SPACING apart.
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(ROW_SPACING)
        self._root_error = make_error_label(self, "form.error")
        self._layout.addWidget(self._root_error)
        self._content = QWidget(self)
        self._layout.addWidget(self._content, stretch=1)
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(ROW_SPACING)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._summary = make_error_label(self, "form.error.summary")
        self._layout.addWidget(self._summary)
        self._level = _Level((), self._content, content_layout)
        self._publication_count = 0
        self._publish()
        self._follow_state(state)

    def _follow_state(self, state: FormState[F]) -> None:
        # The state callback must not keep the QWidget alive. No Python slot is
        # connected to destroyed: cyclic GC may clear such a slot before Qt
        # calls it while deleting the widget, which crashes the process. A
        # publication that finds this view collected or deleted unsubscribes.
        weak_view = ref(self)
        publish = self._scope.callback(self._publish)

        def published() -> None:
            view = weak_view()
            if view is None or view._scope.closed:
                # Bound below, before state can publish to this callback.
                unsubscribe()
            else:
                publish()

        unsubscribe = state.subscribe(published)
        self._scope.add_cleanup(unsubscribe)

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
        names = _completion_candidates(env)
        self._env = env
        self._completion_names = names
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

    def set_idle_commit(self, *, enabled: bool) -> None:
        """Enable 500 ms text commits, or require Enter/ordinary focus loss.

        Disabling cancels pending expiry without parsing or discarding text.
        Dropdowns, checkboxes and commands still commit immediately.
        Applies to mounted rows and rows built later. Default is enabled.
        Raises RuntimeError after detach.
        """
        self._require_state()
        self._idle_commit = enabled
        for row in self._rows.values():
            row.set_idle_commit(enabled=enabled)

    def has_pending_text(self) -> bool:
        """Report text not yet installed in state, without parsing or committing.

        Includes folded rows and rejected text. Hosts use this to preserve an
        active edit across source publications. Raises RuntimeError after detach.
        """
        self._require_state()
        return any(row.has_pending_text() for row in self._rows.values())

    def local_parse_errors(self) -> FormErrors | None:
        """Return detached local input errors, or None when state holds all input.

        Local input is unparseable text or input the host's write rejected;
        its message is the parse error or the WriteRejected message. Paths
        identify active controls. This query does not parse, write, or imply
        that the model is valid. Raises RuntimeError after detach.
        """
        self._require_state()
        messages: dict[FormPath, str] = {
            path: local.error
            for path, local in self._local.items()
            if path in self._rows
        }
        return FormErrors(messages) if messages else None

    def errors(self) -> FormErrors | None:
        """Return merged local parse and current state errors, or None if valid.

        Local errors at a path take precedence over state errors at that path.
        Querying performs no state write. Raises RuntimeError after detach.
        """
        self._require_state()
        messages = dict(self._state_errors.messages) if self._state_errors else {}
        local_errors = self.local_parse_errors()
        if local_errors is not None:
            messages.update(local_errors.messages)
        return FormErrors(messages) if messages else None

    def detach(self) -> None:
        """Idempotently unsubscribe, disable the view and discard local text.

        Do not reset, close or submit caller-owned state. Later publications do
        not update this view. Queued scalar commits become no-ops, including
        commits from retired rows awaiting Qt deletion. A host write callback
        may detach the editor; that commit will not redraw it afterwards.
        The QWidget may still be destroyed normally.
        """
        self._scope.close()

    def _close_controls(self) -> None:
        for close in self._close_control.values():
            close()
        self._close_control.clear()

    def _discard_view(self) -> None:
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
        close = self._close_control.pop(widget, None)
        if close is not None:
            close()
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
        shown = present_fields(tuple(item for item in metadata if not item.hidden))
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
            self._group_expanded.setdefault((level.path, name), name == "Sweep")
            if name not in level.groups:
                block = GroupBlock(
                    level.path,
                    name,
                    tuple(item.path for item in members),
                    level.widget,
                    self._group_expanded,
                    self._scope.callback(self._fold_changed),
                )
                self._close_control[block.widget] = block.close
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
                # Hidden wrappers may still cache hints from their empty bodies.
                block.layout.invalidate()
        _order_widgets(level, shown)
        level.layout.invalidate()

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
        selections = self._require_state().input.selections
        for index, item in enumerate(metadata):
            field = project_field(item, data, preview, selections)
            row = level.rows.get(field.path)
            fresh = row is None or _shape(row.field) != _shape(field)
            if fresh:
                self._remember_initial_fold(field, preview.errors)
                if row is not None:
                    child = level.children.pop(field.path, None)
                    if child is not None:
                        self._clear_level(child)
                    self._retire(row.widget)
                row = FieldRow(
                    field,
                    level.widget,
                    self._scope.callback(self._write),
                    self._scope.callback(self._unparsed),
                    self._model_expanded,
                    RowActions(
                        command=self._scope.callback(self._run_command),
                        select=self._scope.callback(self._select),
                    ),
                )
                row.set_idle_commit(enabled=self._idle_commit)
                self._close_control[row.widget] = row.close
                row.configure_body(element=level.elements)
                row.connect_fold(self._fold_changed)
                row.connect_pick(self._pick_changed)
                level.rows[field.path] = row
                row.widget.show()
            # The new-row branch always assigns row.
            assert row is not None
            level.layout.insertWidget(index, row.widget)
            if row.field.branch != field.branch:
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
                row.set_completion_candidates(self._completion_names)
            self._sync_children(level, row, data, preview, redraw=redraw)
            row.layout.invalidate()
        level.layout.invalidate()
        _align_scalar_columns(list(level.rows.values()))

    def _remember_initial_fold(
        self, field: FieldView, errors: FormErrors | None
    ) -> None:
        if field.kind not in ("model", "elements"):
            return
        # Initial errors include descendants whose rows are not built yet.
        self._model_expanded.setdefault(
            field.path,
            not field.collapsed
            or (
                self._publication_count == 0
                and errors is not None
                and any(
                    path[: len(field.path)] == field.path for path in errors.messages
                )
            ),
        )

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
            child = _Level(path, row.group, row.rows, elements=row.picker is not None)
            level.children[path] = child
        fields = self._require_state().describe(path, self._require_env()).children
        row.set_children_available(available=any(not field.hidden for field in fields))
        self._sync_level(child, fields, data, preview, redraw=redraw)
        if row.picker is not None:
            _show_picked(row, child.rows)

    def _pick_changed(self) -> None:
        # Navigation only: hidden element rows keep their widgets, so local
        # text and errors survive, and nothing is written or committed.
        for row in self._rows.values():
            if row.picker is not None:
                _show_picked(row, self._rows)
        self._show_errors()

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
        """Install a row batch, ignoring callbacks after its editor detached."""
        state = self._require_state()
        publication = self._publication_count
        self._local.pop(path, None)
        try:
            if self._host_write is None:
                state.write(batch, self._require_env())
            else:
                self._host_write(batch)
        except (
            WriteRejected,
            FormConversionError,
            FormInputRejected,
            PydanticCustomError,
        ) as exc:
            # Both routes installed nothing. Redraw the committed selector,
            # retaining unrelated local text; never retry a refused conversion.
            message = exc.message if isinstance(exc, WriteRejected) else str(exc)
            self._local[path] = self._rows[path].rejected_input(message)
        # A changed write already rendered synchronously in the subscriber.
        # Unchanged/rejected writes still need to update view-local errors.
        if not self._scope.closed and self._publication_count == publication:
            self._publish()

    def _select(self, path: FormWritePath, selection: JsonValue) -> None:
        """Dispatch dropdown selection through the same owner guard as value writes."""
        state = self._require_state()
        publication = self._publication_count
        self._local.pop(path, None)
        try:
            if self._host_select is None:
                state.select(path, selection, self._require_env())
            else:
                self._host_select(path, selection)
        except (
            WriteRejected,
            FormConversionError,
            FormInputRejected,
            PydanticCustomError,
        ) as exc:
            message = exc.message if isinstance(exc, WriteRejected) else str(exc)
            self._local[path] = self._rows[path].rejected_input(message)
        if self._publication_count == publication:
            self._publish()

    def _run_command(self, path: FormWritePath, name: str) -> None:
        """Dispatch owner commands, guarding only declared participating local text."""
        if not self._editing_enabled:
            return
        state = self._require_state()
        field = state.describe(path, self._require_env()).target
        command = (
            next((item for item in field.commands if item.name == name), None)
            if field
            else None
        )
        if command is not None and any(key in self._local for key in command.inputs):
            return
        self._dispatch_command(
            path, name, command.inputs if command is not None else None
        )

    def _dispatch_command(
        self,
        path: FormWritePath,
        name: str,
        inputs: tuple[FormWritePath, ...] | None,
    ) -> None:
        """Keep local drafts across a guarded command and its state publication.

        inputs names participating paths for a field command; None denotes an
        option action, whose row-local error is cleared before dispatch.
        """
        saved = dict(self._local) if inputs is not None else {}
        publication = self._publication_count
        if inputs is None:
            self._local.pop(path, None)
        try:
            if self._host_command is None:
                self._require_state().run_command(path, name, self._require_env())
            else:
                self._host_command(path, name)
        except (WriteRejected, FormInputRejected, PydanticCustomError) as exc:
            message = exc.message if isinstance(exc, WriteRejected) else str(exc)
            self._local[path] = self._rows[path].rejected_input(message)
        except FormCommandError:
            if inputs is None:
                raise
            # State preview already reports the participating expression/missing
            # error. Do not create a local parse error on a valid peer field.
            self._local.update(saved)
        else:
            if inputs is not None:
                self._local.update(
                    {key: local for key, local in saved.items() if key not in inputs}
                )
        if self._publication_count == publication or saved:
            self._publish()

    def _unparsed(self, path: FormWritePath, local: LocalText) -> None:
        self._local[path] = local
        self._show_errors()

    def _show_errors(self) -> None:
        errors = self.errors()
        messages = dict(errors.messages) if errors else {}
        first = next(iter(messages.items()), None)
        summary = ""
        if first is not None:
            path, message = first
            location = ".".join(str(part) for part in path)
            summary = f"{location}: {message}" if location else message
        self._summary.setText(summary)
        set_foreground(self._summary, error_colour(self))
        self._summary.setVisible(bool(summary))
        self._root_error.setText(messages.pop((), ""))
        by_row: dict[FormWritePath, list[str]] = {}
        for path, message in messages.items():
            target = self._error_target(path)
            if target is not None:
                row = self._rows[target]
                if row.picker is not None and len(path) > len(target):
                    # The picked element is not drawn: name it in this row.
                    message = f"{_element_text(row, path)}: {message}"
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
            block.set_errors(
                tuple(
                    path
                    for path in (errors.messages if errors else {})
                    if any(path[: len(member)] == member for member in block.members)
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
                row = self._rows[candidate]
                if (
                    candidate == prefix
                    and not row.label.isHidden()
                    and row.widget.isVisibleTo(self._content)
                ):
                    return candidate
        return None


def _show_picked(row: FieldRow, rows: Mapping[FormWritePath, FieldRow]) -> None:
    """Draw only the element row the picker selects."""
    picked = row.picked()
    for path, element in rows.items():
        if len(path) == len(row.field.path) + 1 and path[:-1] == row.field.path:
            element.widget.setVisible(path[-1] == picked)


def _element_text(row: FieldRow, path: FormPath) -> str:
    """Name the element and field a path below a picker row addresses."""
    index = path[len(row.field.path)]
    name = next(
        (item.name for item in row.field.elements or () if item.index == index),
        str(index),
    )
    return ".".join((name, *(str(part) for part in path[len(row.field.path) + 1 :])))


def _align_scalar_columns(rows: list[FieldRow]) -> None:
    """Align scalar label/unit columns, leaving section titles at natural width."""
    _align_labels([row.label for row in rows if not row.field.section])
    units = [row.unit for row in rows if row.unit is not None]
    _align_labels(units)
    has_units = any(unit.text() for unit in units)
    for unit in units:
        unit.setVisible(has_units)


def _align_labels(labels: list[QLabel]) -> None:
    """Give labels one fixed width, the widest label's, as one label column.

    Every control beside these labels then starts at the same left edge.
    """
    width = max((label.sizeHint().width() for label in labels), default=0)
    for label in labels:
        label.setFixedWidth(width)
