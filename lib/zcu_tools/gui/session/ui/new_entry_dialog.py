"""NewEntryDialog — create one library entry from a persistent template."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.cfg_model import Template, TemplateDef
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.library_entry import (
    LIBRARY_WRITE_ERRORS,
    LibraryEntryInvalid,
    LibraryKind,
    ModuleEntryForm,
    WaveformEntryForm,
    library_entry_kind,
    new_library_entry,
    template_placeholders,
)
from zcu_tools.gui.widgets.form import FormEditor

if TYPE_CHECKING:
    from zcu_tools.gui.form import FormModel
    from zcu_tools.gui.session.arb_waveform import ArbWaveformKeys
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.widgets.dialog_presenter import DialogPresenter

logger = logging.getLogger(__name__)


class NewEntryDialog(QDialog):
    """Create a library root from one persistent template, editing it first.

    Public child object names: the dialog itself is inspect.new_entry;
    inspect.new_entry.template is the QComboBox of templates, modules first
    and then waveforms, each shown as "<label>  (<kind>)" with the template
    id as item data; inspect.new_entry.name is the Name QLineEdit;
    inspect.new_entry.scroll is the QScrollArea holding the FormEditor of the
    selected template; inspect.new_entry.error is a QLabel shown only when no
    form can be built (for example, no working point is selected). The
    buttons are "Create" and "Cancel".

    Choosing a template reads one fixed capture from ctrl and builds a fresh
    entry form and FormEditor from it: fields hold the template's initial
    values, folds are the editor defaults and the scroll is at the top. The
    Name is the template's default_name, with _2, _3, ... appended while that
    name is taken in its family, until the user types in the Name field; from
    then on a template change keeps the typed Name. Nothing else is carried
    over between templates.

    Create writes the form's Spec at Name through ctrl.apply_library_change
    at that capture, then accepts the dialog with created set to (kind,
    name). An empty Name, a Name taken in the template's family, invalid
    form input or a rejected write is reported through dialog_presenter and
    writes nothing; the dialog stays open. The caller chooses the modality;
    InspectDialog opens it window-modal with open().
    """

    def __init__(
        self,
        ctrl: ContextControlPort,
        templates: Mapping[str, TemplateDef],
        *,
        arb_waveforms: ArbWaveformKeys,
        dialog_presenter: DialogPresenter,
        parent: QWidget | None = None,
    ) -> None:
        """Build the dialog and the form of the first template.

        ctrl supplies captures and the library write; templates is the
        persistent catalog by template id; arb_waveforms lists the arb asset
        names offered to arb data fields at each capture. Template placeholders
        bind to resolved targets of same-named aliases in that capture. Missing
        or unresolved aliases show a binding error and disable Create; choosing
        another template rebuilds the form. ctrl must provide a readable capture;
        FailedPreconditionError from read_context is shown in the dialog.
        Other capture and arb_waveforms provider failures
        propagate. dialog_presenter shows Create failures. A template whose
        module is not a persistent module or waveform Spec raises TypeError.
        """
        super().__init__(parent)
        self._ctrl = ctrl
        self._templates = templates
        self._arb_waveforms = arb_waveforms
        self._presenter = dialog_presenter
        self._entry: ModuleEntryForm | WaveformEntryForm | None = None
        self._editor: FormEditor[FormModel] | None = None
        # Set by the first keystroke in Name; programmatic setText never sets it.
        self._name_typed = False
        # (kind, name) of a successful Create; None until then.
        self.created: tuple[LibraryKind, str] | None = None
        self.setObjectName("inspect.new_entry")
        self.setWindowTitle("Create ModuleLibrary entry")
        self.resize(520, 560)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._template_combo = QComboBox()
        self._template_combo.setObjectName("inspect.new_entry.template")
        kinds = {
            template_id: library_entry_kind(definition.module)
            for template_id, definition in templates.items()
        }
        for family in ("module", "waveform"):
            for template_id, definition in templates.items():
                if kinds[template_id] == family:
                    self._template_combo.addItem(
                        f"{definition.label}  ({family})", userData=template_id
                    )
        form.addRow("Template:", self._template_combo)
        self._name_edit = QLineEdit()
        self._name_edit.setObjectName("inspect.new_entry.name")
        form.addRow("Name:", self._name_edit)
        layout.addLayout(form)

        self._error = QLabel()
        self._error.setObjectName("inspect.new_entry.error")
        self._error.setWordWrap(True)
        self._error.hide()
        layout.addWidget(self._error)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("inspect.new_entry.scroll")
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self._create_btn = QPushButton("Create")
        cancel_btn = QPushButton("Cancel")
        buttons.addWidget(self._create_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        self._name_edit.textEdited.connect(self._on_name_typed)
        self._template_combo.currentIndexChanged.connect(self._on_template_changed)
        self._create_btn.clicked.connect(self._on_create)
        cancel_btn.clicked.connect(self.reject)
        self.finished.connect(self._detach_editor)
        self._load_template()

    def _on_name_typed(self, _text: str) -> None:
        self._name_typed = True

    def _on_template_changed(self, _index: int) -> None:
        self._load_template()

    def _load_template(self) -> None:
        """Replace the form with a fresh one for the selected template."""
        self._detach_editor()
        self._entry = None
        template_id = self._template_combo.currentData()
        if not isinstance(template_id, str):
            self._show_error("No templates are available.")
            return
        try:
            parameters = self._ctrl.read_context()
        except FailedPreconditionError as exc:
            self._show_error(str(exc))
            return
        bindings = {
            row.alias: row.resolved
            for row in parameters.resolved_aliases()
            if row.resolved is not None
        }
        missing = template_placeholders(self._templates[template_id]) - bindings.keys()
        if missing:
            names = "、".join(f"{{{name}}}" for name in sorted(missing))
            self._show_error(
                f"template 需要綁定 {names}，請先在 Aliases 設定同名 alias"
            )
            return
        entry = new_library_entry(
            parameters,
            Template(template_id, bindings=bindings),
            self._templates,
            arb_names=self._arb_waveforms.list_data_keys(),
        )
        editor = FormEditor(entry.state, entry.env)
        # setWidget deletes the previous form, so folds and scroll start fresh.
        self._scroll.setWidget(editor)
        self._entry = entry
        self._editor = editor
        self._error.hide()
        self._create_btn.setEnabled(True)
        if not self._name_typed:
            self._name_edit.setText(
                self._suggest_name(self._templates[template_id], entry)
            )

    def _show_error(self, message: str) -> None:
        self._scroll.setWidget(QWidget())
        self._error.setText(message)
        self._error.show()
        self._create_btn.setEnabled(False)

    def _detach_editor(self, *_args: object) -> None:
        if self._editor is not None:
            self._editor.detach()
            self._editor = None

    @staticmethod
    def _suggest_name(
        definition: TemplateDef, entry: ModuleEntryForm | WaveformEntryForm
    ) -> str:
        """Return default_name, suffixed _2, _3, ... while taken in its family."""
        base = definition.default_name
        if not base:
            return ""
        taken = _family_keys(entry)
        name = base
        suffix = 2
        while name in taken:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _on_create(self) -> None:
        entry, editor = self._entry, self._editor
        if entry is None or editor is None:
            return
        name = self._name_edit.text().strip()
        if not name:
            self._presenter.warning(
                self, "Invalid name", "Entry name must not be empty."
            )
            return
        if name in _family_keys(entry):
            self._presenter.warning(
                self, "Create failed", f"A {entry.kind} named {name!r} already exists."
            )
            return
        try:
            local_errors = editor.local_parse_errors()
            if local_errors is not None:
                raise LibraryEntryInvalid(entry.kind, local_errors)
            self._ctrl.apply_library_change(entry.parameters, entry.put(name))
        except LIBRARY_WRITE_ERRORS as exc:
            logger.info("Library entry create failed: %s", exc)
            self._presenter.warning(self, "Create failed", str(exc))
            return
        self.created = (entry.kind, name)
        self.accept()


def _family_keys(entry: ModuleEntryForm | WaveformEntryForm) -> tuple[str, ...]:
    library = entry.parameters.library
    if entry.kind == "module":
        return library.module_keys()
    return library.waveform_keys()


__all__ = ["NewEntryDialog"]
