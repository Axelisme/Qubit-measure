"""InspectDialog — measure's context inspector.

Subclasses the app-agnostic ``InspectDialogBase`` (md tab + ml view/rename/delete)
and adds the ml *create / modify* path, which drags the CfgEditor (a measure
concern) and so cannot live in the session core. The two edit dialogs
(``_MlModifyDialog`` / ``_MlCreateDialog``) stay here with it.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Literal

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.main.adapter import CfgSchema
from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
from zcu_tools.gui.session.services.context import MlEntryValidationError
from zcu_tools.gui.session.ui.inspect_base import InspectDialogBase

from .cfg_form import CfgFormWidget

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.event_bus import BaseEventBus

logger = logging.getLogger(__name__)


_MlItemKind = Literal["module", "waveform"]


class _MlModifyDialog(QDialog):
    """Edit an EXISTING ModuleLibrary entry (fixed shape).

    Name and type/style are read-only — modify never changes shape (to change
    shape, delete the entry and create a new one from a role). Creating new
    entries goes through ``_MlCreateDialog`` / ``create_from_role``.
    """

    def __init__(
        self,
        ctrl: Controller,
        item_kind: _MlItemKind,
        name: str,
        cfg: Any,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not name or cfg is None:
            raise ValueError("Modify requires both name and cfg.")

        self._ctrl = ctrl
        self._item_kind = item_kind
        self._name = name
        self.setWindowTitle(f"Modify {item_kind.capitalize()}")
        self.setMinimumSize(560, 500)

        layout = QVBoxLayout(self)

        # ADR-0006: modify an existing ml entry is the UI twin of the agent's
        # open(from_name) → edit → commit flow. Open a committable session loaded
        # from the live ml; Save commits via the single write authority. No
        # UI-side schema build / lowering / raw write.
        self._cfg_editor_owner = f"inspect-{uuid.uuid4().hex[:8]}"
        editor_id, _ = self._ctrl.open_cfg_editor(
            item_kind, from_name=name, gc=False, owner_key=self._cfg_editor_owner
        )
        root = self._ctrl.get_cfg_editor_root(editor_id)
        discriminator = self._read_discriminator(
            CfgSchema(spec=root.spec, value=root.get_value())
        )

        form = QFormLayout()
        form.addRow("Name:", QLabel(name))
        form.addRow(
            "Type:" if item_kind == "module" else "Style:", QLabel(discriminator)
        )
        layout.addLayout(form)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        # CfgFormWidget attaches to the service-owned LiveModel (ADR-0008); edits
        # land in that draft and enter the live ModuleLibrary only on commit.
        self._form_widget = CfgFormWidget()
        self._scroll.setWidget(self._form_widget)
        layout.addWidget(self._scroll, stretch=1)

        self._warning_label = QLabel()
        self._warning_label.setStyleSheet("color: red;")
        layout.addWidget(self._warning_label)

        btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._form_widget.validity_changed.connect(self._validate)
        cancel_btn.clicked.connect(self.reject)
        self._save_btn.clicked.connect(self._on_save)
        # Detach + tear down the service-owned model when the dialog closes.
        self.finished.connect(self._close_cfg_editor)

        self._form_widget.attach(root)
        self._validate()

    def _close_cfg_editor(self, *_: Any) -> None:
        # Detach the widget, then tear down the service-owned model (ADR-0008).
        self._form_widget.detach()
        editor_id = self._ctrl.editor_id_for_owner(self._cfg_editor_owner)
        if editor_id is not None:
            self._ctrl.teardown_cfg_editor(editor_id)

    @property
    def _discriminator_label(self) -> str:
        return "type" if self._item_kind == "module" else "style"

    def _read_discriminator(self, schema: CfgSchema) -> str:
        value = schema.value.fields[self._discriminator_label]
        raw_value = getattr(value, "value", None)
        if not isinstance(raw_value, str):
            raise RuntimeError(
                f"Invalid {self._discriminator_label} value {raw_value!r}"
            )
        return raw_value

    def _validate(self, *_: Any) -> None:
        if self._form_widget.is_valid():
            self._warning_label.setText("")
            self._save_btn.setEnabled(True)
        else:
            self._warning_label.setText("Configuration is invalid.")
            self._save_btn.setEnabled(False)

    def _on_save(self) -> None:
        # ADR-0006: commit the service-owned session through the single write
        # authority (lowering + register happen there). No UI-side lowering.
        editor_id = self._ctrl.editor_id_for_owner(self._cfg_editor_owner)
        if editor_id is None:
            return
        try:
            self._ctrl.commit_cfg_editor(editor_id, self._name)
        except MlEntryValidationError as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))
            return

        self.accept()

    def clear(self) -> None:
        # Teardown of the service-owned model happens in _close_cfg_editor (also
        # wired to `finished`); detach is idempotent, so this just ensures the
        # widget is unbound.
        self._form_widget.detach()


class _MlCreateDialog(QDialog):
    """Create a new ml module/waveform from a role (the single create path).

    One-shot: pick a role + a name → the role's factory seeds the value
    (md-linked defaults for named roles, structural zeros for ``:blank`` roles)
    and registers it directly into ml (no editable form here). To change the
    entry afterwards, use Modify.
    """

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self.setWindowTitle("Create ModuleLibrary entry")

        # The (item_kind, name) of a successful create, read by the parent after
        # the dialog closes to chain straight into Modify. None until create wins.
        self.created: tuple[_MlItemKind, str] | None = None
        # True once the user has typed into the name field by hand: from then on
        # switching role must not clobber their name (least surprise). Qt only
        # fires textEdited on user keystrokes, never on programmatic setText, so
        # seeding the suggestion below does not set this flag.
        self._name_edited = False

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Pick a role and a name. Named roles seed md-linked defaults; "
            "'Blank: …' roles seed an empty shape. Edit afterwards via Modify."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self._role_combo = QComboBox()
        catalog = self._ctrl.get_role_catalog()
        # Modules then waveforms, each labelled; role entry stashed on the item.
        for kind in ("module", "waveform"):
            for entry in catalog.entries_for(kind):  # type: ignore[arg-type]
                self._role_combo.addItem(f"{entry.label}  ({kind})", userData=entry)
        form.addRow("Role:", self._role_combo)
        self._name_edit = QLineEdit()
        form.addRow("Name:", self._name_edit)
        layout.addLayout(form)

        # Seed the name with the initial role's convention-based suggestion.
        initial = self._role_combo.currentData()
        if initial is not None:
            self._name_edit.setText(self._suggest_name(initial))

        self._name_edit.textEdited.connect(self._on_name_edited)
        self._role_combo.currentIndexChanged.connect(self._on_role_changed)

        btn_row = QHBoxLayout()
        create_btn = QPushButton("Create")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(create_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        create_btn.clicked.connect(self._on_create)
        cancel_btn.clicked.connect(self.reject)

    def _suggest_name(self, entry: Any) -> str:
        """Convention-based name suggestion for ``entry``, de-duplicated.

        Blank roles carry no ``default_name`` -> empty (the user must name it).
        Otherwise append ``_2``/``_3``/… until the name is free in the live ml.
        """
        base = entry.default_name
        if not base:
            return ""
        name = base
        suffix = 2
        while self._ctrl.has_ml_entry(entry.item_kind, name):
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _on_name_edited(self, _text: str) -> None:
        self._name_edited = True

    def _on_role_changed(self, _index: int) -> None:
        # Only re-suggest while the name is still the auto-filled one; once the
        # user has typed their own, leave it alone (least surprise).
        if self._name_edited:
            return
        entry = self._role_combo.currentData()
        if entry is not None:
            self._name_edit.setText(self._suggest_name(entry))

    def _on_create(self) -> None:
        entry = self._role_combo.currentData()
        if entry is None:
            return
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid name", "Entry name must not be empty.")
            return
        try:
            self._ctrl.create_from_role(entry.item_kind, entry.role_id, name)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the user
            QMessageBox.critical(self, "Create failed", str(exc))
            return
        self.created = (entry.item_kind, name)
        self.accept()


class InspectDialog(InspectDialogBase):
    """Measure inspect dialog: adds the ml create/modify (CfgEditor) path.

    The base owns the md tab + ml view/rename/delete; this subclass injects the
    Create / Modify buttons and their CfgEditor-backed dialogs through the base's
    two template-method hooks.
    """

    def __init__(
        self,
        ctrl: Controller,
        bus: BaseEventBus,
        parent: QWidget | None = None,
    ) -> None:
        # The base only needs the shared context facet. This subclass keeps the
        # concrete app controller for measure-only CfgEditor and role-catalog
        # commands that deliberately stay outside session core.
        self._app_ctrl = ctrl
        super().__init__(ctrl.context_control, bus, parent=parent)

    def _build_extra_toolbar_buttons(self, toolbar: QHBoxLayout) -> None:
        self._arb_waveform_btn = QPushButton("Arb Waveforms…")
        toolbar.addWidget(self._arb_waveform_btn)
        self._arb_waveform_btn.clicked.connect(self._on_arb_waveform_clicked)

    def _build_extra_ml_buttons(self, btn_layout: QHBoxLayout) -> None:
        self._create_btn = QPushButton("Create...")
        self._modify_ml_btn = QPushButton("Modify...")
        self._modify_ml_btn.setEnabled(False)
        btn_layout.addWidget(self._create_btn)
        btn_layout.addWidget(self._modify_ml_btn)
        self._create_btn.clicked.connect(self._on_create_clicked)
        self._modify_ml_btn.clicked.connect(self._on_modify_ml_clicked)

    def _on_arb_waveform_clicked(self) -> None:
        # InspectDialog is always parented to MainWindow, which provides open_dialog.
        # The local fallback was dead code that could create a registry-invisible
        # dialog; removed for Fast-Fail clarity (ADR-0002).
        opener = getattr(self.parent(), "open_dialog", None)
        if not callable(opener):
            raise RuntimeError(
                "InspectDialog must be parented to a window providing open_dialog()"
            )
        opener(DialogName.ARB_WAVEFORM)

    def _on_ml_selection_changed(self, enabled: bool) -> None:
        self._modify_ml_btn.setEnabled(enabled)

    def _on_create_clicked(self) -> None:
        dlg = _MlCreateDialog(self._app_ctrl, parent=self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # On a successful create, chain straight into Modify so the user can
        # immediately tweak the freshly-seeded entry. Open Modify only after the
        # create dialog has closed (non-modal, no stacked modals).
        dlg.finished.connect(lambda _: self._after_create(dlg))
        dlg.open()

    def _after_create(self, dlg: _MlCreateDialog) -> None:
        created = dlg.created
        if created is None:
            return
        item_kind, name = created
        group = "modules" if item_kind == "module" else "waveforms"
        self._open_ml_modify(group, name)

    def _on_modify_ml_clicked(self) -> None:
        data = self._current_ml_item_data()
        if data is None:
            return
        group, name = data
        self._open_ml_modify(group, name)

    def _open_ml_modify(self, group: str, name: str) -> None:
        # Shared by selection-driven Modify and the auto-open after Create. Re-read
        # the live ml so a just-created entry's cfg is present (create -> ML_CHANGED
        # already refreshed the store).
        ml = self._app_ctrl.get_current_ml()
        if ml is None:
            return

        if group == "modules":
            dlg = _MlModifyDialog(
                self._app_ctrl, "module", name=name, cfg=ml.modules[name], parent=self
            )
        else:
            dlg = _MlModifyDialog(
                self._app_ctrl,
                "waveform",
                name=name,
                cfg=ml.waveforms[name],
                parent=self,
            )
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.finished.connect(lambda _: dlg.clear())
        dlg.open()
