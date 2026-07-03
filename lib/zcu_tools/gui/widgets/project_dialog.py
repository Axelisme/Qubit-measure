"""ProjectDialog — set the project's chip / qubit names (and optional paths).

Shared by fluxdep-gui and dispersive-gui: the project shape is identical, so the
dialog is too. The chip / qubit names locate the default output layout
(``result_dir`` → ``result/<chip>/<qubit>`` for processed outputs,
``database_path`` → ``Database/<chip>/<qubit>`` for raw measurement data). Both
auto-derive from the names and track them as the user types, until the user edits
/ browses a field — a manual change detaches that field from the derivation. Both
have a Browse button (choose a folder). Returns a ``ProjectInfo`` (or None on
cancel).

The single per-app difference is the database field's label (``db_label``):
fluxdep calls it "Database path" (the raw spectrum root), dispersive "One-tone
dir" (its raw one-tone root). The app passes its label; everything else is shared.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from zcu_tools.gui.project import (
    ProjectInfo,
    default_database_root,
    default_result_dir,
)
from zcu_tools.gui.result_scope import ResultScope, ResultScopeManager

logger = logging.getLogger(__name__)

# fluxdep's label; the database field's default form-row label. dispersive passes
# its own ("One-tone dir") explicitly.
DEFAULT_DB_LABEL = "Database path"


class ProjectDialog(QDialog):
    """Edit the project chip/qubit names (+ result/database roots, with Browse)."""

    def __init__(
        self,
        project: ProjectInfo,
        parent: QWidget | None = None,
        *,
        db_label: str = DEFAULT_DB_LABEL,
        project_root: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project")
        self.resize(560, 220)

        self._db_label = db_label
        self._result_scopes: tuple[ResultScope, ...] = ()
        self._syncing_scope_selection = False

        # Anchor derived defaults at the same repo root the project was built
        # with, so the auto-derivation matches the project's actual paths (else a
        # root-anchored project.result_dir would be mis-detected as overridden).
        self._root = project_root if project_root is not None else project.root_dir

        # Each path field tracks chip/qubit until the user edits / browses it.
        # It counts as "overridden" if it already differs from what the names
        # would derive (a previously-customised project).
        derived_result = default_result_dir(
            project.chip_name, project.qub_name, self._root
        )
        derived_db = default_database_root(
            project.chip_name, project.qub_name, self._root
        )
        self._result_overridden = bool(
            project.result_dir and project.result_dir != derived_result
        )
        self._database_overridden = bool(
            project.database_path and project.database_path != derived_db
        )

        form = QFormLayout(self)

        scope_row = QHBoxLayout()
        self._scope_combo = QComboBox()
        self._scope_combo.setMinimumWidth(260)
        self._scope_combo.currentIndexChanged.connect(self._on_scope_selected)
        scope_row.addWidget(self._scope_combo, stretch=1)
        refresh_scopes_btn = QPushButton("↻")
        refresh_scopes_btn.setFixedWidth(28)
        refresh_scopes_btn.setToolTip("Refresh result scopes")
        refresh_scopes_btn.clicked.connect(self._refresh_result_scopes)
        scope_row.addWidget(refresh_scopes_btn)
        form.addRow("Result scope", scope_row)

        self._chip_edit = QLineEdit(project.chip_name)
        self._qub_edit = QLineEdit(project.qub_name)
        self._chip_edit.textChanged.connect(self._on_names_changed)
        self._qub_edit.textChanged.connect(self._on_names_changed)
        form.addRow("Chip name", self._chip_edit)
        form.addRow("Qubit name", self._qub_edit)

        self._result_edit = QLineEdit(project.result_dir or derived_result)
        self._result_edit.textEdited.connect(self._on_result_edited)
        form.addRow("Result dir", self._dir_row(self._result_edit, self._browse_result))

        self._database_edit = QLineEdit(project.database_path or derived_db)
        self._database_edit.textEdited.connect(self._on_database_edited)
        form.addRow(
            self._db_label, self._dir_row(self._database_edit, self._browse_database)
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self._refresh_result_scopes(silent=True)
        self._on_names_changed("")

    def _dir_row(self, edit: QLineEdit, on_browse: Callable[[], None]) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(on_browse)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    # --- auto-derivation -------------------------------------------------

    def _on_names_changed(self, _text: str) -> None:
        if self._syncing_scope_selection:
            return
        chip = self._chip_edit.text().strip()
        qub = self._qub_edit.text().strip()
        if not self._result_overridden:
            self._result_edit.setText(default_result_dir(chip, qub, self._root))
        if not self._database_overridden:
            self._database_edit.setText(default_database_root(chip, qub, self._root))
        self._update_scope_options(chip, qub)

    def _refresh_result_scopes(
        self, _checked: bool = False, *, silent: bool = False
    ) -> None:
        try:
            self._result_scopes = ResultScopeManager(self._root or ".").list_scopes()
        except Exception as exc:  # noqa: BLE001 - scope discovery is best-effort UI.
            logger.warning("ProjectDialog: failed to list result scopes: %s", exc)
            self._result_scopes = ()
            if not silent:
                self._scope_combo.setToolTip(f"Failed to refresh result scopes: {exc}")
        else:
            self._scope_combo.setToolTip("")
        self._update_scope_options(
            self._chip_edit.text().strip(), self._qub_edit.text().strip()
        )

    def _update_scope_options(self, chip: str, qub: str) -> None:
        self._scope_combo.blockSignals(True)
        prev_scope_id = self._scope_combo.currentData()
        current_result = self._result_edit.text().strip()
        self._scope_combo.clear()

        generated_index = -1
        has_names = bool(chip and qub)
        if has_names:
            self._scope_combo.addItem("(new generated scope)", userData=None)
            generated_index = 0

        for scope in self._result_scopes:
            self._scope_combo.addItem(
                f"{scope.chip_name}/{scope.qub_name}",
                userData=scope.scope_id,
            )

        idx = -1
        if prev_scope_id:
            idx = self._scope_combo.findData(prev_scope_id)
        if idx < 0 and current_result:
            for scope in self._result_scopes:
                if (
                    scope.result_dir == current_result
                    or scope.scope_id == current_result
                ):
                    idx = self._scope_combo.findData(scope.scope_id)
                    break
        if idx < 0 and has_names:
            for scope in self._result_scopes:
                if (scope.chip_name, scope.qub_name) == (chip, qub):
                    idx = self._scope_combo.findData(scope.scope_id)
                    break

        if idx >= 0:
            self._scope_combo.setCurrentIndex(idx)
        elif has_names:
            self._scope_combo.setCurrentIndex(generated_index)
        elif self._result_scopes:
            self._scope_combo.setCurrentIndex(0)
        else:
            self._scope_combo.addItem("(no result scopes found)", userData=None)

        self._scope_combo.blockSignals(False)

    def _current_scope(self) -> ResultScope | None:
        scope_id = self._scope_combo.currentData()
        if not scope_id:
            return None
        for scope in self._result_scopes:
            if scope.scope_id == scope_id:
                return scope
        return None

    def _on_scope_selected(self, _index: int) -> None:
        scope = self._current_scope()
        if scope is None:
            return

        self._syncing_scope_selection = True
        try:
            self._chip_edit.setText(scope.chip_name)
            self._qub_edit.setText(scope.qub_name)
            self._result_edit.setText(scope.result_dir)
            if not self._database_overridden:
                self._database_edit.setText(
                    default_database_root(scope.chip_name, scope.qub_name, self._root)
                )
        finally:
            self._syncing_scope_selection = False

        self._result_overridden = scope.result_dir != default_result_dir(
            scope.chip_name, scope.qub_name, self._root
        )
        self._update_scope_options(scope.chip_name, scope.qub_name)

    def _on_result_edited(self, _text: str) -> None:
        # a manual edit detaches the result dir from the chip/qubit derivation
        self._result_overridden = True

    def _on_database_edited(self, _text: str) -> None:
        # a manual edit detaches the database path from the chip/qubit derivation
        self._database_overridden = True

    # --- browse ----------------------------------------------------------

    def _browse_result(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select result dir", self._result_edit.text()
        )
        if path:
            self._result_overridden = True
            self._result_edit.setText(path)

    def _browse_database(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, f"Select {self._db_label.lower()}", self._database_edit.text()
        )
        if path:
            self._database_overridden = True
            self._database_edit.setText(path)

    # --- result ----------------------------------------------------------

    def result_project(self) -> ProjectInfo:
        """The edited project info (names trimmed, paths always concrete).

        An empty path field is left empty so ``ProjectInfo.__post_init__`` derives
        it; a non-empty field is the user's override.
        """
        return ProjectInfo(
            chip_name=self._chip_edit.text().strip(),
            qub_name=self._qub_edit.text().strip(),
            result_dir=self._result_edit.text().strip(),
            database_path=self._database_edit.text().strip(),
            root_dir=self._root,
        )
