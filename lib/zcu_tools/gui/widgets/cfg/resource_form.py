"""Dense cfg tree whose only model is a published editing observation."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from qtpy.QtCore import Qt, QTimer, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QLabel,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSectionSpec,
    ChoiceSectionSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    is_custom_reference_key,
)
from zcu_tools.gui.cfg.binding.observation import CfgNodeObservation
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgEditing,
    CfgObservation,
    CfgPath,
    CfgRef,
    CfgRevision,
    CfgStatus,
)
from zcu_tools.gui.expected_error import ExpectedError

from .fields import (
    CustomReferenceSelection,
    ReferenceInputWidget,
    reference_library_keys,
)
from .fields.common import (
    CenteredSweepInputWidget,
    ScalarInputWidget,
    SweepInputWidget,
)
from .fields.reference_shared import ReferenceSelection
from .presentation import active_choice_keys
from .registry import TextInputEnhancer
from .structure import make_dense_cfg_tree

InputWidget = (
    ScalarInputWidget
    | SweepInputWidget
    | CenteredSweepInputWidget
    | ReferenceInputWidget
)
Row = tuple[CfgPath, CfgPath | None, CfgNodeObservation]


def _visible_rows(
    section: CfgNodeObservation,
    path: CfgPath = (),
    parent_row: CfgPath | None = None,
) -> Iterator[Row]:
    """Project active children; an unadorned singleton reference wrapper has no row."""
    allowed = (
        active_choice_keys(
            section.spec,
            lambda key: (
                section.children[key].value if key in section.children else None
            ),
        )
        if isinstance(section.spec, ChoiceSectionSpec)
        else None
    )
    visible = [
        (key, child)
        for key, child in section.children.items()
        if not isinstance(child.spec, LiteralSpec)
        and (allowed is None or key in allowed)
    ]
    for key, child in visible:
        child_path = (*path, key)
        if (
            isinstance(section.spec, ReferenceSpec)
            and len(visible) == 1
            and isinstance(child.spec, CfgSectionSpec)
        ):
            yield from _visible_rows(child, child_path, parent_row)
            continue
        yield child_path, parent_row, child
        if child.children:
            yield from _visible_rows(child, child_path, child_path)


class ResourceCfgFormWidget(QWidget):
    """Render one caller-owned CfgEditing handle without a writable draft copy.

    Every input change (each keystroke or selection) is submitted at once
    against the owner's latest revision, so the displayed values are always
    the published ones and the last input wins. A rejected input shows its
    error below the tree and the controls return to the published values. Detach discards only view state and never revokes the
    caller-owned handle.
    """

    validity_changed: Signal = Signal(bool)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        text_input_enhancer: TextInputEnhancer | None = None,
    ) -> None:
        super().__init__(parent)
        self._editor: CfgEditing | None = None
        self._observation: CfgObservation | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._generation = 0
        self._editing_enabled = True
        self._rebuild_pending = False
        self._text_input_enhancer = text_input_enhancer
        self._rows: dict[CfgPath, QTreeWidgetItem] = {}
        self._inputs: dict[CfgPath, InputWidget] = {}
        self._signature: tuple[tuple[CfgPath, CfgPath | None, object], ...] = ()
        self._expanded: dict[CfgPath, bool] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tree, self._branch_style = make_dense_cfg_tree()
        layout.addWidget(self._tree)
        self._error = QLabel()
        self._error.setObjectName("cfgSubmitError")
        self._error.setWordWrap(True)
        self._error.hide()
        layout.addWidget(self._error)
        self.setFont(self._tree.font())
        self._tree.itemExpanded.connect(
            lambda item: self._remember_expanded(item, expanded=True)
        )
        self._tree.itemCollapsed.connect(
            lambda item: self._remember_expanded(item, expanded=False)
        )

    def attach(self, editor: CfgEditing) -> None:
        """Observe a caller-owned handle; detach never revokes that handle."""
        self.detach()
        self._generation += 1
        self._editor = editor
        try:
            self._show_publication(editor.observe())
            self._unsubscribe = editor.watch(self._show_publication)
        except Exception:
            self.detach()
            raise

    def detach(self) -> None:
        """Stop watching and discard this view without modifying the owner."""
        self._generation += 1
        unsubscribe = self._unsubscribe
        self._unsubscribe = None
        if unsubscribe is not None:
            unsubscribe()
        self._editor = None
        self._observation = None
        self._error.hide()
        self._rebuild_pending = False
        self._signature = ()
        self._expanded.clear()
        self._tree.blockSignals(True)
        try:
            self._clear_tree()
        finally:
            self._tree.blockSignals(False)
        self._tree.setEnabled(self._editing_enabled)

    def set_editing_enabled(self, enabled: bool) -> None:  # noqa: FBT001 - form API
        self._editing_enabled = enabled
        self._tree.setEnabled(enabled and not self._rebuild_pending)

    def current_ref(self) -> CfgRef:
        """Return the displayed publication's ref without refreshing the owner."""
        if self._observation is None:
            raise RuntimeError("Config form is not attached")
        return self._observation.ref

    def is_valid(self) -> bool:
        observation = self._observation
        return observation is not None and observation.status is CfgStatus.VALID

    def first_invalid_reason(self) -> str | None:
        observation = self._observation
        if observation is None or observation.status is CfgStatus.VALID:
            return None
        return (
            observation.diagnostics[0].message
            if observation.diagnostics
            else "Config is not ready"
        )

    def _show_publication(self, published: CfgObservation) -> None:
        previous = self._observation
        self._observation = published
        changed_references: list[tuple[CfgPath, str]] = []
        if previous is not None:
            before = {
                path: node.value.chosen_key
                for path, _, node in _visible_rows(previous.tree)
                if isinstance(node.value, ReferenceValue)
            }
            for path, _, node in _visible_rows(published.tree):
                if (
                    isinstance(node.value, ReferenceValue)
                    and path in before
                    and before[path] != node.value.chosen_key
                ):
                    self._expanded.pop(path, None)
                    changed_references.append((path, node.value.chosen_key))
        signature = self._shape_signature(published)
        if signature != self._signature or self._rebuild_pending:
            # The first attach is outside any input callback. Subsequent shape
            # changes can be published synchronously inside a Qt change signal.
            if previous is None:
                self._rebuild()
            elif not self._rebuild_pending:
                self._rebuild_pending = True
                self._tree.setEnabled(False)
                generation = self._generation
                QTimer.singleShot(0, lambda: self._finish_rebuild(generation))
        else:
            self._refresh_inputs(published)
            for path, key in changed_references:
                self._rows[path].setExpanded(is_custom_reference_key(key))
        self.validity_changed.emit(self.is_valid())

    def _finish_rebuild(self, generation: int) -> None:
        if generation != self._generation or not self._rebuild_pending:
            return
        self._rebuild_pending = False
        self._rebuild()
        self._tree.setEnabled(self._editing_enabled)

    def _shape_signature(
        self, observation: CfgObservation
    ) -> tuple[tuple[CfgPath, CfgPath | None, object], ...]:
        return tuple(
            (path, parent, node.spec)
            for path, parent, node in _visible_rows(observation.tree)
        )

    def _remember_expanded(self, item: QTreeWidgetItem, *, expanded: bool) -> None:
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(path, tuple):
            self._expanded[path] = expanded

    def _clear_tree(self) -> None:
        for path, widget in self._inputs.items():
            self._tree.removeItemWidget(self._rows[path], 1)
            widget.setParent(None)
            widget.deleteLater()
        self._inputs.clear()
        self._rows.clear()
        self._tree.clear()

    def _rebuild(self) -> None:
        observation = self._observation
        if observation is None:
            return
        self._tree.blockSignals(True)
        try:
            self._clear_tree()
            for path, parent_path, node in _visible_rows(observation.tree):
                parent = self._rows.get(parent_path) if parent_path else None
                item = (
                    QTreeWidgetItem(parent) if parent else QTreeWidgetItem(self._tree)
                )
                item.setData(0, Qt.ItemDataRole.UserRole, path)
                item.setText(0, getattr(node.spec, "label", "") or path[-1])
                self._rows[path] = item
                control = self._make_input(path, node)
                if control is not None:
                    control.setObjectName("cfgInput:" + ".".join(path))
                    self._inputs[path] = control
                    self._tree.setItemWidget(item, 1, control)
                self._color_item(item, node)
                if node.children:
                    default = (
                        isinstance(node.value, ReferenceValue)
                        and is_custom_reference_key(node.value.chosen_key)
                    ) or not isinstance(node.spec, ReferenceSpec)
                    item.setExpanded(self._expanded.get(path, default))
            self._signature = self._shape_signature(observation)
        finally:
            self._tree.blockSignals(False)

    def _make_input(
        self, path: CfgPath, node: CfgNodeObservation
    ) -> InputWidget | None:
        spec, value = node.spec, node.value
        generation = self._generation
        if isinstance(spec, ScalarSpec):
            if not isinstance(value, (DirectValue, EvalValue)):
                raise TypeError(f"Scalar input at {path!r} has no scalar value")
            return ScalarInputWidget(
                spec,
                value,
                options=node.options,
                submit=lambda selected: self._edit(path, selected, generation),
                text_input_enhancer=self._text_input_enhancer,
            )
        if isinstance(spec, SweepSpec):
            if not isinstance(value, SweepValue):
                raise TypeError(f"Sweep input at {path!r} has no sweep value")
            return SweepInputWidget(
                spec,
                value,
                submit=lambda key, selected: self._edit(
                    (*path, key), selected, generation
                ),
                text_input_enhancer=self._text_input_enhancer,
            )
        if isinstance(spec, CenteredSweepSpec):
            if not isinstance(value, CenteredSweepValue):
                raise TypeError(f"Centered sweep at {path!r} has no centered value")
            return CenteredSweepInputWidget(
                spec,
                value,
                submit=lambda key, selected: self._edit(
                    (*path, key), selected, generation
                ),
                text_input_enhancer=self._text_input_enhancer,
            )
        if isinstance(spec, ReferenceSpec):
            if not isinstance(value, (ReferenceValue, type(None))):
                raise TypeError(f"Reference at {path!r} has no reference value")
            return ReferenceInputWidget(
                spec,
                value,
                library_keys=reference_library_keys(node),
                valid=node.valid,
                submit=lambda selected: self._select_reference(
                    path, selected, generation
                ),
            )
        if isinstance(spec, CfgSectionSpec):
            return None
        raise TypeError(f"No cfg input renderer for {type(spec).__name__} at {path!r}")

    def _edit(
        self, path: CfgPath, value: DirectValue | EvalValue, generation: int
    ) -> None:
        self._submit(
            path,
            generation,
            lambda editor, revision: editor.edit(revision, (CfgEdit(path, value),)),
        )

    def _select_reference(
        self, path: CfgPath, selected: ReferenceSelection, generation: int
    ) -> None:
        if isinstance(selected, CustomReferenceSelection):
            self._submit(
                path,
                generation,
                lambda editor, revision: editor.select_custom_reference(
                    revision, path, selected.label
                ),
            )
        else:
            self._submit(
                path,
                generation,
                lambda editor, revision: editor.edit(
                    revision, (CfgEdit((*path, "__ref"), selected),)
                ),
            )

    def _submit(
        self,
        path: CfgPath,
        generation: int,
        apply: Callable[[CfgEditing, CfgRevision], object],
    ) -> None:
        """Apply one committed input to the owner's latest revision."""
        editor = self._editor
        if (
            generation != self._generation
            or self._rebuild_pending
            or editor is None
            or self._observation is None
        ):
            return
        try:
            # The latest revision, not the shown one: the last committed input wins.
            apply(editor, editor.observe().ref.revision)
        except ExpectedError as exc:
            self._error.setText(f"{'.'.join(path)}: {exc}")
            self._error.show()
            QTimer.singleShot(0, lambda: self._restore_inputs(generation))
            return
        self._error.hide()

    def _restore_inputs(self, generation: int) -> None:
        observation = self._observation
        if (
            observation is None
            or generation != self._generation
            or self._rebuild_pending
        ):
            return
        self._refresh_inputs(observation)

    def _refresh_inputs(self, published: CfgObservation) -> None:
        for path, _, node in _visible_rows(published.tree):
            self._color_item(self._rows[path], node)
            widget = self._inputs.get(path)
            if isinstance(widget, ScalarInputWidget):
                if not isinstance(node.value, (DirectValue, EvalValue)):
                    raise TypeError(f"Scalar input at {path!r} has no scalar value")
                widget.display(node.value, options=node.options)
            elif isinstance(widget, SweepInputWidget):
                if not isinstance(node.value, SweepValue):
                    raise TypeError(f"Sweep input at {path!r} has no sweep value")
                widget.display(node.value)
            elif isinstance(widget, CenteredSweepInputWidget):
                if not isinstance(node.value, CenteredSweepValue):
                    raise TypeError(f"Centered sweep at {path!r} has no centered value")
                widget.display(node.value)
            elif isinstance(widget, ReferenceInputWidget):
                if not isinstance(node.value, (ReferenceValue, type(None))):
                    raise TypeError(f"Reference at {path!r} has no reference value")
                widget.display(
                    node.value,
                    library_keys=reference_library_keys(node),
                    valid=node.valid,
                )

    @staticmethod
    def _color_item(item: QTreeWidgetItem, node: CfgNodeObservation) -> None:
        item.setForeground(0, QBrush() if node.valid else QBrush(QColor("red")))
