"""Dense cfg tree whose only model is a published editing observation."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from qtpy.QtCore import Qt, QTimer, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
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
    CfgStatus,
)
from zcu_tools.gui.expected_error import ExpectedError
from zcu_tools.gui.widgets.dialog_presenter import DialogPresenter, QtDialogPresenter

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

    Text edits stay local until submit_pending(). The first edit captures its
    publication ref; later publications never replace pending controls. Stale
    submission keeps the input until explicit discard or confirmed reapplication.
    Selectors first submit pending input, then change the published selection.
    Detach discards only view state and never revokes the caller-owned handle.
    """

    validity_changed: Signal = Signal(bool)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        text_input_enhancer: TextInputEnhancer | None = None,
        dialog_presenter: DialogPresenter | None = None,
    ) -> None:
        super().__init__(parent)
        self._dialogs = dialog_presenter or QtDialogPresenter()
        self._pending: dict[CfgPath, DirectValue | EvalValue] = {}
        self._pending_base: CfgRef | None = None
        self._editor: CfgEditing | None = None
        self._observation: CfgObservation | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._generation = 0
        self._editing_enabled = True
        self._rebuild_pending = False
        self._text_input_enhancer = text_input_enhancer
        self._rows: dict[CfgPath, QTreeWidgetItem] = {}
        self._inputs: dict[CfgPath, InputWidget] = {}
        self._rendered_nodes: dict[CfgPath, CfgNodeObservation] = {}
        self._signature: tuple[tuple[CfgPath, CfgPath | None, object], ...] = ()
        self._expanded: dict[CfgPath, bool] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tree, self._branch_style = make_dense_cfg_tree()
        layout.addWidget(self._tree)
        self._pending_bar = QWidget(self)
        pending_layout = QHBoxLayout(self._pending_bar)
        pending_layout.setContentsMargins(0, 0, 0, 0)
        self._pending_message = QLabel()
        self._pending_message.setWordWrap(True)
        pending_layout.addWidget(self._pending_message, stretch=1)
        discard = QPushButton("Discard local input")
        discard.setObjectName("cfgDiscardPending")
        discard.clicked.connect(self.discard_pending)
        pending_layout.addWidget(discard)
        reapply = QPushButton("Reapply local input…")
        reapply.setObjectName("cfgReapplyPending")
        reapply.clicked.connect(self._confirm_reapply)
        pending_layout.addWidget(reapply)
        self._pending_bar.hide()
        layout.addWidget(self._pending_bar)
        self.setFont(self._tree.font())
        self._tree.itemExpanded.connect(
            lambda item: self._remember_expanded(item, True)
        )
        self._tree.itemCollapsed.connect(
            lambda item: self._remember_expanded(item, False)
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
        self._pending.clear()
        self._pending_base = None
        self._pending_bar.hide()
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
        self._pending_bar.setEnabled(enabled)

    def has_pending(self) -> bool:
        """Whether this view has input not yet submitted to its owner."""
        return bool(self._pending)

    def submit_pending(self) -> CfgRef:
        """Submit once against the first input\'s ref; retain local input on failure."""
        base = self._pending_base
        if base is None:
            return self.current_ref()
        return self._submit_at(base)

    def _submit_at(self, base: CfgRef) -> CfgRef:
        editor = self._editor
        if editor is None:
            raise RuntimeError("Config form is not attached")
        edits = tuple(CfgEdit(path, value) for path, value in self._pending.items())
        published = editor.edit(base.revision, edits)
        self._pending.clear()
        self._pending_base = None
        self._show_publication(published)
        return published.ref

    def discard_pending(self) -> None:
        """Use the latest delivered publication, without reading or editing the owner."""
        self._pending.clear()
        self._pending_base = None
        if self._observation is not None:
            self._show_publication(self._observation)
        else:
            self._pending_bar.hide()

    def _update_pending_message(self) -> None:
        self._pending_bar.setVisible(self.has_pending())
        conflict = self.has_pending() and self._pending_base != self.current_ref()
        self._pending_message.setText(
            "Config changed externally. Local input is preserved; discard or reapply."
            if conflict
            else "Local input will be submitted before Run."
        )

    def _confirm_reapply(self) -> None:
        observation = self._observation
        if observation is None or not self.has_pending():
            return
        shown_ref = observation.ref
        shown_edits = tuple(self._pending.items())
        generation = self._generation
        differences = []
        for path, value in shown_edits:
            node = observation.tree
            for key in path:
                child = node.children.get(key)
                if child is None:
                    break
                node = child
            differences.append(
                f"{'.'.join(path)}: current {node.value!r} → local {value!r}"
            )

        def apply(confirmed: bool) -> None:  # noqa: FBT001 - dialog callback
            if not confirmed or generation != self._generation:
                return
            if tuple(self._pending.items()) != shown_edits:
                self._pending_message.setText(
                    "Local input changed. Review the differences again."
                )
                return
            try:
                self._submit_at(shown_ref)
            except ExpectedError as exc:
                self._pending_message.setText(str(exc))

        self._dialogs.confirm_async(
            self,
            "Reapply local input",
            f"Apply to config revision {shown_ref.revision}?\n\n"
            + "\n".join(differences),
            on_decision=apply,
        )

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
        self._update_pending_message()
        if self.has_pending():
            self.validity_changed.emit(self.is_valid())
            return
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
        if not self.has_pending():
            self._rebuild()
        self._tree.setEnabled(self._editing_enabled)

    def _shape_signature(
        self, observation: CfgObservation
    ) -> tuple[tuple[CfgPath, CfgPath | None, object], ...]:
        return tuple(
            (path, parent, node.spec)
            for path, parent, node in _visible_rows(observation.tree)
        )

    def _remember_expanded(self, item: QTreeWidgetItem, expanded: bool) -> None:  # noqa: FBT001 - Qt signal
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(path, tuple):
            self._expanded[path] = expanded

    def _clear_tree(self) -> None:
        for path, widget in self._inputs.items():
            self._tree.removeItemWidget(self._rows[path], 1)
            widget.setParent(None)
            widget.deleteLater()
        self._inputs.clear()
        self._rendered_nodes.clear()
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
            self._rendered_nodes[path] = node
            immediate = (
                node.options is not None or spec.type is bool or self._is_selector(path)
            )
            return ScalarInputWidget(
                spec,
                value,
                options=node.options,
                submit=lambda selected: self._edit(
                    path,
                    selected,
                    generation,
                    immediate=immediate,
                ),
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

    def _is_selector(self, path: CfgPath) -> bool:
        observation = self._observation
        if observation is None:
            return False
        node = observation.tree
        for key in path[:-1]:
            node = node.children[key]
        return isinstance(node.spec, ChoiceSectionSpec) and any(
            binding.selector_key == path[-1] for binding in node.spec.bindings
        )

    def _edit(
        self,
        path: CfgPath,
        value: DirectValue | EvalValue,
        generation: int,
        *,
        immediate: bool = False,
    ) -> None:
        editor, observation = self._editor, self._observation
        if (
            generation != self._generation
            or self._rebuild_pending
            or editor is None
            or observation is None
        ):
            return
        if immediate:
            try:
                ref = self.submit_pending()
                editor.edit(ref.revision, (CfgEdit(path, value),))
            except ExpectedError as exc:
                self._pending_message.setText(str(exc))
                self._pending_bar.show()
                QTimer.singleShot(0, lambda: self._restore_selection(path, generation))
            return
        if self._pending_base is None:
            self._pending_base = observation.ref
        # The owner applies edits in order, so the last sampling operation must win.
        self._pending.pop(path, None)
        self._pending[path] = value
        widget = self._inputs.get(path)
        if isinstance(widget, ScalarInputWidget):
            widget.display(value, options=self._rendered_nodes[path].options)
        else:
            range_widget = self._inputs.get(path[:-1])
            key = path[-1]
            if isinstance(range_widget, SweepInputWidget) and key in ("start", "stop"):
                range_widget.display_edge(key, value)
            elif isinstance(range_widget, CenteredSweepInputWidget) and key == "center":
                range_widget.display_center(value)
        self._update_pending_message()
        self.validity_changed.emit(self.is_valid())

    def _select_reference(
        self, path: CfgPath, selected: ReferenceSelection, generation: int
    ) -> None:
        editor, observation = self._editor, self._observation
        if (
            generation != self._generation
            or self._rebuild_pending
            or editor is None
            or observation is None
        ):
            return
        try:
            ref = self.submit_pending()
            if isinstance(selected, CustomReferenceSelection):
                editor.select_custom_reference(ref.revision, path, selected.label)
            else:
                editor.edit(ref.revision, (CfgEdit((*path, "__ref"), selected),))
        except ExpectedError as exc:
            self._pending_message.setText(str(exc))
            self._pending_bar.show()
            QTimer.singleShot(0, lambda: self._restore_selection(path, generation))

    def _restore_selection(self, path: CfgPath, generation: int) -> None:
        observation = self._observation
        if observation is None or generation != self._generation:
            return
        node = observation.tree
        for key in path:
            child = node.children.get(key)
            if child is None:
                return
            node = child
        widget = self._inputs.get(path)
        if isinstance(widget, ScalarInputWidget) and isinstance(
            node.value, (DirectValue, EvalValue)
        ):
            widget.display(node.value, options=node.options)
        elif isinstance(widget, ReferenceInputWidget) and isinstance(
            node.value, (ReferenceValue, type(None))
        ):
            widget.display(
                node.value, library_keys=reference_library_keys(node), valid=node.valid
            )

    def _refresh_inputs(self, published: CfgObservation) -> None:
        for path, _, node in _visible_rows(published.tree):
            self._color_item(self._rows[path], node)
            widget = self._inputs.get(path)
            if isinstance(widget, ScalarInputWidget):
                self._rendered_nodes[path] = node
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
