"""Shared structural presentation adapters for cfg Qt widgets.

Provides the explicit structural presentation seam (S1) and the dense tree adapter
(S2 view-only folding/depth/connectors/elision). The form adapter remains the
default and reuses the exact field-renderer registry; the tree adapter varies
only structural node composition.
"""

from __future__ import annotations

import logging
from typing import Protocol, cast, final

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QBrush, QColor, QPainter, QPen  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProxyStyle,
    QStyle,
    QStyleOption,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    ChoiceSectionSpec,
    DirectValue,
    LiteralSpec,
    ReferenceSpec,
)
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    CfgField,
    ReferenceField,
    ScalarField,
    SectionField,
    SweepField,
)
from zcu_tools.gui.cfg.reference_key import make_custom_reference_key

from .registry import FieldRenderContext, FieldWidgetProtocol

logger = logging.getLogger(__name__)

TREE_DEPTH_COLORS: tuple[str, ...] = (
    "#e2ebf6",
    "#e3f0e6",
    "#f4e9d2",
    "#eadff1",
    "#dceeee",
)

_INDENTATION_PX = 10
_TREE_FONT_SIZE_PX = 13


class _TreeBranchStyle(QProxyStyle):
    """Classic vertical/horizontal branch lines without triangles."""

    def drawPrimitive(  # type: ignore[override]
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption | None,
        painter: QPainter | None,
        widget: QWidget | None = None,
    ) -> None:
        if option is None or painter is None:
            super().drawPrimitive(element, option, painter, widget)  # type: ignore[arg-type]
            return
        if element != QStyle.PrimitiveElement.PE_IndicatorBranch:  # type: ignore[attr-defined]
            super().drawPrimitive(element, option, painter, widget)
            return
        state = option.state  # type: ignore[attr-defined]
        has_sibling = bool(state & QStyle.StateFlag.State_Sibling)  # type: ignore[attr-defined]
        has_item = bool(state & QStyle.StateFlag.State_Item)  # type: ignore[attr-defined]
        if not (has_sibling or has_item):
            return
        rect = option.rect  # type: ignore[attr-defined]
        x = rect.center().x()
        y = rect.center().y()
        painter.save()
        painter.setPen(QPen(QColor("#b8c1cc"), 1))
        if has_sibling:
            painter.drawLine(x, rect.top(), x, rect.bottom())
        else:
            painter.drawLine(x, rect.top(), x, y)
        if has_item:
            painter.drawLine(x, y, rect.right(), y)
        painter.restore()


def _choice_visible_keys(field: SectionField) -> set[str] | None:
    spec = field.spec
    if not isinstance(spec, ChoiceSectionSpec):
        return None
    visible = set(spec.fields)
    for binding in spec.bindings:
        selector = field.fields.get(binding.selector_key)
        value = selector.get_value() if selector is not None else None
        choice = str(value.value) if isinstance(value, DirectValue) else ""
        try:
            active_spec = binding.choices[choice]
        except KeyError as exc:
            expected = ", ".join(sorted(binding.choices))
            raise ValueError(
                f"ChoiceSectionSpec selector {binding.selector_key!r} has unknown "
                f"value {choice!r}; expected one of: {expected}"
            ) from exc
        active = set(active_spec.fields)
        visible -= binding.controlled_field_keys() - active
    return visible


def _decorated_label(label: str, decoration: object | None) -> str:
    # decoration is FieldDecorationProtocol | None; avoid tight import.
    if decoration is None:
        return label
    badge = getattr(decoration, "badge", "") or ""
    suffix = getattr(decoration, "label_suffix", "") or ""
    text = f"{label}{suffix}"
    if badge:
        return f"{text} [{badge}]"
    return text


def _is_hidden(field: CfgField, path: str, context: FieldRenderContext) -> bool:
    decoration = None
    resolver = context.decoration_for_path
    if resolver is not None:
        try:
            decoration = resolver(path, field)
        except Exception:
            logger.debug("decoration resolver failed for %r", path, exc_info=True)
            decoration = None
    if isinstance(field.spec, LiteralSpec):
        # Literal rows are hidden by default unless decoration explicitly unhides.
        if decoration is None:
            return True
        return bool(getattr(decoration, "hidden", False))
    if decoration is not None:
        return bool(getattr(decoration, "hidden", False))
    return False


class _TreeReferenceHeader(QWidget):
    """Combo header for a reference row in the tree."""

    _NONE_KEY = "<None>"

    def __init__(self, field: ReferenceField, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._field = field
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(4)

        self._combo = QComboBox()
        self._refresh_combo_items()
        self._combo.setMinimumWidth(20)
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        layout.addWidget(self._combo, stretch=1)

        field.on_change.connect(self._on_model_changed)
        if field.spec.optional:
            field.on_enabled_changed.connect(self._on_enabled_changed)  # type: ignore[attr-defined]

    def _refresh_combo_items(self) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()
        field = self._field
        current = field.get_chosen_key()
        if field.spec.optional:
            self._combo.addItem("None", self._NONE_KEY)
            self._combo.insertSeparator(self._combo.count())
        for spec in field.spec.allowed:
            label = spec.label or "Custom"
            key = make_custom_reference_key(label)
            self._combo.addItem(label, key)
        compatible = field.available_keys()
        if compatible:
            self._combo.insertSeparator(self._combo.count())
            for name in compatible:
                if name == current and field.is_modified():  # type: ignore[attr-defined]
                    self._combo.addItem(f"Lib: {name} (modified)", name)
                    self._combo.addItem(f"Revert to Lib: {name}", name)
                else:
                    self._combo.addItem(f"Lib: {name}", name)
        if field.spec.optional and not field.is_enabled:  # type: ignore[attr-defined]
            self._combo.setCurrentIndex(0)
        else:
            idx = self._combo.findData(current)
            if idx < 0 and field.has_missing_library_ref():  # type: ignore[attr-defined]
                self._combo.addItem(f"Missing: {current}", current)
                idx = self._combo.findData(current)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)

    def _on_combo_changed(self, index: int) -> None:
        key = self._combo.itemData(index)
        field = self._field
        if key == self._NONE_KEY:
            field.set_enabled(False)  # type: ignore[attr-defined]
            return
        if field.spec.optional and not field.is_enabled:  # type: ignore[attr-defined]
            field.set_enabled(True)  # type: ignore[attr-defined]
        field.set_chosen_key(key)

    def _on_model_changed(self, *_: object) -> None:
        self._refresh_combo_items()

    def _on_enabled_changed(self, *_: object) -> None:
        self._refresh_combo_items()

    def teardown(self) -> None:
        field = self._field
        try:
            field.on_change.disconnect(self._on_model_changed)
        except Exception:
            pass
        if field.spec.optional:
            try:
                field.on_enabled_changed.disconnect(self._on_enabled_changed)  # type: ignore[attr-defined]
            except Exception:
                pass


@final
class TreeCfgWidget(QWidget):
    """Dense QTreeWidget presentation for a ``CfgDraft`` section.

    Implements :class:`FieldWidgetProtocol` so it can be used as the root widget
    of :class:`CfgFormWidget`. Reference shape-row elision, depth coloring,
    indentation, connectors, and whole-row folding are view-only.
    """

    def __init__(
        self,
        field: SectionField,
        context: FieldRenderContext,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._field: SectionField = field
        self._context: FieldRenderContext = context
        self._path: str = context.path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tree = QTreeWidget()
        layout.addWidget(self._tree)

        self._tree.setObjectName("cfgTree")
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(2)
        self._tree.setRootIsDecorated(False)
        self._tree.setIndentation(_INDENTATION_PX)
        self._tree.setAlternatingRowColors(False)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)  # type: ignore[attr-defined]
        self._tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # type: ignore[attr-defined]
        # 13 px field text per spec; use pixel-size font to avoid masking the branch proxy.
        font = self._tree.font()
        font.setPixelSize(_TREE_FONT_SIZE_PX)
        self._tree.setFont(font)
        self.setFont(font)
        self._branch_style = _TreeBranchStyle()
        self._branch_style.setParent(self._tree)
        self._tree.setStyle(self._branch_style)
        header = self._tree.header()
        assert header is not None
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # type: ignore[attr-defined]
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # type: ignore[attr-defined]

        self._path_to_item: dict[str, QTreeWidgetItem] = {}
        self._item_depth: dict[int, int] = {}  # id(item) -> depth
        self._leaf_widgets: list[FieldWidgetProtocol] = []
        self._ref_headers: list[_TreeReferenceHeader] = []
        self._ref_connections: list[tuple[ReferenceField, object]] = []
        self._expanded_state: dict[str, bool] = {}

        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemExpanded.connect(
            lambda item: self._remember_expanded(item, True)
        )
        self._tree.itemCollapsed.connect(
            lambda item: self._remember_expanded(item, False)
        )

        field.on_validity_changed.connect(self._on_validity_changed)

        self._rebuild_tree()

    @property
    def field(self) -> CfgField:
        return self._field

    def teardown(self) -> None:
        try:
            self._field.on_validity_changed.disconnect(self._on_validity_changed)
        except Exception:
            pass
        self._disconnect_refs()
        for header in self._ref_headers:
            header.teardown()
        self._ref_headers.clear()
        for widget in self._leaf_widgets:
            try:
                widget.teardown()
            except Exception:
                pass
        self._leaf_widgets.clear()
        self._tree.clear()

    def refresh_section(self, path: str) -> bool:
        # Normalize dotted path handling similar to SectionWidget.
        if path == self._path:
            self._rebuild_tree()
            return True
        prefix = f"{self._path}." if self._path else ""
        if prefix and not path.startswith(prefix):
            return False
        # Accept any descendant section/reference path that currently has an item.
        # For simplicity, any known path triggers a full rebuild and returns True
        # to satisfy CfgFormWidget's section-local refresh contract.
        if path in self._path_to_item:
            self._rebuild_tree()
            return True
        # Also accept paths that are ancestors of current items (e.g., parent section
        # whose children include hidden fields). Walk visible keys to decide.
        remainder = path.removeprefix(prefix)
        # Quick existence check: traverse field tree to see if path is a SectionField.
        sec = self._find_section_field(path)
        if sec is not None:
            self._rebuild_tree()
            return True
        return False

    def _find_section_field(self, path: str) -> SectionField | None:
        if path == self._path or path == "":
            return self._field
        # path is like "modules.readout" where readout is ReferenceField with sub_field
        # We only consider true SectionField targets.
        parts = path.split(".") if path else []
        # Determine root traversal start: if self._path is not empty, we need to strip prefix
        # But we usually call with full dotted path matching root path prefix.
        cur: CfgField = self._field
        # If path starts with self._path prefix, strip it.
        remaining = path
        if self._path:
            if not path.startswith(self._path):
                return None
            if path == self._path:
                return cast(SectionField, cur)
            remaining = path.removeprefix(self._path + ".")
            parts = remaining.split(".") if remaining else []
        for part in parts:
            if isinstance(cur, SectionField):
                nxt = cur.fields.get(part)
                if nxt is None:
                    return None
                cur = nxt
            elif isinstance(cur, ReferenceField):
                sub = cur.sub_field
                if sub is None:
                    return None
                nxt = sub.fields.get(part)
                if nxt is None:
                    return None
                cur = nxt
            else:
                return None
        return cur if isinstance(cur, SectionField) else None

    def _on_validity_changed(self, valid: bool) -> None:
        del valid
        # Visual invalid marking handled via draft validity propagation at CfgFormWidget
        # level; tree does not need additional styling.

    def _remember_expanded(self, item: QTreeWidgetItem, expanded: bool) -> None:
        path = item.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
        if isinstance(path, str):
            self._expanded_state[path] = expanded

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        del column
        if item.childCount() > 0:
            item.setExpanded(not item.isExpanded())

    def _level_color(self, depth: int) -> QColor:
        return QColor(TREE_DEPTH_COLORS[depth % len(TREE_DEPTH_COLORS)])

    def _set_depth_background(self, item: QTreeWidgetItem, depth: int) -> None:
        brush = QBrush(self._level_color(depth))
        for col in range(2):
            item.setBackground(col, brush)

    def _disconnect_refs(self) -> None:
        for field, callback in self._ref_connections:
            try:
                field.on_change.disconnect(callback)  # type: ignore[attr-defined]
            except Exception:
                pass
        self._ref_connections.clear()

    def _rebuild_tree(self) -> None:
        # Preserve expanded state before clear via internal map (already tracked).
        self._disconnect_refs()
        for header in self._ref_headers:
            header.teardown()
        self._ref_headers.clear()
        for widget in self._leaf_widgets:
            try:
                widget.teardown()
            except Exception:
                pass
        self._leaf_widgets.clear()
        self._path_to_item.clear()
        self._item_depth.clear()
        self._tree.clear()

        # Decide if root header should be shown.
        root_label = getattr(self._field.spec, "label", "") or ""
        if self._context.top_level and not root_label:
            # No root header — add its children directly at depth 0.
            root = self._tree.invisibleRootItem()
            assert root is not None
            self._add_section_children(
                root,
                self._field,
                self._path,
                depth=0,
            )
        elif self._context.top_level and root_label:
            # Show root as a foldable header at depth 0.
            root_item = QTreeWidgetItem(self._tree, (root_label, ""))
            root_item.setData(0, Qt.ItemDataRole.UserRole, self._path)  # type: ignore[attr-defined]
            font = root_item.font(0)
            font.setBold(True)
            font.setPixelSize(_TREE_FONT_SIZE_PX)
            root_item.setFont(0, font)
            self._set_depth_background(root_item, 0)
            self._path_to_item[self._path] = root_item
            self._item_depth[id(root_item)] = 0
            root_item.setExpanded(self._expanded_state.get(self._path, True))
            self._add_section_children(root_item, self._field, self._path, depth=1)
        else:
            # Non-top-level section used as subtree (reference shape elision delegates here)
            root = self._tree.invisibleRootItem()
            assert root is not None
            self._add_section_children(
                root,
                self._field,
                self._path,
                depth=0,
            )

    def _add_section_children(
        self,
        parent_item: QTreeWidget | QTreeWidgetItem,
        section_field: SectionField,
        path_prefix: str,
        depth: int,
    ) -> None:
        visible = _choice_visible_keys(section_field)
        # Group handling: collect grouped field entries to render under a collapsible group item.
        grouped: dict[str, list[tuple[str, str, CfgField]]] = {}
        # First pass: gather hidden/visible without creating items for hidden.
        # We need to know which keys are visible and not hidden, and their grouping.
        entries: list[
            tuple[str, str, CfgField, str]
        ] = []  # key, child_path, field, group
        for key, child_field in section_field.fields.items():
            if visible is not None and key not in visible:
                continue
            child_path = f"{path_prefix}.{key}" if path_prefix else key
            if _is_hidden(child_field, child_path, self._context):
                continue
            group = getattr(child_field.spec, "group", "") or ""
            entries.append((key, child_path, child_field, group))
            if group:
                grouped.setdefault(group, []).append((key, child_path, child_field))

        # Render ungrouped entries first.
        for key, child_path, child_field, group in entries:
            if group:
                continue
            self._create_item_for_field(
                parent_item, child_field, key, child_path, depth
            )

        # Render grouped entries under a foldable group header per group label.
        for group_label, group_entries in grouped.items():
            group_item = QTreeWidgetItem(parent_item, (group_label, ""))  # type: ignore[arg-type]
            group_path = f"{path_prefix}.{group_label}" if path_prefix else group_label
            group_item.setData(0, Qt.ItemDataRole.UserRole, group_path)  # type: ignore[attr-defined]
            font = group_item.font(0)
            font.setBold(True)
            font.setPixelSize(_TREE_FONT_SIZE_PX)
            group_item.setFont(0, font)
            self._set_depth_background(group_item, depth)
            # Collapsed by default per spec (Advanced group).
            group_item.setExpanded(self._expanded_state.get(group_path, False))
            self._path_to_item[group_path] = group_item
            self._item_depth[id(group_item)] = depth
            for key, child_path, child_field in group_entries:
                self._create_item_for_field(
                    group_item, child_field, key, child_path, depth + 1
                )

    def _create_item_for_field(
        self,
        parent_item: QTreeWidget | QTreeWidgetItem,
        child_field: CfgField,
        key: str,
        child_path: str,
        depth: int,
    ) -> None:
        # Section
        if isinstance(child_field, SectionField):
            label = _decorated_label(
                child_field.spec.label or key,
                self._resolve_decoration(child_path, child_field),
            )
            item = QTreeWidgetItem(parent_item, (label, ""))  # type: ignore[arg-type]
            item.setData(0, Qt.ItemDataRole.UserRole, child_path)  # type: ignore[attr-defined]
            font = item.font(0)
            font.setBold(True)
            font.setPixelSize(_TREE_FONT_SIZE_PX)
            item.setFont(0, font)
            self._set_depth_background(item, depth)
            self._path_to_item[child_path] = item
            self._item_depth[id(item)] = depth
            item.setExpanded(self._expanded_state.get(child_path, True))
            self._add_section_children(item, child_field, child_path, depth + 1)
            return
        # Reference
        if isinstance(child_field, ReferenceField):
            spec = child_field.spec  # type: ignore[attr-defined]
            label = _decorated_label(
                spec.label or key, self._resolve_decoration(child_path, child_field)
            )
            item = QTreeWidgetItem(parent_item, (label, ""))  # type: ignore[arg-type]
            item.setData(0, Qt.ItemDataRole.UserRole, child_path)  # type: ignore[attr-defined]
            font = item.font(0)
            font.setBold(True)
            font.setPixelSize(_TREE_FONT_SIZE_PX)
            item.setFont(0, font)
            self._set_depth_background(item, depth)
            self._path_to_item[child_path] = item
            self._item_depth[id(item)] = depth
            item.setExpanded(self._expanded_state.get(child_path, True))
            header = _TreeReferenceHeader(child_field)
            header.setFont(self._tree.font())
            self._ref_headers.append(header)
            self._tree.setItemWidget(item, 1, header)
            # Elide guaranteed single materialized shape row: render shape fields directly under reference.
            sub = child_field.sub_field
            if sub is not None:
                # Shape elision: children depth advances only by one, not two.
                self._add_section_children(item, sub, child_path, depth + 1)

            # Keep reference children in sync when chosen key changes.
            def _on_ref_change(
                *_: object, path: str = child_path, field: ReferenceField = child_field
            ) -> None:
                # Full rebuild preserves consistency; expanded state kept.
                self._rebuild_tree()

            child_field.on_change.connect(_on_ref_change)  # type: ignore[attr-defined]
            self._ref_connections.append((child_field, _on_ref_change))
            return
        # Sweep / CenteredSweep / Scalar / Literal leaf
        leaf_label = self._leaf_label(key, child_field, child_path)
        item = QTreeWidgetItem(parent_item, (leaf_label, ""))  # type: ignore[arg-type]
        item.setData(0, Qt.ItemDataRole.UserRole, child_path)  # type: ignore[attr-defined]
        self._set_depth_background(item, depth)
        # Create editor widget via exact registry.
        child_context = self._context.derive(path=child_path, top_level=False)
        widget = self._context.registry.render(child_field, child_context)
        cast(QWidget, widget).setFont(self._tree.font())
        # Track for teardown.
        self._leaf_widgets.append(widget)
        self._tree.setItemWidget(item, 1, cast(QWidget, widget))
        # Apply enabled decoration via widget enablement if needed (registry widgets already handle decoration enabled?).
        # Respect decoration enabled.
        dec = self._resolve_decoration(child_path, child_field)
        if dec is not None and not bool(getattr(dec, "enabled", True)):
            cast(QWidget, widget).setEnabled(False)

    def _leaf_label(self, key: str, field: CfgField, path: str) -> str:
        spec_label = getattr(field.spec, "label", "") or key
        dec = self._resolve_decoration(path, field)
        return _decorated_label(spec_label, dec)

    def _resolve_decoration(self, path: str, field: CfgField) -> object | None:
        resolver = self._context.decoration_for_path
        if resolver is None:
            return None
        try:
            return resolver(path, field)
        except Exception:
            return None


class StructuralAdapter(Protocol):
    """View-only structural composition over a ``SectionField``."""

    def create_root(
        self,
        field: SectionField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol: ...


@final
class FormStructure:
    """Default form composition (existing QFormLayout presentation)."""

    def create_root(
        self,
        field: SectionField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol:
        return context.registry.render(field, context)


@final
class TreeStructure:
    """Dense tree composition per spec.md (S1/S2)."""

    def create_root(
        self,
        field: SectionField,
        context: FieldRenderContext,
    ) -> FieldWidgetProtocol:
        return TreeCfgWidget(field, context)


# Convenience singletons
form_structure = FormStructure()
tree_structure = TreeStructure()


__all__ = [
    "FormStructure",
    "StructuralAdapter",
    "TREE_DEPTH_COLORS",
    "TreeCfgWidget",
    "TreeStructure",
    "form_structure",
    "tree_structure",
]
