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
    QHeaderView,
    QProxyStyle,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.cfg import CfgSectionSpec, ReferenceSpec
from zcu_tools.gui.cfg.binding import (
    CenteredSweepField,
    CfgField,
    ReferenceField,
    ScalarField,
    SectionField,
    SweepField,
)

from .presentation import (
    apply_tree_item_decoration,
    choice_visible_keys,
    decorated_label,
    is_hidden,
)
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


def _branch_color(depth: int) -> QColor:
    """Return depth-cycled guide-line color."""
    return QColor(TREE_DEPTH_COLORS[depth % len(TREE_DEPTH_COLORS)])


class _TreeBranchStyle(QProxyStyle):
    """Classic vertical/horizontal branch lines without triangles.

    Guide lines are depth-colored via TREE_DEPTH_COLORS cycling; rows no
    longer use depth backgrounds (A1).
    """

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
        # A1: depth is per-segment indentation column only; do not overwrite with row model depth
        pen_color = QColor("#b8c1cc")
        try:
            if _INDENTATION_PX:
                depth_guess = (
                    max(0, int(rect.x() // _INDENTATION_PX)) if rect.x() >= 0 else 0
                )
                pen_color = _branch_color(depth_guess)
        except Exception:
            pen_color = QColor("#b8c1cc")
        painter.save()
        painter.setPen(QPen(pen_color, 1))
        if has_sibling:
            painter.drawLine(x, rect.top(), x, rect.bottom())
        else:
            painter.drawLine(x, rect.top(), x, y)
        if has_item:
            painter.drawLine(x, y, rect.right(), y)
        painter.restore()


# choice_visible_keys, is_hidden, decorated_label now imported from presentation (single source)


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
        # A2: viewport follows available panel height — tree expands, no fixed threshold
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # type: ignore[attr-defined]
        self._tree = QTreeWidget()
        self._tree.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )  # type: ignore[attr-defined]
        self._tree.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        self._tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        layout.addWidget(self._tree, stretch=1)

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
        self._leaf_path_to_widget: dict[str, FieldWidgetProtocol] = {}
        self._ref_headers: list[FieldWidgetProtocol] = []
        self._ref_connections: list[tuple[ReferenceField, object]] = []
        self._ref_enabled_connections: list[tuple[ReferenceField, object]] = []
        self._ref_prev_state: dict[str, tuple[str, int | None, str | None]] = {}
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
            try:
                header.teardown()
            except Exception:
                pass
        self._ref_headers.clear()
        for widget in self._leaf_widgets:
            try:
                widget.teardown()
            except Exception:
                pass
        self._leaf_widgets.clear()
        self._leaf_path_to_widget.clear()
        self._tree.clear()
        self._ref_prev_state.clear()

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
        # A1: rows no longer use depth background colors; guide lines are colored instead.
        # Kept for compatibility but intentionally no-op.
        return

    def _disconnect_refs(self) -> None:
        for field, callback in self._ref_connections:
            try:
                field.on_change.disconnect(callback)  # type: ignore[attr-defined]
            except Exception:
                pass
        self._ref_connections.clear()
        for field, callback in self._ref_enabled_connections:
            try:
                field.on_enabled_changed.disconnect(callback)  # type: ignore[attr-defined]
            except Exception:
                pass
        self._ref_enabled_connections.clear()

    def _rebuild_tree(self) -> None:
        # Preserve expanded state before clear via internal map (already tracked).
        self._disconnect_refs()
        for header in self._ref_headers:
            try:
                header.teardown()
            except Exception:
                pass
        self._ref_headers.clear()
        for widget in self._leaf_widgets:
            try:
                widget.teardown()
            except Exception:
                pass
        self._leaf_widgets.clear()
        self._leaf_path_to_widget.clear()
        self._path_to_item.clear()
        self._item_depth.clear()
        self._ref_prev_state.clear()
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
        # Re-populate ref prev state for all references after full rebuild
        # (handled incrementally in _create_item_for_field for new refs)

    def _rebuild_reference_children(self, path: str) -> None:
        item = self._path_to_item.get(path)
        if item is None:
            self._rebuild_tree()
            return
        # Preserve expanded state of the reference itself
        depth = self._item_depth.get(id(item), 0)
        # Find the ReferenceField for this path
        field = self._find_reference_field(path)
        if field is None or field.sub_field is None:
            # No shape – clear children and teardown leaf widgets under this reference
            to_remove_empty = [
                p for p in list(self._path_to_item.keys()) if p.startswith(path + ".")
            ]
            for p in to_remove_empty:
                widget = self._leaf_path_to_widget.pop(p, None)
                if widget is not None:
                    try:
                        widget.teardown()
                    except Exception:
                        pass
                    try:
                        self._leaf_widgets.remove(widget)
                    except ValueError:
                        pass
                    cast(QWidget, widget).deleteLater()
                self._path_to_item.pop(p, None)
            while item.childCount():
                item.takeChild(0)
            return
        # Tear down existing children widgets under this reference
        to_remove = [
            p for p in list(self._path_to_item.keys()) if p.startswith(path + ".")
        ]
        # Teardown leaf widgets whose path is under this reference
        for p in to_remove:
            widget = self._leaf_path_to_widget.pop(p, None)
            if widget is not None:
                try:
                    widget.teardown()
                except Exception:
                    pass
                try:
                    self._leaf_widgets.remove(widget)
                except ValueError:
                    pass
                cast(QWidget, widget).deleteLater()
            self._path_to_item.pop(p, None)
        # Clean up item depth entries for removed paths (best effort)
        # Remove all children from the reference item
        while item.childCount():
            item.takeChild(0)
        # Re-add children for the new shape
        self._add_section_children(item, field.sub_field, path, depth + 1)

    def _find_reference_field(self, path: str) -> ReferenceField | None:
        # Walk from root to locate ReferenceField at ``path``
        if not path:
            return None
        parts = path.split(".") if path else []
        cur: CfgField = self._field
        # Strip root prefix if needed
        remaining = path
        if self._path:
            if not path.startswith(self._path):
                return None
            if path == self._path:
                return None
            remaining = path.removeprefix(self._path + ".")
            parts = remaining.split(".") if remaining else []
        for i, part in enumerate(parts):
            if isinstance(cur, SectionField):
                nxt = cur.fields.get(part)
                if nxt is None:
                    return None
                if i == len(parts) - 1:
                    return nxt if isinstance(nxt, ReferenceField) else None
                cur = nxt
            elif isinstance(cur, ReferenceField):
                sub = cur.sub_field
                if sub is None:
                    return None
                nxt = sub.fields.get(part)
                if nxt is None:
                    return None
                if i == len(parts) - 1:
                    return nxt if isinstance(nxt, ReferenceField) else None
                cur = nxt
            else:
                return None
        return None

    def _find_field_for_path(self, path: str) -> CfgField | None:
        """Generic field lookup by full dotted cfg path (for decoration/ancestor checks)."""
        if not path:
            return None
        if path == self._path:
            return self._field
        remaining = path
        if self._path:
            if not path.startswith(self._path):
                return None
            if path == self._path:
                return self._field
            remaining = path.removeprefix(self._path + ".")
        parts = remaining.split(".") if remaining else []
        cur: CfgField = self._field
        for part in parts:
            if isinstance(cur, SectionField):
                nxt = cur.fields.get(part)  # type: ignore[attr-defined]
                if nxt is None:
                    return None
                cur = nxt
            elif isinstance(cur, ReferenceField):
                sub = cur.sub_field
                if sub is None:
                    return None
                nxt = sub.fields.get(part)  # type: ignore[attr-defined]
                if nxt is None:
                    return None
                cur = nxt
            else:
                return None
        return cur

    def _add_section_children(
        self,
        parent_item: QTreeWidget | QTreeWidgetItem,
        section_field: SectionField,
        path_prefix: str,
        depth: int,
    ) -> None:
        visible = choice_visible_keys(section_field)
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
            if is_hidden(child_path, child_field, self._context):
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
            # Propagate disabled from parent section if any (ignore invisible root)
            if (
                isinstance(parent_item, QTreeWidgetItem)
                and parent_item is not self._tree.invisibleRootItem()
                and parent_item.isDisabled()
            ):
                group_item.setDisabled(True)
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
        # Resolve decoration once for this path/field
        dec = None
        if self._context.decoration_for_path is not None:
            try:
                dec = self._context.decoration_for_path(child_path, child_field)  # type: ignore[arg-type]
            except Exception:
                dec = None
        # Section
        if isinstance(child_field, SectionField):
            label = decorated_label(child_field, key, child_path, self._context)
            item = QTreeWidgetItem(parent_item, (label, ""))  # type: ignore[arg-type]
            item.setData(0, Qt.ItemDataRole.UserRole, child_path)  # type: ignore[attr-defined]
            font = item.font(0)
            font.setBold(True)
            font.setPixelSize(_TREE_FONT_SIZE_PX)
            item.setFont(0, font)
            apply_tree_item_decoration(item, None, dec)
            # Propagate disabled from parent (ignore invisible root)
            if (
                isinstance(parent_item, QTreeWidgetItem)
                and parent_item is not self._tree.invisibleRootItem()
                and parent_item.isDisabled()
            ):
                item.setDisabled(True)
            self._path_to_item[child_path] = item
            self._item_depth[id(item)] = depth
            item.setExpanded(self._expanded_state.get(child_path, True))
            self._add_section_children(item, child_field, child_path, depth + 1)
            return
        # Reference
        if isinstance(child_field, ReferenceField):
            spec = child_field.spec  # type: ignore[attr-defined]
            label = decorated_label(child_field, key, child_path, self._context)
            item = QTreeWidgetItem(parent_item, (label, ""))  # type: ignore[arg-type]
            item.setData(0, Qt.ItemDataRole.UserRole, child_path)  # type: ignore[attr-defined]
            font = item.font(0)
            font.setBold(True)
            font.setPixelSize(_TREE_FONT_SIZE_PX)
            item.setFont(0, font)
            self._path_to_item[child_path] = item
            self._item_depth[id(item)] = depth
            item.setExpanded(self._expanded_state.get(child_path, True))
            # Use exact renderer authority: obtain reference header via registry
            # (shared ReferenceWidget with render_reference_children=False)
            ref_context = self._context.derive(
                path=child_path, top_level=False, render_reference_children=False
            )
            header_widget = self._context.registry.render(child_field, ref_context)
            cast(QWidget, header_widget).setFont(self._tree.font())
            self._ref_headers.append(header_widget)
            self._tree.setItemWidget(item, 1, cast(QWidget, header_widget))
            apply_tree_item_decoration(item, cast(QWidget, header_widget), dec)
            # Propagate disabled from parent (ignore invisible root)
            if (
                isinstance(parent_item, QTreeWidgetItem)
                and parent_item is not self._tree.invisibleRootItem()
                and parent_item.isDisabled()
            ):
                item.setDisabled(True)
                cast(QWidget, header_widget).setEnabled(False)
            # Elide guaranteed single materialized shape row: render shape fields directly under reference.
            sub = child_field.sub_field
            if sub is not None:
                # Shape elision: children depth advances only by one, not two.
                self._add_section_children(item, sub, child_path, depth + 1)
                # If reference is optional and currently disabled, disable its subtree
                # (header itself remains enabled so combo can re-enable)
                if child_field.spec.optional and not child_field.is_enabled:
                    stack = [item]
                    while stack:
                        cur = stack.pop()
                        for idx in range(cur.childCount()):
                            ch = cur.child(idx)
                            if ch is None:
                                continue
                            ch.setDisabled(True)
                            w = self._tree.itemWidget(ch, 1)
                            if w is not None:
                                w.setEnabled(False)
                            stack.append(ch)
            # Keep reference children in sync only when structural identity changes
            # (chosen key, materialized shape, or sub_field identity). Value edits
            # must preserve leaf editors/focus.
            prev_key = child_field.get_chosen_key()
            prev_sub_id = id(sub) if sub is not None else None
            prev_label = sub.spec.label if sub is not None else None  # type: ignore[attr-defined]
            self._ref_prev_state[child_path] = (prev_key, prev_sub_id, prev_label)

            def _on_ref_change(
                *_: object, path: str = child_path, field: ReferenceField = child_field
            ) -> None:
                prev = self._ref_prev_state.get(path)
                cur_key = field.get_chosen_key()
                cur_sub = field.sub_field
                cur_id = id(cur_sub) if cur_sub is not None else None
                cur_label = cur_sub.spec.label if cur_sub is not None else None  # type: ignore[attr-defined]
                cur_state = (cur_key, cur_id, cur_label)
                if prev == cur_state:
                    # Only value edits inside the shape – keep editors.
                    return
                self._ref_prev_state[path] = cur_state
                self._rebuild_reference_children(path)

            child_field.on_change.connect(_on_ref_change)  # type: ignore[attr-defined]
            self._ref_connections.append((child_field, _on_ref_change))

            def _on_ref_enabled_changed(
                enabled: bool,
                path: str = child_path,
                field: ReferenceField = child_field,
            ) -> None:
                # Re-enable must not overwrite descendant local authorities:
                # nested optional references that remain disabled and
                # decoration-disabled containers/children stay disabled
                # (S1/A4, form/tree A2 parity). Compute inherited
                # effective-enabled from decoration projection plus
                # ancestor optional-reference gating, keeping the
                # nested optional header itself enabled for re-selection.
                ref_item = self._path_to_item.get(path)
                if ref_item is None:
                    return
                # Walk the QTreeWidget hierarchy so leaf rows (which are not in _path_to_item) are covered.
                stack: list[QTreeWidgetItem] = [ref_item]
                while stack:
                    cur = stack.pop()
                    for idx in range(cur.childCount()):
                        ch = cur.child(idx)
                        if ch is None:
                            continue
                        desc_path_obj = ch.data(0, Qt.ItemDataRole.UserRole)  # type: ignore[attr-defined]
                        desc_path = (
                            desc_path_obj if isinstance(desc_path_obj, str) else ""
                        )
                        # Compute inherited effective-enabled for this descendant:
                        # own decoration plus every strict ancestor's decoration
                        # and optional-reference gating. The descendant's own
                        # optional is_enabled does NOT disable its header.
                        effective = True
                        # Own decoration
                        if desc_path:
                            desc_field = self._find_field_for_path(desc_path)
                            if desc_field is not None:
                                dec = None
                                if self._context.decoration_for_path is not None:
                                    try:
                                        dec = self._context.decoration_for_path(
                                            desc_path, desc_field
                                        )  # type: ignore[arg-type]
                                    except Exception:
                                        dec = None
                                if dec is not None and not dec.enabled:
                                    effective = False
                        # Strict ancestors: decoration gate + optional-reference gate
                        if effective and desc_path and "." in desc_path:
                            parts = desc_path.split(".")
                            for i in range(1, len(parts)):
                                anc = ".".join(parts[:i])
                                anc_field = self._find_field_for_path(anc)
                                if anc_field is None:
                                    # Synthetic group header has no field; its
                                    # ancestors are still checked in other iterations.
                                    continue
                                # Decoration gate for ancestor container (Section/Reference)
                                anc_dec = None
                                if self._context.decoration_for_path is not None:
                                    try:
                                        anc_dec = self._context.decoration_for_path(
                                            anc, anc_field
                                        )  # type: ignore[arg-type]
                                    except Exception:
                                        anc_dec = None
                                if anc_dec is not None and not anc_dec.enabled:
                                    effective = False
                                    break
                                # Optional-reference gate for ancestors only
                                if (
                                    isinstance(anc_field, ReferenceField)
                                    and anc_field.spec.optional
                                    and not anc_field.is_enabled
                                ):
                                    effective = False
                                    break
                        ch.setDisabled(not effective)
                        w = self._tree.itemWidget(ch, 1)
                        if w is not None:
                            w.setEnabled(effective)
                        stack.append(ch)

            if child_field.spec.optional:
                child_field.on_enabled_changed.connect(_on_ref_enabled_changed)  # type: ignore[attr-defined]
                self._ref_enabled_connections.append(
                    (child_field, _on_ref_enabled_changed)
                )
            return
        # Sweep / CenteredSweep / Scalar / Literal leaf
        leaf_label = decorated_label(child_field, key, child_path, self._context)
        item = QTreeWidgetItem(parent_item, (leaf_label, ""))  # type: ignore[arg-type]
        item.setData(0, Qt.ItemDataRole.UserRole, child_path)  # type: ignore[attr-defined]
        # Create editor widget via exact registry.
        child_context = self._context.derive(path=child_path, top_level=False)
        widget = self._context.registry.render(child_field, child_context)
        cast(QWidget, widget).setFont(self._tree.font())
        # Track for teardown.
        self._leaf_widgets.append(widget)
        self._leaf_path_to_widget[child_path] = widget
        self._tree.setItemWidget(item, 1, cast(QWidget, widget))
        apply_tree_item_decoration(item, cast(QWidget, widget), dec)
        # Propagate disabled from parent (ignore invisible root)
        if (
            isinstance(parent_item, QTreeWidgetItem)
            and parent_item is not self._tree.invisibleRootItem()
            and parent_item.isDisabled()
        ):
            item.setDisabled(True)
            cast(QWidget, widget).setEnabled(False)

    # _leaf_label and _resolve_decoration removed; use presentation.decorated_label and apply_tree_item_decoration directly


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
