"""Dense cfg tree guide painting and row presentation contracts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QRect, Qt
from qtpy.QtGui import QColor, QImage, QPainter
from qtpy.QtWidgets import QApplication, QStyle, QStyleOption, QTreeWidget
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.gui.widgets.cfg import TREE_DEPTH_COLORS, CfgFormWidget
from zcu_tools.gui.widgets.cfg.structure import make_dense_cfg_tree


def _paint_branch(
    tree: QTreeWidget, viewport_x: int, state: QStyle.StateFlag
) -> tuple[QImage, QRect]:
    rect = QRect(viewport_x, 0, tree.indentation(), 20)
    image = QImage(rect.right() + 1, rect.height(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    option = QStyleOption()
    option.rect = rect
    option.state = state
    style = tree.style()
    assert style is not None
    painter = QPainter(image)
    try:
        style.drawPrimitive(
            QStyle.PrimitiveElement.PE_IndicatorBranch, option, painter, tree
        )
    finally:
        painter.end()
    return image, rect


@pytest.mark.parametrize("depth", range(2 * len(TREE_DEPTH_COLORS)))
def test_tree_guides_cycle_at_each_logical_depth(
    qapp: QApplication, depth: int
) -> None:
    tree, style = make_dense_cfg_tree()
    try:
        assert style.parent() is tree
        image, rect = _paint_branch(
            tree,
            depth * tree.indentation(),
            QStyle.StateFlag.State_Sibling | QStyle.StateFlag.State_Item,
        )
        expected = QColor(TREE_DEPTH_COLORS[depth % len(TREE_DEPTH_COLORS)])
        assert image.pixelColor(rect.center().x(), rect.top() + 1) == expected
        assert image.pixelColor(rect.right() - 1, rect.center().y()) == expected
    finally:
        tree.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("depth", [2, 7])
def test_tree_guide_color_is_stable_under_horizontal_scroll(
    qapp: QApplication, depth: int
) -> None:
    tree, _style = make_dense_cfg_tree()
    try:
        logical_x = depth * tree.indentation()
        state = QStyle.StateFlag.State_Sibling | QStyle.StateFlag.State_Item
        unscrolled, original_rect = _paint_branch(tree, logical_x, state)
        scroll = tree.horizontalScrollBar()
        assert scroll is not None
        scroll.setRange(0, 100)
        scroll.setValue(15)
        assert scroll.value() == 15
        scrolled, scrolled_rect = _paint_branch(tree, logical_x - scroll.value(), state)
        expected = QColor(TREE_DEPTH_COLORS[depth % len(TREE_DEPTH_COLORS)])
        assert unscrolled.pixelColor(original_rect.center()) == expected
        assert scrolled.pixelColor(scrolled_rect.center()) == expected
    finally:
        tree.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    ("state", "has_sibling", "has_item"),
    [
        (QStyle.StateFlag.State_None, False, False),
        (QStyle.StateFlag.State_Sibling, True, False),
        (QStyle.StateFlag.State_Item, False, True),
        (QStyle.StateFlag.State_Sibling | QStyle.StateFlag.State_Item, True, True),
    ],
)
def test_tree_branch_segments_follow_item_and_sibling_state(
    qapp: QApplication, state: QStyle.StateFlag, has_sibling: bool, has_item: bool
) -> None:
    tree, _style = make_dense_cfg_tree()
    try:
        image, rect = _paint_branch(tree, 0, state)
        expected = QColor(TREE_DEPTH_COLORS[0])
        transparent = QColor(Qt.GlobalColor.transparent)
        assert image.pixelColor(rect.center().x(), rect.top() + 1) == (
            expected if has_sibling or has_item else transparent
        )
        assert image.pixelColor(rect.center().x(), rect.bottom() - 1) == (
            expected if has_sibling else transparent
        )
        assert image.pixelColor(rect.right() - 1, rect.center().y()) == (
            expected if has_item else transparent
        )
    finally:
        tree.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(("scroll_offset", "depth"), [(0, 0), (10, 0), (20, 1)])
def test_clipped_tree_guide_uses_normalized_logical_depth(
    qapp: QApplication, scroll_offset: int, depth: int
) -> None:
    tree, _style = make_dense_cfg_tree()
    try:
        scroll = tree.horizontalScrollBar()
        assert scroll is not None
        scroll.setRange(0, 100)
        scroll.setValue(scroll_offset)
        assert scroll.value() == scroll_offset
        image, rect = _paint_branch(tree, -5, QStyle.StateFlag.State_Item)
        assert image.pixelColor(rect.right(), rect.center().y()) == QColor(
            TREE_DEPTH_COLORS[depth]
        )
    finally:
        tree.deleteLater()
        qapp.processEvents()


def test_nested_tree_keeps_row_backgrounds_and_root_alignment(
    qapp: QApplication, ctrl: MagicMock
) -> None:
    nested_spec = CfgSectionSpec(
        label="L6", fields={"leaf": ScalarSpec(label="Leaf", type=int)}
    )
    nested_value = CfgSectionValue(fields={"leaf": DirectValue(1)})
    for depth in reversed(range(1, 6)):
        nested_spec = CfgSectionSpec(label=f"L{depth}", fields={"child": nested_spec})
        nested_value = CfgSectionValue(fields={"child": nested_value})
    schema = CfgSchema(
        spec=CfgSectionSpec(label="Root", fields={"child": nested_spec}),
        value=CfgSectionValue(fields={"child": nested_value}),
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        tree = form.findChild(QTreeWidget, "cfgTree")
        assert tree is not None
        assert tree.isHeaderHidden()
        assert not tree.rootIsDecorated()
        assert tree.indentation() == 10
        root = tree.topLevelItem(0)
        assert root is not None
        assert root.parent() is None
        item = root
        depth_colors = {QColor(color).name() for color in TREE_DEPTH_COLORS}
        for depth in range(8):
            assert item.background(0).color().name() not in depth_colors
            if depth < 7:
                assert item.childCount() == 1
                child = item.child(0)
                assert child is not None
                item = child
        assert item.text(0) == "Leaf"
        assert form.read_values() == schema.value
    finally:
        form.detach()
        draft.close()
        form.deleteLater()
        qapp.processEvents()
