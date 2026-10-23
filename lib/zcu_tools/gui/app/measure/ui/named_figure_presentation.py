"""Shared name-model and tile selection for attached native figure canvases."""

from collections.abc import Sequence

from qtpy.QtWidgets import QComboBox

from .figure_pane_layout import FigurePaneLayout


def sync_figure_selection(model: QComboBox, stack_index: int) -> None:
    """Store the selected canvas name in a hidden model without emitting signals.

    The stack's placeholder has index zero. Named canvases occupy subsequent
    indices in model order. Page changes and tile clicks are authoritative.
    """
    model.blockSignals(True)
    try:
        model.setCurrentIndex(stack_index - 1)
    finally:
        model.blockSignals(False)


def present_named_figure_tiles(
    stack: FigurePaneLayout,
    model: QComboBox,
    names: Sequence[str],
    *,
    selected: str,
    pane: str,
) -> None:
    """Page attached canvases and retain selected, or select the first name.

    Call on the Qt owner after attachment, with names in stack order excluding
    the placeholder. Empty names select the placeholder and hide paging. The
    hidden combo stores names only; no Figure ownership or membership changes.
    pane identifies public canvas interaction targets as figures.tile.<pane>.<i>.
    """
    model.blockSignals(True)
    try:
        model.clear()
        model.addItems(list(names))
        if selected in names:
            model.setCurrentText(selected)
    finally:
        model.blockSignals(False)
    for index in range(1, stack.count()):
        canvas = stack.widget(index)
        if canvas is not None:
            canvas.setObjectName(f"figures.tile.{pane}.{index - 1}")
    stack.setCurrentIndex(model.currentIndex() + 1 if names else 0)
    stack.set_tiled(tiled=bool(names))
