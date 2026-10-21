"""Locate presented widgets through Qt's public child tree and accessible labels."""

from qtpy.QtWidgets import QComboBox, QLabel, QStackedWidget, QWidget
from zcu_tools.gui.app.measure.ui.data_figure_preview_gallery import (
    DataFigurePreviewGallery,
)


def gallery(tab: QWidget) -> DataFigurePreviewGallery:
    """Return the unique shipped gallery below a tab, failing on missing/ambiguous UI."""
    galleries = tab.findChildren(DataFigurePreviewGallery)
    assert len(galleries) == 1
    return galleries[0]


def right_stack(tab: QWidget) -> QStackedWidget:
    """Return the gallery's containing page stack using public Qt ancestry."""
    stack = gallery(tab).parentWidget()
    assert isinstance(stack, QStackedWidget)
    return stack


def right_page(tab: QWidget, pane: str) -> QWidget:
    """Locate a right page by the gallery, selector label or placeholder widget kind."""
    if pane == "data":
        return gallery(tab)
    if pane in ("run", "analysis", "post_analysis"):
        label = {
            "run": "Run figure",
            "analysis": "Analysis figure",
            "post_analysis": "Post-analysis figure",
        }[pane]
        selectors = [
            w for w in tab.findChildren(QComboBox) if w.accessibleName() == label
        ]
        assert len(selectors) == 1
        parent = selectors[0].parentWidget()
        assert parent is not None
        return parent
    kind = QLabel
    stack = right_stack(tab)
    pages = [stack.widget(i) for i in range(stack.count())]
    matches = [page for page in pages if isinstance(page, kind)]
    assert len(matches) == 1
    return matches[0]
