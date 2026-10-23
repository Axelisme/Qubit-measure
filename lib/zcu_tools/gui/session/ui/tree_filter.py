"""Live name/path filtering shared by the Inspect parameter and library trees."""

from qtpy.QtCore import QSignalBlocker
from qtpy.QtWidgets import (
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
)


def _path(item: QTreeWidgetItem) -> tuple[str, ...]:
    parts = [item.text(0)]
    parent = item.parent()
    while parent is not None:
        parts.append(parent.text(0))
        parent = parent.parent()
    return tuple(reversed(parts))


class TreeFilter(QLineEdit):
    """Filter column-zero names and dotted paths without changing tree selection.

    Typing shows case-insensitive substring matches and their ancestors, expanding
    ancestors to reveal results. Clearing restores the user's unfiltered expansion
    choices, keyed by caption path for this widget's lifetime. Expansion changes
    while filtering are temporary. Call refresh after rebuilding the tree; new
    branches use default_expanded. The tree must have unique caption paths.
    """

    def __init__(
        self, tree: QTreeWidget, *, object_name: str, default_expanded: bool
    ) -> None:
        super().__init__()
        self._tree = tree
        self._default_expanded = default_expanded
        self._expanded: dict[tuple[str, ...], bool] = {}
        self.setObjectName(object_name)
        self.setPlaceholderText("Filter")
        self.setClearButtonEnabled(True)
        self.textChanged.connect(self.refresh)
        tree.itemExpanded.connect(self._remember)
        tree.itemCollapsed.connect(self._remember)

    def _remember(self, item: QTreeWidgetItem) -> None:
        if not self.text():
            self._expanded[_path(item)] = item.isExpanded()

    def refresh(self) -> None:
        """Apply the current query and expansion choices to the current tree rows."""
        rows: list[tuple[QTreeWidgetItem, tuple[str, ...]]] = []
        iterator = QTreeWidgetItemIterator(self._tree)
        while (item := iterator.value()) is not None:
            rows.append((item, _path(item)))
            iterator += 1
        query = self.text().casefold()
        visible: set[tuple[str, ...]] = set()
        if query:
            for _item, path in rows:
                if query in ".".join(path).casefold():
                    visible.update(path[:depth] for depth in range(1, len(path) + 1))
        ancestors = {path[:-1] for path in visible}
        # Auto-expansion must not become the user's saved browsing state.
        with QSignalBlocker(self._tree):
            for item, path in rows:
                item.setHidden(bool(query) and path not in visible)
                if item.childCount():
                    expanded = (
                        path in ancestors
                        if query
                        else self._expanded.get(path, self._default_expanded)
                    )
                    item.setExpanded(expanded)
