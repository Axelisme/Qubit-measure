"""SpectrumListWidget — the Spectra page's list of spectra, one two-line row each.

Each row shows the literal spectrum name over its details line
(``kind · stage · points`` plus an unfinished draft kind and the Filter-excluded
count when present). Rows elide to the list width instead of scrolling
sideways; a long name elides in the middle and the tooltip carries it whole.
"""

from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import QModelIndex, Qt
from qtpy.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from zcu_tools.gui.app.fluxdep.controller import SpectrumRow

# List item role holding a row's second line (kind · stage · points · ...).
_DETAILS_ROLE = Qt.ItemDataRole.UserRole + 1


class _SpectrumRowDelegate(QStyledItemDelegate):
    """Draw a spectrum row as its name over its details, eliding a long name.

    Each item holds the literal name (UserRole) and the details line
    (_DETAILS_ROLE). A name wider than the cell is elided in the middle so
    both ends stay readable; the item tooltip carries the full name. Qt
    right-elides an overlong details line as usual.
    """

    def initStyleOption(
        self,
        option: QStyleOptionViewItem | None,
        index: QModelIndex,
    ) -> None:
        super().initStyleOption(option, index)
        if option is None:
            raise TypeError("initStyleOption needs the option Qt paints with")
        name = index.data(Qt.ItemDataRole.UserRole)
        details = index.data(_DETAILS_ROLE)
        if not isinstance(name, str) or not isinstance(details, str):
            raise TypeError("spectrum rows hold a str name and str details")
        widget = option.widget
        style = widget.style()
        if style is None:
            raise RuntimeError("spectrum rows are painted by a styled list widget")
        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, option, widget
        )
        # Qt insets item text by the focus-frame margin plus one on each side.
        margin = (
            style.pixelMetric(QStyle.PixelMetric.PM_FocusFrameHMargin, None, widget) + 1
        )
        width = max(text_rect.width() - 2 * margin, 0)
        elided = option.fontMetrics.elidedText(
            name, Qt.TextElideMode.ElideMiddle, width
        )
        # Qt lays item text out with QTextLayout, which breaks lines only at
        # U+2028 (QChar.LineSeparator), not at "\n".
        option.text = f"{elided} {details}"


def _row_details(row: SpectrumRow) -> str:
    """A row's second line, from the query that spectrum.list reports.

    Spec type, stage and published point count always show; an unfinished
    draft kind and the Filter-excluded count only when present.
    """
    parts = [row.spec_type, row.stage, f"{row.point_count} pts"]
    if row.pending is not None:
        parts.append(f"{row.pending} draft")
    if row.filter_excluded_count:
        parts.append(f"{row.filter_excluded_count} excluded")
    return " · ".join(parts)


class SpectrumListWidget(QListWidget):
    """The spectrum list, objectName ``fluxdep.spectra``.

    Each item's text is ``"<name>\\n<details>"``, its UserRole data is the
    literal name and its tooltip the full name. Hosts listen to
    currentItemChanged for human picks.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fluxdep.spectra")
        # Rows elide to the list width instead of scrolling sideways.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setItemDelegate(_SpectrumRowDelegate(self))

    def show_rows(self, rows: Sequence[SpectrumRow], active: str | None) -> None:
        """Replace every row with rows, in order, and select the row named active.

        Signals are blocked while refilling, so neither clearing nor selecting
        emits currentItemChanged. An active name not in rows selects nothing.
        Build rows completely before calling: the row query may re-enter.
        """
        self.blockSignals(True)
        self.clear()
        for row in rows:
            details = _row_details(row)
            # Literal name over its details; the delegate elides a long name.
            item = QListWidgetItem(f"{row.name}\n{details}")
            item.setData(Qt.ItemDataRole.UserRole, row.name)
            item.setData(_DETAILS_ROLE, details)
            item.setToolTip(row.name)
            self.addItem(item)
            if row.name == active:
                self.setCurrentItem(item)
        self.blockSignals(False)
