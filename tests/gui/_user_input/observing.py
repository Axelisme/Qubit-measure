"""Read visible widget/plot state and capture caller-owned screenshot files."""

from pathlib import Path

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QStatusBar,
    QStyle,
    QStyleOptionComboBox,
    QTreeWidget,
    QWidget,
)

from .._observations import (
    DialogObservation,
    LegendObservation,
    ListRowObservation,
    PlotAxesObservation,
    PlotLabelBoundsObservation,
    PlotObservation,
    StatusObservation,
    TreeRowObservation,
    VisibleObservation,
    WidgetObservation,
)
from . import collections
from .selection import display_text, on_screen, unique


def _caption_fits(widget: QWidget) -> bool:
    if isinstance(widget, QComboBox):
        option = QStyleOptionComboBox()
        # Native dialog children cannot expose QComboBox's protected initializer.
        option.initFrom(widget)
        option.editable = widget.isEditable()
        option.frame = widget.hasFrame()
        option.currentText = widget.currentText()
        option.currentIcon = widget.itemIcon(widget.currentIndex())
        option.iconSize = widget.iconSize()
        style = widget.style()
        if style is None:
            raise RuntimeError("Combo box has no Qt style")
        text_rect = style.subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            widget,
        )
        return (
            widget.fontMetrics().horizontalAdvance(widget.currentText())
            <= text_rect.width()
        )
    if isinstance(widget, QLabel) and widget.wordWrap():
        bounds = widget.fontMetrics().boundingRect(
            widget.contentsRect(), Qt.TextFlag.TextWordWrap, widget.text()
        )
        return bounds.height() <= widget.contentsRect().height()
    if isinstance(widget, (QAbstractButton, QLabel)):
        return (
            widget.fontMetrics().horizontalAdvance(widget.text())
            <= widget.contentsRect().width()
        )
    return True


def observe_widget(widget: QWidget) -> WidgetObservation:
    """Snapshot display/availability/help without exposing hidden widget contents."""
    shown = on_screen(widget)
    return WidgetObservation(
        kind=type(widget).__name__,
        text=display_text(widget) if shown else "",
        enabled=widget.isEnabled(),
        visible=shown,
        fully_visible=shown and widget.visibleRegion().contains(widget.rect()),
        object_name=widget.objectName(),
        tooltip=widget.toolTip() if shown else "",
        bounds=(
            widget.mapToGlobal(QPoint()).x(),
            widget.mapToGlobal(QPoint()).y(),
            widget.width(),
            widget.height(),
        )
        if shown
        else None,
        caption_fits=shown and _caption_fits(widget),
        checked=(
            widget.isChecked()
            if shown and isinstance(widget, QAbstractButton) and widget.isCheckable()
            else None
        ),
    )


def list_rows(window: QWidget) -> tuple[ListRowObservation, ...]:
    """Observe rows intersecting visible QListWidget viewports in this window."""
    rows: list[ListRowObservation] = []
    for listing in window.findChildren(QListWidget):
        if not on_screen(listing):
            continue
        viewport = listing.viewport()
        assert viewport is not None
        for index in range(listing.count()):
            item = listing.item(index)
            if item is not None and listing.visualItemRect(item).intersects(
                viewport.rect()
            ):
                rows.append(
                    ListRowObservation(item.text(), item.toolTip(), item.isSelected())
                )
    return tuple(rows)


def visible(window: QWidget) -> VisibleObservation:
    """Snapshot all visible widgets for completion predicates with changing captions."""
    return VisibleObservation(
        widgets=tuple(
            observe_widget(w)
            for w in [window, *window.findChildren(QWidget)]
            if on_screen(w)
        ),
        tree_rows=tuple(
            TreeRowObservation(
                tree.objectName(),
                path,
                collections.tree_columns(tree, item),
                item.isExpanded(),
            )
            for tree in window.findChildren(QTreeWidget)
            if on_screen(tree)
            for item, path in collections.visible_tree_items(tree)
        ),
    )


def status(window: QWidget) -> StatusObservation:
    """Read the one visible status bar's message or selectable direct labels."""
    bars = [bar for bar in window.findChildren(QStatusBar) if on_screen(bar)]
    bar = unique("status bar", bars, [bar.objectName() for bar in bars])
    message = bar.currentMessage()
    if not message:
        labels = bar.findChildren(
            QLabel, options=Qt.FindChildOption.FindDirectChildrenOnly
        )
        message = "\n".join(label.text() for label in labels if on_screen(label))
    return StatusObservation(message)


def dialogs(window: QWidget) -> tuple[DialogObservation, ...]:
    """List visible descendant dialogs and their visible, nonempty widget text."""
    return tuple(
        DialogObservation(
            dialog.windowTitle(),
            tuple(
                text
                for widget in dialog.findChildren(QWidget)
                if on_screen(widget) and (text := display_text(widget))
            ),
            _dialog_frame_within_screen(dialog),
            _dialog_height_fraction(dialog),
        )
        for dialog in window.findChildren(QDialog)
        if on_screen(dialog)
    )


def legends(window: QWidget) -> LegendObservation:
    """Read visible artist text only from visible matplotlib figure canvases."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    texts: list[str] = []
    for canvas in window.findChildren(FigureCanvasQTAgg):
        if not on_screen(canvas):
            continue
        for axes in canvas.figure.axes:
            legend = axes.get_legend()
            if axes.get_visible() and legend is not None and legend.get_visible():
                texts.extend(
                    t.get_text() for t in legend.get_texts() if t.get_visible()
                )
    return LegendObservation(tuple(texts))


def plots(window: QWidget) -> PlotObservation:
    """Read labels and visible stroked curves from shown matplotlib canvases.

    Hidden axes/canvases, transparent lines, empty paths and paths outside
    the displayed axes do not count. Counts describe public rendered artists,
    not acquisition results; use a PNG to review appearance and occlusion.
    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    observed: list[PlotAxesObservation] = []
    for canvas in window.findChildren(FigureCanvasQTAgg):
        if not on_screen(canvas):
            continue
        for axes in canvas.figure.axes:
            if not axes.get_visible():
                continue
            curves = sum(
                1
                for line in axes.lines
                if line.get_visible()
                and line.get_alpha() != 0
                and line.get_linewidth() > 0
                and line.get_linestyle() not in ("None", "", " ")
                and line.get_path()
                .transformed(line.get_transform())
                .intersects_bbox(axes.bbox, filled=False)
            )
            xmin, xmax = axes.get_xlim()
            ymin, ymax = axes.get_ylim()
            observed.append(
                PlotAxesObservation(
                    axes.get_xlabel()
                    if axes.xaxis.get_visible() and axes.xaxis.label.get_visible()
                    else "",
                    axes.get_ylabel()
                    if axes.yaxis.get_visible() and axes.yaxis.label.get_visible()
                    else "",
                    curves,
                    (float(xmin), float(xmax)),
                    (float(ymin), float(ymax)),
                    (
                        canvas.mapToGlobal(QPoint()).x()
                        + axes.bbox.x0 / canvas.device_pixel_ratio,
                        canvas.mapToGlobal(QPoint()).y()
                        + (canvas.figure.bbox.height - axes.bbox.y1)
                        / canvas.device_pixel_ratio,
                        axes.bbox.width / canvas.device_pixel_ratio,
                        axes.bbox.height / canvas.device_pixel_ratio,
                    ),
                    tuple(
                        t.get_text()
                        for t in axes.texts
                        if t.get_visible() and t.get_text()
                    ),
                )
            )
    return PlotObservation(tuple(observed))


def plot_label_bounds(window: QWidget) -> tuple[PlotLabelBoundsObservation, ...]:
    """Read renderer bounds for nonempty visible ylabels on shown canvases.

    Call after a screenshot or Qt paint has rendered the current geometry.
    Coordinates share the canvas renderer's pixel space, not screen pixels.
    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    observed: list[PlotLabelBoundsObservation] = []
    for canvas in window.findChildren(FigureCanvasQTAgg):
        if not on_screen(canvas):
            continue
        renderer = canvas.get_renderer()
        for axes in canvas.figure.axes:
            label = axes.yaxis.label
            if (
                axes.get_visible()
                and axes.yaxis.get_visible()
                and label.get_visible()
                and label.get_text()
            ):
                bounds = label.get_window_extent(renderer)
                observed.append(
                    PlotLabelBoundsObservation(
                        canvas.objectName(),
                        float(canvas.figure.bbox.x0),
                        float(axes.bbox.x0),
                        float(bounds.x0),
                        float(bounds.x1),
                        float(canvas.figure.bbox.x1),
                    )
                )
    return tuple(observed)


def screenshot(window: QWidget, path: Path) -> Path:
    """Grab the window to caller-chosen PNG path, creating parent directories.

    Visible dialogs are grabbed separately beside it as ``<stem>-dialog-N.png``
    because top-level dialog windows are not included in QWidget.grab().
    Owned top-level main windows are saved as ``<stem>-window-N.png``.
    The active visible popup belonging to this window is saved separately as
    ``<stem>-popup.png``. Other windows' popups are never captured.
    Raises OSError if Qt cannot save any image.
    """
    QApplication.processEvents()
    path.parent.mkdir(parents=True, exist_ok=True)
    widgets = [window, *window.findChildren(QDialog)]
    for index, widget in enumerate(w for w in widgets if w.isVisible()):
        output = path if index == 0 else path.with_stem(f"{path.stem}-dialog-{index}")
        if not widget.grab().save(str(output), "PNG"):
            raise OSError(f"Could not save screenshot: {output}")
    for index, owned_window in enumerate(window.findChildren(QMainWindow), 1):
        if owned_window.isVisible() and owned_window.isWindow():
            output = path.with_stem(f"{path.stem}-window-{index}")
            if not owned_window.grab().save(str(output), "PNG"):
                raise OSError(f"Could not save screenshot: {output}")
    popup = QApplication.activePopupWidget()
    if popup is not None and popup.isVisible():
        # isAncestorOf excludes child windows, so follow Qt's public parent chain.
        parent = popup.parentWidget()
        while parent is not None and parent is not window:
            parent = parent.parentWidget()
        if parent is window:
            output = path.with_stem(f"{path.stem}-popup")
            if not popup.grab().save(str(output), "PNG"):
                raise OSError(f"Could not save screenshot: {output}")
    return path


def _dialog_frame_within_screen(dialog: QDialog) -> bool:
    screen = dialog.screen()
    if screen is None:
        raise RuntimeError("Visible dialog has no screen")
    return screen.availableGeometry().contains(dialog.frameGeometry())


def _dialog_height_fraction(dialog: QDialog) -> float:
    screen = dialog.screen()
    if screen is None:
        raise RuntimeError("Visible dialog has no screen")
    return dialog.height() / screen.availableGeometry().height()
