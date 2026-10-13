"""Public Qt seams of the fluxdep MainWindow, its pickers and Fit page, by objectName.

UI tests use these helpers instead of button text or private attributes.
Each lookup asserts that the seam exists, so a missing seam fails as an
assertion rather than as an attribute error.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeVar

from matplotlib.figure import Figure
from qtpy.QtCore import QMetaObject, QPoint, Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QWidget,
)
from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget

Page = Literal["spectra", "fit"]
W = TypeVar("W", bound=QWidget)


def named(root: QWidget, kind: type[W], name: str) -> W:
    """The descendant of root of type kind whose objectName is name."""
    found = root.findChild(kind, name)
    assert found is not None, f"missing {kind.__name__} {name!r}"
    return found


def command(window: QWidget, name: str) -> QPushButton:
    """The window command button whose objectName is name."""
    found = window.findChild(QPushButton, name)
    assert found is not None, f"missing command {name!r}"
    return found


def spectra_list(window: QWidget) -> QListWidget:
    """The spectrum list on the Spectra page."""
    found = window.findChild(QListWidget, "fluxdep.spectra")
    assert found is not None, "missing spectrum list"
    return found


def list_labels(window: QWidget) -> list[str]:
    """Visible row texts of the spectrum list, in display order."""
    spectra = spectra_list(window)
    items = [spectra.item(row) for row in range(spectra.count())]
    return [item.text() for item in items if item is not None]


def project_label(window: QWidget) -> QLabel:
    """The top-bar label that names the current project."""
    found = window.findChild(QLabel, "fluxdep.session.project")
    assert found is not None, "missing project label"
    return found


def pages(window: QWidget) -> QTabWidget:
    """The main-area tabs that hold the Spectra and Fit pages."""
    found = window.findChild(QTabWidget, "fluxdep.pages")
    assert found is not None, "missing page tabs"
    return found


def page(window: QWidget, name: Page) -> QWidget:
    """The Spectra or Fit page widget."""
    found = window.findChild(QWidget, f"fluxdep.page.{name}")
    assert found is not None, f"missing {name} page"
    return found


def show_page(window: QWidget, name: Page) -> None:
    """Select a page the way a user clicks its tab."""
    pages(window).setCurrentWidget(page(window, name))


def current_page(window: QWidget) -> Page:
    """The page the window currently shows."""
    shown = pages(window).currentWidget()
    assert shown is not None
    if shown is page(window, "fit"):
        return "fit"
    assert shown is page(window, "spectra")
    return "spectra"


def editor(window: QWidget) -> QWidget:
    """The widget the Spectra page editing area currently shows."""
    stack = window.findChild(QStackedWidget, "fluxdep.editor")
    assert stack is not None, "missing editing area"
    shown = stack.currentWidget()
    assert shown is not None
    return shown


def fit_panel(window: QWidget) -> AnalyzePanelWidget | None:
    """The analysis panel hosted by the Fit page, or None before first use."""
    return page(window, "fit").findChild(AnalyzePanelWidget)


def picker_control[W: QWidget](root: QWidget, action: str, kind: type[W]) -> W:
    """The picker control fluxdep.picker.<action> of the given Qt type."""
    found = root.findChild(kind, f"fluxdep.picker.{action}")
    assert found is not None, f"missing picker control {action!r}"
    return found


def picker_button(root: QWidget, action: str) -> QPushButton:
    """The picker button fluxdep.picker.<action> under root."""
    return picker_control(root, action, QPushButton)


def has_picker_control(root: QWidget, action: str) -> bool:
    """Whether root currently holds a fluxdep.picker.<action> control."""
    return bool(root.findChildren(QWidget, f"fluxdep.picker.{action}"))


def column_tops(root: QWidget, actions: list[str]) -> list[int]:
    """Top y of each named picker control within a shown root, in given order."""
    return [
        picker_control(root, action, QWidget).mapTo(root, QPoint(0, 0)).y()
        for action in actions
    ]


def unlabelled_inputs(root: QWidget) -> list[str]:
    """objectNames of sliders and drop-downs that no QLabel names as its buddy."""
    named = [label.buddy() for label in root.findChildren(QLabel)]
    return [
        control.objectName()
        for control in root.findChildren(QWidget)
        if isinstance(control, (QSlider, QComboBox))
        and not any(buddy is control for buddy in named)
    ]


def background_cmaps(figure: Figure) -> list[str]:
    """Colormap of the first (background) image of every axes that has one."""
    return [
        axes.get_images()[0].get_cmap().name
        for axes in figure.axes
        if axes.get_images()
    ]


def preview_button(root: QWidget, action: str) -> QPushButton:
    """The result-preview next-step button fluxdep.preview.<action>."""
    found = root.findChild(QPushButton, f"fluxdep.preview.{action}")
    assert found is not None, f"missing preview action {action!r}"
    return found


def preview_actions(root: QWidget) -> set[str]:
    """Next-step actions the result preview under root offers."""
    prefix = "fluxdep.preview."
    return {
        button.objectName().removeprefix(prefix)
        for button in root.findChildren(QPushButton)
        if button.objectName().startswith(prefix)
    }


def choose_file(path: Path | str, activate: Callable[[], None], *, cancel=False):
    """Accept or cancel the real Local file chooser that activate() opens."""
    errors: list[Exception] = []
    seen: list[str] = []

    def respond() -> None:
        chooser = QApplication.activeModalWidget()
        try:
            assert isinstance(chooser, QFileDialog)
            seen.append(chooser.windowTitle())
            chooser.selectFile(str(path))
            QMetaObject.invokeMethod(
                chooser,
                "reject" if cancel else "accept",
                Qt.ConnectionType.DirectConnection,
            )
        except (AssertionError, RuntimeError, TypeError) as exc:
            errors.append(exc)
            if isinstance(chooser, QDialog):
                chooser.reject()

    def abort() -> None:
        chooser = QApplication.activeModalWidget()
        errors.append(AssertionError("Static chooser did not finish within 2 seconds"))
        if isinstance(chooser, QDialog):
            chooser.reject()

    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(abort)
    timeout.start(2000)
    QTimer.singleShot(0, respond)
    activate()
    timeout.stop()
    if errors:
        raise errors[0]
    assert seen, "The action must present a real modal file chooser"
