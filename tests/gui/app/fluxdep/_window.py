"""Public Qt seams of the fluxdep MainWindow, located by objectName.

UI tests use these helpers instead of button text or private attributes.
Each lookup asserts that the seam exists, so a missing seam fails as an
assertion rather than as an attribute error.
"""

from __future__ import annotations

from typing import Literal

from qtpy.QtWidgets import (
    QLabel,
    QListWidget,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QWidget,
)
from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget

Page = Literal["spectra", "fit"]


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
