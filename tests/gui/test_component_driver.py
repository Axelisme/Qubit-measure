"""The component driver's public seam uses only rendered widgets and input events."""

from collections.abc import Generator
from pathlib import Path

import pytest
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tests.gui import _user_input


@pytest.fixture
def surface(qapp) -> Generator[QMainWindow]:
    window = QMainWindow()
    window.resize(640, 480)
    tabs = QTabWidget()
    window.setCentralWidget(tabs)
    page = QWidget()
    layout = QFormLayout(page)
    entry = QLineEdit("original")
    entry.setObjectName("sample_name")
    layout.addRow("Sample name:", entry)
    status = QLabel("Ready")
    window.statusBar().addWidget(status)
    entry.editingFinished.connect(lambda: status.setText(f"Committed: {entry.text()}"))
    for name in ("first", "second"):
        button = QPushButton("Duplicate")
        button.setObjectName(name)
        layout.addRow(button)
    disabled = QPushButton("Unavailable")
    disabled.setEnabled(False)
    disabled.setToolTip("No result yet")
    disabled.clicked.connect(lambda: status.setText("Disabled action fired"))
    layout.addRow(disabled)
    hidden = QPushButton("Hidden action")
    hidden.setObjectName("hidden_action")
    hidden.clicked.connect(lambda: status.setText("Hidden action fired"))
    layout.addRow(hidden)
    hidden.hide()
    tabs.addTab(page, "Run")
    data = QWidget()
    data_layout = QVBoxLayout(data)
    data_layout.addWidget(QLabel("Data preview"))
    tabs.addTab(data, "Data")
    window.show()
    qapp.processEvents()
    yield window
    window.deleteLater()
    qapp.processEvents()


def test_lookup_reports_missing_and_ambiguous_candidates(surface) -> None:
    driver = _user_input.ComponentDriver(surface)
    with pytest.raises(LookupError, match="Missing.*0 candidates"):
        driver.observe("Missing")
    with pytest.raises(LookupError, match="Duplicate.*2 candidates.*first.*second"):
        driver.click("Duplicate")
    assert driver.observe(object_name="first").text == "Duplicate"


def test_label_input_enter_commits_and_tab_click_changes_visible_page(surface) -> None:
    driver = _user_input.ComponentDriver(surface)
    driver.enter_text("Sample name:", "changed")
    assert driver.observe(object_name="sample_name").text == "changed"
    assert driver.status().text == "Committed: changed"
    driver.select_tab("Data")
    assert driver.tabs("Data").current == "Data"
    assert driver.observe("Data preview").visible
    assert not driver.observe(object_name="sample_name").visible
    driver.select_tab("Run")
    assert driver.tabs("Run").current == "Run"
    with pytest.raises(LookupError, match="Unknown.*0 candidates"):
        driver.select_tab("Unknown")


def test_disabled_and_hidden_actions_leave_visible_status_unchanged(
    surface, tmp_path: Path
) -> None:
    driver = _user_input.ComponentDriver(surface)
    driver.screenshot(tmp_path / "01-before.png")
    button = driver.observe("Unavailable")
    assert button.visible and not button.enabled
    assert button.tooltip == "No result yet"
    driver.click("Unavailable")
    driver.screenshot(tmp_path / "02-disabled.png")
    assert driver.status().text == "Ready"
    assert not driver.observe(object_name="hidden_action").visible
    driver.click(object_name="hidden_action")
    driver.screenshot(tmp_path / "03-hidden.png")
    assert driver.status().text == "Ready"


def test_visible_dialog_text_and_caller_chosen_png_path(surface, qapp, tmp_path) -> None:
    dialog = QDialog(surface)
    dialog.setWindowTitle("Confirmation")
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("Keep this measurement?"))
    close = QPushButton("Dismiss")
    close.clicked.connect(dialog.reject)
    layout.addWidget(close)
    dialog.open()
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    try:
        (observation,) = driver.dialogs()
        assert observation.title == "Confirmation"
        assert "Keep this measurement?" in observation.texts
        path = driver.screenshot(tmp_path / "nested" / "confirmation.png")
        assert path == tmp_path / "nested" / "confirmation.png"
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        driver.click("Dismiss")
        assert driver.dialogs() == ()
    finally:
        dialog.deleteLater()


def test_only_visible_legend_text_is_observed(surface, qapp) -> None:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    figure = Figure()
    axes = figure.subplots()
    axes.plot([0, 1], [1, 2], label="Measured signal")
    axes.legend()
    canvas = FigureCanvasQTAgg(figure)
    surface.setCentralWidget(canvas)
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    assert driver.legends().texts == ("Measured signal",)
    canvas.hide()
    assert driver.legends().texts == ()
