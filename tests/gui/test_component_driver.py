"""The component driver's public seam uses only rendered widgets and input events."""

from collections.abc import Generator
from pathlib import Path

import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tests.gui import _user_input
from tests.gui._qt_waits import qt_until


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
    status_bar = QStatusBar()
    window.setStatusBar(status_bar)
    status_bar.addWidget(status)
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


def test_visible_dialog_text_and_caller_chosen_png_path(
    surface, qapp, tmp_path
) -> None:
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
        dialog_path = path.with_stem(f"{path.stem}-dialog-1")
        assert dialog_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        driver.click("Dismiss")
        assert driver.dialogs() == ()
    finally:
        dialog.deleteLater()


def test_visible_snapshot_distinguishes_running_button_from_caption(
    surface, qapp
) -> None:
    page = surface.centralWidget().currentWidget()
    button = QPushButton("Run")
    page.layout().addRow(button)
    button.clicked.connect(lambda: button.setText("Stop"))
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    driver.click("Run")
    running = driver.visible()
    assert not any(w.kind == "QPushButton" and w.text == "Run" for w in running.widgets)
    assert any(w.kind == "QPushButton" and w.text == "Stop" for w in running.widgets)
    assert not any(w.object_name == "hidden_action" for w in running.widgets)
    QTimer.singleShot(0, lambda: button.setText("Run"))
    # The queued synthetic terminal callback restores Run on the next owner turn.
    qt_until(
        lambda: any(
            w.kind == "QPushButton" and w.text == "Run" and w.enabled and w.visible
            for w in driver.visible().widgets
        ),
        label="synthetic button terminal caption",
    )
    assert driver.observe("Run").enabled


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


def test_disabled_or_hidden_input_cannot_commit_and_tab_availability_is_respected(
    surface, qapp
) -> None:
    entry = surface.findChild(QLineEdit, "sample_name")
    entry.setEnabled(False)
    driver = _user_input.ComponentDriver(surface)
    driver.enter_text(object_name="sample_name", value="blocked")
    assert driver.status().text == "Ready"
    assert driver.observe(object_name="sample_name").text == "original"
    entry.setEnabled(True)
    entry.hide()
    driver.enter_text(object_name="sample_name", value="hidden")
    assert driver.status().text == "Ready"
    tabs = surface.centralWidget()
    tabs.setTabEnabled(1, False)
    driver.select_tab("Data")
    assert driver.tabs("Run").current == "Run"
    tabs.setTabEnabled(1, True)
    tabs.setTabVisible(1, False)
    driver.select_tab("Data")
    assert driver.tabs("Run").current == "Run"
    tabs.setTabVisible(1, True)
    tabs.addTab(QWidget(), "Data")
    qapp.processEvents()
    with pytest.raises(LookupError, match="Data.*2 candidates.*1.*2"):
        driver.select_tab("Data")


def test_blocking_menu_selection_uses_visible_nested_captions(surface, qapp) -> None:
    button = QPushButton("Open menu", surface.centralWidget())
    button.show()
    button.move(300, 120)
    menu = QMenu(surface)
    group = menu.addMenu("Group")
    assert group is not None
    nested = group.addAction("Open")
    root = menu.addAction("Open")
    assert nested is not None and root is not None
    nested.triggered.connect(
        lambda: surface.statusBar().showMessage("Selected nested Open")
    )
    root.triggered.connect(lambda: surface.statusBar().showMessage("Wrong root Open"))
    selected: list[str] = []

    def open_menu() -> None:
        action = menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        if action is not None:
            selected.append(action.text())

    button.clicked.connect(open_menu)
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    assert callable(getattr(driver, "select_menu", None)), (
        "ComponentDriver must deliver a caption path inside a blocking QMenu"
    )
    driver.select_menu("Open menu", ("Group", "Open"))
    assert selected == ["Open"]
    assert driver.status().text == "Selected nested Open"
    assert not menu.isVisible()


@pytest.fixture
def tree_surface(surface, qapp):
    from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem

    tree = QTreeWidget()
    tree.setObjectName("parameters")
    tree.setHeaderLabels(["Field", "Value", "Unit"])
    root = QTreeWidgetItem(tree, ["general", "general", ""])
    QTreeWidgetItem(root, ["peak", "1.25", "MHz"])
    hidden = QTreeWidgetItem(tree, ["hidden", "secret", ""])
    hidden.setHidden(True)
    surface.setCentralWidget(tree)
    qapp.processEvents()
    return surface, tree


def test_visible_tree_rows_and_real_expand_reveal_displayed_values(
    tree_surface,
) -> None:
    window, _ = tree_surface
    driver = _user_input.ComponentDriver(window)
    before = driver.visible().tree_rows
    assert [(r.tree_name, r.path, r.columns, r.expanded) for r in before] == [
        ("parameters", ("general",), ("general", "general", ""), False)
    ]
    driver.expand_tree(("general",), object_name="parameters")
    rows = driver.visible().tree_rows
    assert [(r.path, r.columns, r.expanded) for r in rows] == [
        (("general",), ("general", "general", ""), True),
        (("general", "peak"), ("peak", "1.25", "MHz"), False),
    ]


def test_tree_expand_refuses_disabled_hidden_and_collapsed_descendants(
    tree_surface,
) -> None:
    window, tree = tree_surface
    driver = _user_input.ComponentDriver(window)
    with pytest.raises(LookupError, match="peak.*0 candidates"):
        driver.expand_tree(("general", "peak"), object_name="parameters")
    tree.setEnabled(False)
    driver.expand_tree(("general",), object_name="parameters")
    assert not driver.visible().tree_rows[0].expanded
    tree.setEnabled(True)
    tree.hide()
    driver.expand_tree(("general",), object_name="parameters")
    assert driver.visible().tree_rows == ()
    tree.show()
    assert not driver.visible().tree_rows[0].expanded


def test_tree_expand_reports_missing_and_ambiguous_visible_paths(
    tree_surface, qapp
) -> None:
    from PyQt6.QtWidgets import QTreeWidgetItem

    window, tree = tree_surface
    QTreeWidgetItem(tree, ["general", "duplicate", ""])
    qapp.processEvents()
    driver = _user_input.ComponentDriver(window)
    with pytest.raises(LookupError, match="general.*2 candidates.*duplicate"):
        driver.expand_tree(("general",), object_name="parameters")
    with pytest.raises(LookupError, match="missing.*0 candidates"):
        driver.expand_tree(("missing",), object_name="parameters")


def test_wide_checkbox_click_reaches_indicator_and_reports_visible_status(
    surface, qapp
) -> None:
    from PyQt6.QtWidgets import QCheckBox

    checkbox = QCheckBox("Use Simulate Env")
    checkbox.setObjectName("simulate")
    surface.centralWidget().currentWidget().layout().addRow(checkbox)
    checkbox.toggled.connect(
        lambda checked: surface.statusBar().showMessage("on" if checked else "off")
    )
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    assert driver.observe("Use Simulate Env").visible
    assert driver.observe("Use Simulate Env").enabled
    assert driver.dialogs() == ()
    driver.click("Use Simulate Env")
    assert driver.status().text == "on"


def test_checkbox_snapshot_tracks_binary_clicks_and_disabled_hidden_noops(
    surface, qapp
) -> None:
    from PyQt6.QtWidgets import QCheckBox

    checkbox = QCheckBox("Use Simulate Env")
    surface.centralWidget().currentWidget().layout().addRow(checkbox)
    checkbox.toggled.connect(
        lambda checked: surface.statusBar().showMessage("on" if checked else "off")
    )
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    assert driver.observe("Use Simulate Env").checked is False
    driver.click("Use Simulate Env")
    assert driver.status().text == "on"
    assert driver.observe("Use Simulate Env").checked is True
    driver.click("Use Simulate Env")
    assert driver.status().text == "off"
    assert driver.observe("Use Simulate Env").checked is False
    checkbox.setEnabled(False)
    driver.click("Use Simulate Env")
    assert driver.status().text == "off"
    assert driver.observe("Use Simulate Env").checked is False
    checkbox.setObjectName("simulate")
    checkbox.hide()
    driver.click(object_name="simulate")
    assert driver.status().text == "off"
    assert driver.observe(object_name="simulate").checked is None
    assert driver.observe("Unavailable").checked is None


def test_tree_snapshot_does_not_expose_hidden_column_text(tree_surface, qapp) -> None:
    window, tree = tree_surface
    tree.setColumnHidden(1, True)
    qapp.processEvents()
    driver = _user_input.ComponentDriver(window)
    assert driver.visible().tree_rows[0].columns == ("general", "", "")


def test_plot_axes_and_curves_observe_rendered_artists_only(surface, qapp) -> None:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    figure = Figure()
    axes = figure.subplots()
    axes.set(xlabel="Frequency (MHz)", ylabel="Amplitude", xlim=(0, 1), ylim=(0, 2))
    axes.plot([0, 1], [0.5, 1.5])  # Raw plots need not have a legend.
    axes.plot([0, 1], [1, 1], visible=False)
    axes.plot([0, 1], [1, 1], alpha=0)
    axes.plot([2, 3], [1, 1])  # Outside the displayed axes.
    axes.plot([], [])
    canvas = FigureCanvasQTAgg(figure)
    surface.setCentralWidget(canvas)
    canvas.draw()
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    assert callable(getattr(driver, "plots", None)), (
        "Driver must observe raw plot axes and visible curves"
    )
    observed = driver.plots().axes
    assert len(observed) == 1
    assert observed[0].xlabel == "Frequency (MHz)"
    assert observed[0].ylabel == "Amplitude"
    assert observed[0].visible_curves == 1
    assert driver.legends().texts == ()
    axes.xaxis.label.set_visible(False)
    assert driver.plots().axes[0].xlabel == ""
    axes.set_visible(False)
    assert driver.plots().axes == ()
    axes.set_visible(True)
    canvas.hide()
    assert driver.plots().axes == ()


def test_file_chooser_enters_existing_path_through_real_modal_events(surface, tmp_path) -> None:
    native = tmp_path / "saved signal.h5"
    native.write_bytes(b"synthetic native file")
    button = QPushButton("Choose file")
    surface.setCentralWidget(button)

    def choose() -> None:
        filename, _ = QFileDialog.getOpenFileName(
            surface, "Open native data", str(tmp_path), "Data (*.h5)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        surface.statusBar().showMessage(f"Selected {filename}")

    button.clicked.connect(choose)
    driver = _user_input.ComponentDriver(surface)
    assert callable(getattr(driver, "choose_file", None)), "Driver must select files using real modal input"
    driver.choose_file("Choose file", native)
    assert driver.dialogs() == ()
    assert driver.status().text == f"Selected {native}"
    button.setEnabled(False)
    driver.choose_file("Choose file", native)
    assert driver.status().text == f"Selected {native}"
    assert driver.dialogs() == ()
