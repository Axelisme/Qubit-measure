"""The component driver's public seam uses only rendered widgets and input events."""

from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtGui import QAction, QImage
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
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


def test_list_double_click_activates_only_unique_visible_enabled_rows(
    surface, qapp
) -> None:
    listing = QListWidget(surface)
    listing.setGeometry(10, 80, 220, 90)
    listing.addItems(["Ready", "Duplicate row", "Duplicate row"] + ["Below fold"] * 10)
    activated = QLabel("Not activated", surface)
    activated.setObjectName("activation")
    activated.setGeometry(240, 80, 200, 30)
    listing.itemDoubleClicked.connect(lambda item: activated.setText(item.text()))
    listing.show()
    activated.show()
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    driver.double_click_list_row("Ready")
    assert driver.observe(object_name="activation").text == "Ready"
    for caption in ("Duplicate row", "Below fold", "Missing"):
        with pytest.raises(LookupError):
            driver.double_click_list_row(caption)
    listing.setEnabled(False)
    with pytest.raises(LookupError):
        driver.double_click_list_row("Ready")
    listing.hide()
    with pytest.raises(LookupError):
        driver.double_click_list_row("Ready")
    assert driver.observe(object_name="activation").text == "Ready"


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


def test_enter_empty_text_clears_a_filled_field(surface, tmp_path: Path) -> None:
    driver = _user_input.ComponentDriver(surface)
    driver.screenshot(tmp_path / "before-clear.png")
    driver.enter_text("Sample name:", "")
    driver.screenshot(tmp_path / "after-clear.png")
    assert driver.observe(object_name="sample_name").text == ""
    assert driver.status().text == "Committed: "


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
    axes.set_visible(False)
    canvas.draw()
    assert driver.legends().texts == ()
    axes.set_visible(True)
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


@pytest.fixture
def chooser_directory(monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    """Keep chooser files visible and resolve relative input from its start directory."""
    with (
        TemporaryDirectory(prefix="chooser-", dir=Path.cwd()) as directory,
        monkeypatch.context() as context,
    ):
        context.chdir(directory)
        yield Path(directory)


def test_file_chooser_enters_existing_path_through_real_modal_events(
    surface, chooser_directory: Path, qapp
) -> None:
    native = Path("saved signal.h5")
    native.write_bytes(b"synthetic native file")
    button = QPushButton("Choose file")
    surface.setCentralWidget(button)
    qapp.processEvents()

    def choose() -> None:
        filename, _ = QFileDialog.getOpenFileName(
            surface,
            "Open native data",
            str(chooser_directory),
            "Data (*.h5)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        surface.statusBar().showMessage(f"Selected {filename}")

    button.clicked.connect(choose)
    driver = _user_input.ComponentDriver(surface)
    hidden_copy = QPushButton("Choose file", surface)
    hidden_copy.hide()
    driver.choose_file("Choose file", native)
    assert driver.dialogs() == ()
    assert driver.status().text == f"Selected {native.resolve()}"
    button.setEnabled(False)
    driver.choose_file("Choose file", native)
    assert driver.status().text == f"Selected {native.resolve()}"
    assert driver.dialogs() == ()


def test_viewport_clipped_widgets_cannot_be_observed_or_operated(surface, qapp) -> None:
    scroll = QScrollArea()
    content = QWidget()
    content.setFixedSize(500, 1200)
    entry = QLineEdit("retained original", content)
    entry.setObjectName("clipped_input")
    entry.setToolTip("Offscreen help")
    entry.setGeometry(10, 1000, 250, 30)
    entry.editingFinished.connect(
        lambda: surface.statusBar().showMessage("Input committed")
    )
    button = QPushButton("Offscreen action", content)
    button.setObjectName("clipped_action")
    button.setGeometry(10, 1040, 250, 30)
    button.clicked.connect(lambda: surface.statusBar().showMessage("Action fired"))
    scroll.setWidget(content)
    surface.setCentralWidget(scroll)
    surface.statusBar().showMessage("Unchanged")
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    observed = driver.observe(object_name="clipped_input")
    assert observed.visible is False, "A viewport-clipped field is not on screen"
    assert observed.text == "" and observed.tooltip == ""
    assert not driver.observe(object_name="clipped_action").visible
    assert not any(
        w.object_name.startswith("clipped_") for w in driver.visible().widgets
    )
    with pytest.raises(LookupError, match="0 candidates"):
        driver.click("Offscreen action")
    driver.click(object_name="clipped_action")
    driver.enter_text(object_name="clipped_input", value="must not commit")
    assert driver.status().text == "Unchanged"
    # Reveal the retained synthetic field to check no hidden edit was performed.
    bar = scroll.verticalScrollBar()
    if bar is None:
        raise RuntimeError("Synthetic scroll area has no vertical scrollbar")
    bar.setValue(bar.maximum())
    qapp.processEvents()
    assert driver.observe(object_name="clipped_input").visible
    assert driver.observe(object_name="clipped_input").text == "retained original"
    driver.enter_text(object_name="clipped_input", value="visible edit")
    assert driver.status().text == "Input committed"
    assert driver.observe(object_name="clipped_input").text == "visible edit"
    driver.click("Offscreen action")
    assert driver.status().text == "Action fired"


def test_file_chooser_saves_png_before_real_selection_closes_modal(
    surface, tmp_path, chooser_directory: Path, qapp
) -> None:
    native = Path("native.h5")
    native.write_bytes(b"synthetic native file")
    button = QPushButton("Choose with evidence")
    surface.setCentralWidget(button)
    # This external chooser is deliberately larger than the owner. A grab of the
    # owner after acceptance cannot satisfy the image-size assertions below.
    dialog = QFileDialog(
        surface, "Open native data", str(chooser_directory), "Data (*.h5)"
    )
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
    dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    dialog.setFixedSize(900, 600)
    qapp.processEvents()

    def choose() -> None:
        if dialog.exec() == QDialog.DialogCode.Accepted:
            surface.statusBar().showMessage(f"Selected {dialog.selectedFiles()[0]}")

    button.clicked.connect(choose)
    driver = _user_input.ComponentDriver(surface)
    owner_shot = driver.screenshot(tmp_path / "owner.png")
    shot = tmp_path / "evidence" / "chooser.png"
    driver.choose_file("Choose with evidence", native, evidence=shot)
    assert shot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    image = QImage(str(shot))
    assert image.size() == dialog.size() * dialog.devicePixelRatioF()
    assert image.size() != QImage(str(owner_shot)).size()
    assert driver.dialogs() == ()
    assert driver.status().text == f"Selected {native.resolve()}"


@pytest.mark.parametrize(
    "field_visible", [True, False], ids=["field-visible", "value-visible"]
)
def test_tree_snapshot_redacts_cells_outside_horizontal_viewport(
    tree_surface, qapp, field_visible: bool
) -> None:
    from PyQt6.QtWidgets import QAbstractItemView

    window, tree = tree_surface
    tree.header().setStretchLastSection(False)
    tree.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    for column in range(3):
        tree.setColumnWidth(column, 900)
    root = tree.topLevelItem(0)
    root.setExpanded(True)
    tree.horizontalScrollBar().setValue(0 if field_visible else 900)
    qapp.processEvents()
    driver = _user_input.ComponentDriver(window)
    rows = driver.visible().tree_rows
    if field_visible:
        assert [(r.path, r.columns) for r in rows] == [
            (("general",), ("general", "", "")),
            (("general", "peak"), ("peak", "", "")),
        ]
    else:
        assert [(r.path, r.columns) for r in rows] == [
            (("",), ("", "general", "")),
            (("", ""), ("", "1.25", "")),
        ]
        with pytest.raises(LookupError, match="general.*0 candidates"):
            driver.expand_tree(("general",), object_name="parameters")
        # Even an exact redacted path cannot reach an offscreen expansion control.
        root.setExpanded(False)
        driver.expand_tree(("",), object_name="parameters")
        assert not driver.visible().tree_rows[0].expanded


@pytest.mark.parametrize("bad_path_kind", ["missing", "directory"])
@pytest.mark.parametrize("opener_state", ["enabled", "disabled", "hidden"])
def test_file_chooser_rejects_bad_path_before_opener_side_effects(
    surface, tmp_path: Path, qapp, bad_path_kind: str, opener_state: str
) -> None:
    path = tmp_path / bad_path_kind
    expected: type[Exception] = FileNotFoundError
    if bad_path_kind == "directory":
        path.mkdir()
        expected = ValueError
    button = QPushButton("Safe synthetic opener")
    surface.setCentralWidget(button)
    clicked: list[bool] = []
    # A click has no modal side effect, so the old implementation cannot hang.
    button.clicked.connect(lambda: clicked.append(True))
    button.setEnabled(opener_state != "disabled")
    button.setVisible(opener_state != "hidden")
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    errors: list[Exception] = []
    try:
        driver.choose_file("Safe synthetic opener", path)
    except (LookupError, OSError, RuntimeError, ValueError) as error:
        errors.append(error)
    assert len(errors) == 1 and type(errors[0]) is expected
    assert clicked == []
    assert driver.dialogs() == ()


@pytest.mark.parametrize("opener_state", ["disabled", "hidden", "clipped"])
def test_file_chooser_valid_path_unavailable_opener_is_noop(
    surface, tmp_path: Path, chooser_directory: Path, qapp, opener_state: str
) -> None:
    native = Path("native.h5")
    native.write_bytes(b"synthetic")
    scroll = QScrollArea()
    content = QWidget()
    content.setFixedSize(500, 1200)
    button = QPushButton("Unavailable chooser", content)
    button.move(10, 1000 if opener_state == "clipped" else 10)
    button.setEnabled(opener_state != "disabled")
    button.setVisible(opener_state != "hidden")
    clicked: list[bool] = []
    button.clicked.connect(lambda: clicked.append(True))
    scroll.setWidget(content)
    surface.setCentralWidget(scroll)
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    shot = tmp_path / "must-not-exist.png"
    errors: list[Exception] = []
    try:
        driver.choose_file("Unavailable chooser", native, evidence=shot)
    except (LookupError, OSError, RuntimeError, ValueError) as error:
        errors.append(error)
    assert errors == []
    assert clicked == []
    assert driver.dialogs() == ()
    assert not shot.exists()


@pytest.fixture
def choice_surface(qapp) -> Generator[QWidget]:
    window = QWidget()
    layout = QFormLayout(window)
    combo = QComboBox()
    combo.setObjectName("choice")
    combo.addItems(["One", "Two", "Three"])
    status = QLabel("Ready")
    status.setObjectName("choice_status")
    combo.currentTextChanged.connect(lambda text: status.setText(text))
    action = QAction("Copy choice", combo)
    action.triggered.connect(lambda: status.setText("Copied"))
    combo.addAction(action)
    combo.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
    layout.addRow("Selection:", combo)
    layout.addRow(status)
    window.show()
    qapp.processEvents()
    yield window
    window.deleteLater()
    qapp.processEvents()


def test_choose_option_interacts_with_active_modal_popup_but_not_blocked_window(
    choice_surface, qapp, tmp_path: Path
) -> None:
    dialog = QDialog(choice_surface)
    dialog.setWindowTitle("Modal selection")
    layout = QVBoxLayout(dialog)
    combo = QComboBox()
    combo.setObjectName("modal_choice")
    combo.addItems(["One", "Two", "Three"])
    layout.addWidget(combo)
    dismiss = QPushButton("Dismiss")
    dismiss.clicked.connect(dialog.reject)
    layout.addWidget(dismiss)
    dialog.open()
    qapp.processEvents()
    driver = _user_input.ComponentDriver(choice_surface)
    try:
        driver.screenshot(tmp_path / "modal-before.png")
        driver.choose_option(object_name="choice", value="Two")
        assert driver.observe(object_name="choice").text == "One"
        driver.choose_option(object_name="modal_choice", value="Three")
        driver.screenshot(tmp_path / "modal-after.png")
        assert driver.observe(object_name="modal_choice").text == "Three"
        driver.click("Dismiss")
        driver.choose_option(object_name="choice", value="Two")
        assert driver.observe(object_name="choice").text == "Two"
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_choose_option_selects_existing_item_in_editable_combo(
    choice_surface, qapp, tmp_path: Path
) -> None:
    combo = choice_surface.findChild(QComboBox, "choice")
    combo.setEditable(True)
    combo.setCurrentText("")
    qapp.processEvents()
    driver = _user_input.ComponentDriver(choice_surface)
    driver.screenshot(tmp_path / "01-empty-editable-choice.png")
    assert driver.observe(object_name="choice").text == ""

    driver.choose_option("Selection:", "Three")
    driver.screenshot(tmp_path / "02-selected-existing-option.png")
    assert driver.observe(object_name="choice").text == "Three"
    assert driver.observe(object_name="choice_status").text == "Three"

    driver.choose_option(object_name="choice", value="One")
    driver.screenshot(tmp_path / "03-changed-existing-option.png")
    assert driver.observe(object_name="choice").text == "One"
    assert driver.observe(object_name="choice_status").text == "One"


def test_choose_option_and_context_action_use_visible_captions(choice_surface) -> None:
    driver = _user_input.ComponentDriver(choice_surface)
    driver.choose_option("Selection:", "Three")
    assert driver.observe(object_name="choice").text == "Three"
    assert driver.observe(object_name="choice_status").text == "Three"
    driver.invoke_context_action(object_name="choice", action="Copy choice")
    assert driver.observe(object_name="choice_status").text == "Copied"
    with pytest.raises(LookupError):
        driver.choose_option(object_name="choice", value="Missing")
    with pytest.raises(LookupError):
        driver.invoke_context_action(object_name="choice", action="Missing")
    assert driver.dialogs() == ()
    driver.choose_option(object_name="choice", value="Two")
    assert driver.observe(object_name="choice_status").text == "Two"


def test_choice_operations_reject_wrong_selector_and_ignore_unavailable_widgets(
    choice_surface, qapp
) -> None:
    driver = _user_input.ComponentDriver(choice_surface)
    with pytest.raises(TypeError):
        driver.choose_option(object_name="choice_status", value="Two")
    with pytest.raises(ValueError, match="exactly one"):
        driver.choose_option(value="Two")
    with pytest.raises(LookupError):
        driver.invoke_context_action(object_name="missing", action="Copy choice")
    combo = choice_surface.findChild(QComboBox, "choice")
    combo.setEnabled(False)
    driver.choose_option(object_name="choice", value="Two")
    driver.invoke_context_action(object_name="choice", action="Copy choice")
    assert driver.observe(object_name="choice_status").text == "Ready"
    combo.setEnabled(True)
    combo.hide()
    qapp.processEvents()
    driver.choose_option(object_name="choice", value="Two")
    driver.invoke_context_action(object_name="choice", action="Copy choice")
    assert driver.observe(object_name="choice_status").text == "Ready"


class _BlockComboOpening(QObject):
    """Consume activation clicks while leaving the synthetic combo readable."""

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        if a1 is not None and a1.type() in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
        ):
            return True
        return super().eventFilter(a0, a1)


@pytest.mark.parametrize("blockade", ["opening", "disabled_popup"])
def test_choose_option_leaves_selection_unchanged_without_interactable_popup(
    choice_surface, blockade: str, tmp_path: Path
) -> None:
    driver = _user_input.ComponentDriver(choice_surface)
    combo = choice_surface.findChild(QComboBox, "choice")
    opening_filter = _BlockComboOpening(combo)
    view = combo.view()
    assert view is not None
    if blockade == "opening":
        combo.installEventFilter(opening_filter)
    else:
        view.setEnabled(False)
    driver.screenshot(tmp_path / "before-blocked-selection.png")
    assert driver.observe(object_name="choice").enabled
    assert driver.observe(object_name="choice").visible

    driver.choose_option("Selection:", "Three")
    driver.screenshot(tmp_path / "after-blocked-selection.png")
    assert driver.observe(object_name="choice").text == "One"
    assert driver.observe(object_name="choice_status").text == "Ready"

    combo.removeEventFilter(opening_filter)
    view.setEnabled(True)
    driver.choose_option("Selection:", "Three")
    driver.screenshot(tmp_path / "after-restored-selection.png")
    assert driver.observe(object_name="choice").text == "Three"
    assert driver.observe(object_name="choice_status").text == "Three"


def test_choose_option_does_not_mutate_viewport_clipped_combo(
    surface, qapp, tmp_path: Path
) -> None:
    scroll = QScrollArea()
    content = QWidget()
    content.setFixedSize(500, 1200)
    combo = QComboBox(content)
    combo.setObjectName("clipped_choice")
    combo.addItems(["One", "Two", "Three"])
    combo.setGeometry(10, 1000, 250, 30)
    combo.currentTextChanged.connect(surface.statusBar().showMessage)
    scroll.setWidget(content)
    surface.setCentralWidget(scroll)
    surface.statusBar().showMessage("Unchanged")
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    driver.screenshot(tmp_path / "before-clipped-selection.png")
    assert not driver.observe(object_name="clipped_choice").visible

    driver.choose_option(object_name="clipped_choice", value="Three")
    driver.screenshot(tmp_path / "after-clipped-selection.png")
    assert driver.status().text == "Unchanged"

    bar = scroll.verticalScrollBar()
    assert bar is not None
    bar.setValue(bar.maximum())
    qapp.processEvents()
    driver.screenshot(tmp_path / "after-revealing-choice.png")
    assert driver.observe(object_name="clipped_choice").text == "One"
    driver.choose_option(object_name="clipped_choice", value="Three")
    driver.screenshot(tmp_path / "after-visible-selection.png")
    assert driver.observe(object_name="clipped_choice").text == "Three"
    assert driver.status().text == "Three"


def test_object_name_prefers_visible_tab_without_weakening_ambiguity_or_hidden_gates(
    surface, qapp, tmp_path
) -> None:
    tabs = surface.centralWidget()
    hidden_page = QWidget()
    layout = QFormLayout(hidden_page)
    hidden_field = QLineEdit("hidden draft")
    hidden_field.setObjectName("sample_name")
    layout.addRow("Other sample:", hidden_field)
    hidden_button = QPushButton("Hidden duplicate")
    hidden_button.setObjectName("first")
    layout.addRow(hidden_button)
    tabs.addTab(hidden_page, "Other")
    qapp.processEvents()
    driver = _user_input.ComponentDriver(surface)
    assert driver.observe(object_name="sample_name").text == "original"
    driver.enter_text(object_name="sample_name", value="visible edit")
    driver.screenshot(tmp_path / "01-visible-edit.png")
    assert driver.status().text == "Committed: visible edit"
    driver.click(object_name="first")
    driver.select_tab("Other")
    driver.screenshot(tmp_path / "02-other-tab.png")
    assert driver.observe(object_name="sample_name").text == "hidden draft"
    # Two simultaneously visible controls still cannot be selected by this name.
    duplicate = QLineEdit("second visible")
    duplicate.setObjectName("sample_name")
    layout.addRow(duplicate)
    qapp.processEvents()
    with pytest.raises(LookupError, match="sample_name.*2 candidates"):
        driver.enter_text(object_name="sample_name", value="ambiguous")
    duplicate.hide()
    hidden_field.hide()
    qapp.processEvents()
    with pytest.raises(LookupError, match="sample_name.*3 candidates"):
        driver.observe(object_name="sample_name")
    assert not driver.observe(object_name="hidden_action").visible
    driver.click(object_name="hidden_action")
    driver.select_tab("Run")
    driver.screenshot(tmp_path / "03-hidden-noop.png")
    assert driver.status().text == "Committed: visible edit"
