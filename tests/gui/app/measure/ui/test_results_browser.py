"""Results browsing through visible controls over synthetic Database content."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import Container, ContainerAttrs, save
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot, SoftwareProvenance
from zcu_tools.gui.app.measure.services.results_index import scan_results
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.session.context_types import CommandOrigin, EntryRequest, PointCreateRequest, PointUseRequest
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.resources.entry import AcquiredPayload, LedgerEvent, ResultEntry

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure.ui._main_window_fakes import apply_window_defaults, mock_ctrl
from tests.gui.app.measure._native_save_support import NativeSaveSession
from tests.gui.app.measure._record_support import require_record


@pytest.fixture
def browser_host(
    qapp: QApplication,
    tmp_path: Path,
    main_window_control_factory: Callable[[], ControlIdentity],
) -> Iterator[MainWindow]:
    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.query_results.side_effect = lambda: scan_results(tmp_path)
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    window.show()
    qapp.processEvents()
    yield window
    window.deleteLater()
    qapp.processEvents()


def test_menu_opens_independent_results_window(
    browser_host: MainWindow,
    tmp_path: Path,
) -> None:
    ui = ComponentDriver(browser_host)
    assert "Results ▾" in tuple(w.text for w in ui.visible().widgets)
    ui.select_menu("Results ▾", ("Results browser…",))
    assert ui.observe(object_name="results.browser").visible
    assert ui.observe(object_name="results.focus").text == "All results · 0 runs"
    ui.screenshot(tmp_path / "empty-browser.png")


@pytest.fixture
def populated_browser(browser_host: MainWindow, tmp_path: Path) -> ComponentDriver:
    for entry_name, point, run_id, experiment, day in (
        ("alpha", "p1", "20261021T010000Z-aaaaaa", "mux/readout", "21"),
        ("alpha", "p2", "20261022T010000Z-bbbbbb", "mux/readout", "22"),
        ("beta", "p3", "20261023T010000Z-cccccc", "sweep", "23"),
    ):
        entry_path = tmp_path / "result" / entry_name
        entry = (
            ResultEntry.open(
                entry_name,
                result_root=tmp_path / "result",
                database_root=tmp_path / "Database",
            )
            if entry_path.exists()
            else ResultEntry.create(
                entry_name,
                result_root=tmp_path / "result",
                database_root=tmp_path / "Database",
            )
        )
        run = tmp_path / "Database" / entry_name / "runs" / run_id
        (run / "figures").mkdir(parents=True)
        metadata = RunMetadata(
            run_id=run_id,
            experiment=experiment,
            started_at=f"2026-10-{day}T01:00:00Z",
            finished_at=None,
            completion="unknown",
            snapshot=RunSnapshot(
                entry_id=UUID(entry.entry_id),
                entry_name=entry_name,
                point=point,
                description=None,
                roles={"qubit": "Q1"},
            ),
            provenance=SoftwareProvenance(
                software_versions={},
                git_commit=None,
                git_dirty=None,
                qick_version=None,
                soc_fingerprint=None,
                hostname=None,
            ),
        )
        save(
            run / "data.h5",
            Container(
                vars={},
                docs={
                    "metadata": metadata.model_dump(mode="json"),
                    "cfg": {
                        "values": {"readout": {"frequency": 5100.0, "tone": "Q2"}},
                        "cfg_type": "Synthetic",
                        "schema_version": "1.0",
                    },
                },
                groups={},
                attrs=ContainerAttrs("3.0", "synthetic-browser", None, {}),
            ),
        )
        (run / "analysis").mkdir()
        (run / "analysis" / "analysis.json").write_text(
            '{"format":"zcu.analysis-result","format_version":"1.0",'
            '"pane":"analysis","parameters":{"n_iter":12},'
            '"summary":{"tones":[{"name":"Q2","freq":5100.0}]}}',
            encoding="utf-8",
        )
        for i in range(8 if point == "p1" else 1):
            image = QImage(400, 220, QImage.Format.Format_RGB32)
            image.fill(Qt.GlobalColor.cyan if i % 2 else Qt.GlobalColor.yellow)
            assert image.save(str(run / "figures" / f"plot-{i}.png"))
        if point == "p1":
            entry.ledger.append(
                LedgerEvent(
                    id=str(uuid4()),
                    entry_id=entry.entry_id,
                    kind="acquired",
                    at="2026-10-21T01:05:00Z",
                    origin="notebook",
                    payload=AcquiredPayload(
                        run_id=run_id,
                        tab="Synthetic",
                        experiment=experiment,
                        point=point,
                        roles={"qubit": "Q1"},
                    ),
                )
            )
    ui = ComponentDriver(browser_host)
    ui.select_menu("Results ▾", ("Results browser…",))
    return ui


def test_filters_grouping_and_focus_history(populated_browser: ComponentDriver, tmp_path: Path) -> None:
    ui = populated_browser
    assert ui.observe(object_name="results.focus").text == "All results · 3 runs"
    assert {r.text for r in ui.list_rows()} >= {"mux/readout", "sweep"}
    ui.select_combo("Point → experiment", object_name="results.group")
    assert {r.text for r in ui.list_rows()} >= {"alpha / p1", "alpha / p2", "beta / p3"}
    ui.enter_text(value="Q2", object_name="results.search")
    ui.select_combo("mux/readout", object_name="results.experiment")
    ui.enter_text(value="2026-10-22", object_name="results.date_from")
    ui.enter_text(value="2026-10-22", object_name="results.date_to")
    assert ui.observe(object_name="results.focus").text == "All results · 1 runs"
    ui.select_list_row("alpha / p2")
    assert ui.observe(object_name="results.focus").text == "alpha / p2 · 1 runs"
    ui.click("Back")
    assert ui.observe(object_name="results.focus").text == "All results · 1 runs"
    ui.click("Forward")
    assert ui.observe(object_name="results.focus").text == "alpha / p2 · 1 runs"
    ui.enter_text(value="no-such-tone", object_name="results.search")
    assert ui.observe(object_name="results.focus").text == "alpha / p2 · 0 runs"
    ui.screenshot(tmp_path / "filtered-focus.png")


def test_figure_drilldown_details_scroll_and_pins(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    assert "results.gallery" in {w.object_name for w in ui.visible().widgets}
    assert ui.scroll_area(object_name="results.gallery").vertical_scrollable
    ui.click(object_name="results.figure.0")
    assert ui.observe(object_name="results.focus").text == "mux/readout · 2 runs"
    ui.click(object_name="results.figure.0")
    assert ui.observe(object_name="results.focus").text == "alpha / p1 · 1 runs"
    ui.click(object_name="results.figure.0")
    assert "20261021T010000Z-aaaaaa" in ui.observe(object_name="results.focus").text
    ui.click(object_name="results.figure.1")
    assert (
        ui.observe(object_name="results.current_figure").text
        == "Current figure: plot-1.png"
    )
    details = ui.observe(object_name="results.details").text
    assert "readout.frequency = 5100.0" in details
    assert "tones[0].freq = 5100.0" in details
    assert "Parameters · analysis.json" in details and "n_iter = 12" in details
    assert "alpha / p1" in ui.observe(object_name="results.figure.0").tooltip
    ui.invoke_context_action(action="Add to comparison", object_name="results.figure.1")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 1"
    ui.click("Add to comparison")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 1"
    assert any("20261021T010000Z-aaaaaa" in row.text for row in ui.list_rows())
    ui.invoke_context_action(action="Copy path", object_name="results.figure.1")
    assert "Copied path:" in ui.observe(object_name="results.feedback").text
    ui.invoke_context_action(action="Copy image", object_name="results.figure.1")
    assert ui.observe(object_name="results.feedback").text == "Copied image: plot-1.png"
    ui.screenshot(tmp_path / "run-details-pinned.png")
    ui.press_key(Qt.Key.Key_End, object_name="results.gallery")
    assert ui.observe(object_name="results.figure.7").visible
    ui.screenshot(tmp_path / "gallery-bottom.png")
    ui.press_key(Qt.Key.Key_Home, object_name="results.pins")
    ui.click("Unpin selected")
    assert ui.observe(object_name="results.pin_count").text == "Pinned runs: 0"


def test_figure_only_run_explains_open_refusal(
    browser_host: MainWindow, tmp_path: Path
) -> None:
    run = tmp_path / "Database" / "orphan" / "runs" / "unknown"
    (run / "figures").mkdir(parents=True)
    image = QImage(100, 100, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.red)
    assert image.save(str(run / "figures" / "only.png"))
    ui = ComponentDriver(browser_host)
    ui.select_menu("Results ▾", ("Results browser…",))
    assert ui.observe(object_name="results.focus").text == "All results · 1 runs"
    assert "results.figure.0" in {w.object_name for w in ui.visible().widgets}
    for _ in range(3):
        ui.click(object_name="results.figure.0")
    button = ui.observe("Open in new analysis tab")
    assert not button.enabled
    assert "No data.h5" in button.tooltip
    assert "Unregistered" in ui.observe(object_name="results.details").text
    ui.screenshot(tmp_path / "figure-only.png")


@pytest.fixture
def load_browser(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    request: pytest.FixtureRequest,
    qapp: QApplication,
) -> Iterator[tuple[ComponentDriver, str]]:
    session, ctrl, window, _tab_id = live_ui
    variant = request.param
    if variant in {"missing", "pointless"}:
        tab = session.state.get_tab("tab")
        history = replace(session.metadata.snapshot, point="missing" if variant == "missing" else None)
        session.state.update_tab_result("tab", replace(require_record(tab.run),
            metadata=session.metadata.model_copy(update={"snapshot": history})), parameters=tab.run.parameters)
    session.save()
    image = QImage(400, 220, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.yellow)
    figures = session.native_data_path.parent / "figures"
    figures.mkdir()
    assert image.save(str(figures / "saved.png"))
    ctrl.setup_control.open_entry(EntryRequest("original"))
    if variant != "empty":
        ctrl.context_control.use_point(PointUseRequest("p"))
    if variant == "foreign":
        ctrl.setup_control.new_entry(EntryRequest("foreign"))
        ctrl.setup_control.new_point(PointCreateRequest("other", "", CommandOrigin("gui")))
    ctrl.add_view(window)
    window.show()
    qapp.processEvents()
    before = set(ctrl.list_tab_ids())
    yield ComponentDriver(window), variant
    for tab_id in set(ctrl.list_tab_ids()) - before:
        ctrl.close_tab(tab_id)


@pytest.mark.parametrize("load_browser", ["available", "empty", "foreign", "pointless", "missing"], indirect=True)
def test_new_analysis_tab_projects_the_same_load_source_reason(
    load_browser: tuple[ComponentDriver, str], tmp_path: Path,
) -> None:
    ui, variant = load_browser
    ui.select_menu("Results ▾", ("Results browser…",))
    for _ in range(3):
        ui.click(object_name="results.figure.0")
    ui.click("Open in new analysis tab")
    feedback = ui.observe(object_name="results.feedback").text
    if variant == "available":
        assert feedback.startswith("Loaded history:")
        assert not ui.observe(object_name="history.analysis_source").visible
    else:
        expected = {
            "empty": "No active context is available",
            "foreign": "Historical entry differs from the opened entry",
            "pointless": "Historical metadata has no point",
            "missing": "Historical point 'missing' is unavailable",
        }[variant]
        assert "History loaded, no analysis source:" in feedback and expected in feedback
        tab_message = ui.observe(object_name="history.analysis_source")
        assert tab_message.visible and expected in tab_message.text
        assert "History loaded, no analysis source:" in tab_message.text
    ui.screenshot(tmp_path / f"load-{variant}.png")
