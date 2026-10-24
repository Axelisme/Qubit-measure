"""Historical Load keeps form failure local while preserving the session status."""

from pathlib import Path

from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.ui.main_window import MainWindow

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._history_form_support import history_controller
from tests.gui.app.measure._load_support import write_native_file


def test_browser_load_keeps_factory_failure_in_form_and_run(
    qapp: QApplication, tmp_path: Path
) -> None:
    run = tmp_path / "Database" / "historical" / "runs" / "saved"
    run.mkdir(parents=True)
    write_native_file(run / "data.h5", tag="test/saving")
    with history_controller(tmp_path) as ctrl:
        window = MainWindow(
            ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
        )
        ctrl.add_view(window)
        window.resize(1400, 900)
        window.show()
        window.show_status_message("History browser ready")
        qapp.processEvents()
        ui = ComponentDriver(window)
        try:
            ui.select_menu("More ▾", ("Results browser…",))
            ui.select_combo("Experiment → point", object_name="results.group")
            for depth in (1, 2, 3):
                group = next(
                    row.path
                    for row in ui.visible().tree_rows
                    if row.tree_name == "results.navigation" and len(row.path) == depth
                )
                ui.expand_tree(group, object_name="results.navigation")
            path = next(
                row.path
                for row in ui.visible().tree_rows
                if row.tree_name == "results.navigation" and len(row.path) == 4
            )
            ui.select_tree_row(path, object_name="results.navigation")
            ui.click("Load into analysis tab")
            assert "Loaded history:" in ui.observe(object_name="results.feedback").text
            assert ui.tabs("history").current == "history"
            reason = "factory: history_source is missing"
            notice = ui.observe(object_name="history.form_error")
            assert notice.visible and notice.text == reason
            button = ui.observe("Run")
            assert not button.enabled and reason in button.tooltip
            assert ui.status().text == "History browser ready"
            ui.select_tab("Data")
            assert ui.observe(object_name="dataFigurePreviewGallery").visible
            assert ui.status().text == "History browser ready"
            ui.screenshot(tmp_path / "history-form-diagnostic.png")
        finally:
            ctrl.close()
            window.deleteLater()
            qapp.processEvents()
