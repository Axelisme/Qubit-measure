"""Tests for the dispersive MainWindow shell."""

from __future__ import annotations

from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.state import DispersiveState


def test_main_window_builds_with_pipeline_panel(qapp):
    from zcu_tools.gui.app.dispersive.ui.main_window import MainWindow
    from zcu_tools.gui.app.dispersive.ui.pipeline_panel import PipelinePanelWidget

    state = DispersiveState()
    window = MainWindow(Controller(state))

    assert isinstance(window.centralWidget(), PipelinePanelWidget)


def test_predictor_and_native_load_are_driven_through_visible_controls(
    qapp, analysis_target, tmp_path
):
    from dataclasses import replace

    from qtpy.QtCore import QTimer
    from zcu_tools.gui.app.dispersive.ui.main_window import MainWindow
    from zcu_tools.resources.predictor_file import (
        PredictorFlux,
        load_predictor_file,
        save_predictor_file,
    )

    from tests.gui._user_input import ComponentDriver
    from tests.gui.app.analysis_project._native import native_spectrum_file

    predictor_path = analysis_target.resource.predictors_path / "fit.json"
    save_predictor_file(
        predictor_path,
        replace(
            load_predictor_file(predictor_path),
            flux=PredictorFlux(1.25, {"bias": -0.002}, {"bias": "A"}),
        ),
    )
    native_path = native_spectrum_file(tmp_path / "native.h5", {})
    window = MainWindow(Controller(DispersiveState(analysis_target)))
    window.show()
    qapp.processEvents()
    driver = ComponentDriver(window)
    failures = []

    def select_predictor():
        try:
            driver.screenshot(tmp_path / "predictor-choice.png")
            assert driver.observe("fit.json").visible
            driver.click("OK")
        except (AssertionError, LookupError, ValueError) as error:
            failures.append(error)
            driver.click("Cancel")

    try:
        driver.screenshot(tmp_path / "initial.png")
        QTimer.singleShot(0, select_predictor)
        driver.click("Load predictor…")
        if failures:
            raise failures[0]
        driver.screenshot(tmp_path / "predictor-loaded.png")
        assert "offset=1.25 Φ0" in driver.observe(object_name="predictorInfo").text
        driver.click("Load one-tone…")
        driver.screenshot(tmp_path / "source-dialog.png")
        driver.choose_file(
            "Browse…", native_path, evidence=tmp_path / "source-chooser.png"
        )
        driver.screenshot(tmp_path / "source-selected.png")
        assert driver.observe(object_name="native_device").text == "bias"
        driver.click("OK")
        driver.screenshot(tmp_path / "source-loaded.png")
        assert driver.observe("Loaded: native.h5").visible
        assert driver.observe("Run preprocessing").enabled
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
