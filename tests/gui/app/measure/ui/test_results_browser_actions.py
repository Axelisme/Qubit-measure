"""Run-only tree commands share Gallery pinning and the browser Load boundary."""

from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.registry import ExperimentResolution
from zcu_tools.gui.app.measure.services.load import LoadTabResultOutcome
from zcu_tools.gui.app.measure.services.results_index import scan_results
from zcu_tools.gui.app.measure.ui.results_browser import ResultsBrowser

from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure.ui._results_browser_support import navigation_path


def test_tree_run_menu_pins_clicked_run_deduplicates_and_excludes_groups(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    ui.click(object_name="results.figure.0")
    group = ("alpha", "p2", "mux/readout")
    ui.expand_tree(navigation_path(ui, group), object_name="results.navigation")
    ui.click("All results")
    ui.click(object_name="results.figure.0")
    run = navigation_path(ui, (*group, "bbbbbb  ·  01:00:00"))
    ui.invoke_context_action(
        action="Add to comparison",
        object_name="results.navigation",
        tree_path=run,
        evidence=tmp_path / "tree-run-menu.png",
    )
    ui.select_tab("Comparison (1)")
    assert "bbbbbb" in ui.observe(object_name="results.compare.header.0").text
    ui.invoke_context_action(
        action="Add to comparison",
        object_name="results.navigation",
        tree_path=run,
    )
    assert ui.tabs("Gallery").captions == ("Gallery", "Comparison (1)")
    ui.select_tab("Gallery")
    ui.invoke_context_action(action="Add to comparison", object_name="results.figure.0")
    assert ui.tabs("Gallery").captions == ("Gallery", "Comparison (2)")
    ui.screenshot(tmp_path / "count-two.png")
    ui.click(object_name="results.compare.remove.1")
    assert ui.tabs("Gallery").captions == ("Gallery", "Comparison (1)")
    ui.click("Clear comparison")
    assert ui.tabs("Gallery").captions == ("Gallery", "Comparison")
    for path in (("alpha",), ("alpha", "p2"), group):
        for action in ("Add to comparison", "Load into analysis tab"):
            with pytest.raises(LookupError, match="No visible popup menu"):
                ui.invoke_context_action(
                    action=action,
                    object_name="results.navigation",
                    tree_path=navigation_path(ui, path),
                )
    ui.screenshot(tmp_path / "count-cleared.png")


def test_tree_load_targets_clicked_run_and_uses_the_button_load_boundary(
    populated_browser: ComponentDriver,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    resolutions: list[str | None] = []

    def resolve(experiment: str | None) -> ExperimentResolution:
        resolutions.append(experiment)
        return ExperimentResolution("synthetic/readout", None)

    def open_run(adapter: str, path: str) -> LoadTabResultOutcome:
        calls.append((adapter, path))
        return LoadTabResultOutcome("loaded", path, "SavedReadout", None, False)

    browser = ResultsBrowser(
        scan_results(tmp_path), open_run, resolve_experiment=resolve
    )
    browser.resize(1400, 850)
    browser.show()
    qapp.processEvents()
    ui = ComponentDriver(browser)
    try:
        ui.click(object_name="results.figure.0")
        group = ("alpha", "p2", "mux/readout")
        ui.expand_tree(navigation_path(ui, group), object_name="results.navigation")
        ui.click("All results")
        ui.click(object_name="results.figure.0")
        ui.invoke_context_action(
            action="Load into analysis tab",
            object_name="results.navigation",
            tree_path=navigation_path(ui, (*group, "bbbbbb  ·  01:00:00")),
            evidence=tmp_path / "tree-load-menu.png",
        )
        second = tmp_path / "Database/alpha/runs/20261022T010000Z-bbbbbb/data.h5"
        assert calls == [("synthetic/readout", str(second))]
        assert resolutions == ["mux/readout"]
        assert (
            ui.observe(object_name="results.feedback").text
            == f"Loaded history: {second}"
        )
        ui.click("All results")
        ui.click(object_name="results.figure.0")
        ui.click("Load into analysis tab")
        first = tmp_path / "Database/alpha/runs/20261021T010000Z-aaaaaa/data.h5"
        assert calls == [
            ("synthetic/readout", str(second)),
            ("synthetic/readout", str(first)),
        ]
        assert resolutions == ["mux/readout", "mux/readout"]
        ui.screenshot(tmp_path / "tree-load-complete.png")
        browser.set_writable(writable=False)
        ui.invoke_context_action(
            action="Load into analysis tab",
            object_name="results.navigation",
            tree_path=navigation_path(ui, (*group, "bbbbbb  ·  01:00:00")),
        )
        assert len(calls) == 2
    finally:
        browser.close()
        browser.deleteLater()
        qapp.processEvents()
