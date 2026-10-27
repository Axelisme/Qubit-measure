"""Export status shows the same component calibration result as RPC and MCP."""

from pathlib import Path

from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
from zcu_tools.resources.entry import ResultEntry

from tests.gui._user_input import ComponentDriver
from tests.gui.app.fluxdep._components import skip_calibration
from tests.gui.app.fluxdep._export import aligned_export_state


def test_export_shows_skipped_calibration_and_next_action(
    qapp, fluxonium_entry: ResultEntry, tmp_path: Path
) -> None:
    target = AnalysisTarget.open(
        fluxonium_entry.result_path.parent.parent, "synthetic", "Q1"
    )
    ctrl = Controller(aligned_export_state(target))
    ctrl.configure_export(skip_calibration)
    panel = AnalyzePanelWidget(ctrl)
    panel.resize(1200, 900)
    panel.show()
    try:
        driver = ComponentDriver(panel)
        driver.screenshot(tmp_path / "01-filter.png")
        driver.click(object_name="fluxdep.fit.stage.search")
        driver.screenshot(tmp_path / "02-search.png")
        assert driver.observe(object_name="fluxdep.fit.export_params").enabled
        driver.click(object_name="fluxdep.fit.export_params")
        driver.screenshot(tmp_path / "03-export.png")
        status = driver.observe(object_name="fluxdep.fit.status")
        assert "Exported" in status.text
        assert "Flux calibration skipped" in status.text
        assert "Use fluxdep measurement writeback" in status.text
        assert status.visible and status.caption_fits
    finally:
        panel.quiesce()
        panel.dispose()
        panel.close()
        panel.deleteLater()
        ctrl.interactive.dispose()
