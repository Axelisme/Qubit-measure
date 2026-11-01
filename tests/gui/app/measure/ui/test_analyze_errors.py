"""Analyze terminal feedback through real owners and visible component input."""

import os
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._synthetic import (
    SampleAdapter,
    SampleAnalysis,
    SampleCfg,
    SampleParams,
    SampleResult,
)
from tests.program._profiles import load_soccfg


@pytest.mark.parametrize("error_type", [AnalysisFailed, ValueError, None])
def test_analyze_shows_errors_and_publishes_returns(
    qapp: QApplication, tmp_path: Path, error_type: type[Exception] | None
) -> None:
    message = "Synthetic analysis could not produce a result"

    def analyze(
        req: AnalyzeRequest[RunRecord[SampleCfg, SampleResult], SampleParams],
        *,
        plots: Plots,
    ) -> SampleAnalysis:
        if error_type is not None:
            raise error_type(message)
        _, axes = plots.subplots("fit")
        axes.plot([0, 1], [0.1, 0.3], label="Synthetic fit")
        axes.set_xlabel("Sample")
        axes.set_ylabel("Signal")
        axes.legend()
        return SampleAnalysis(peak=0.3)

    registry = Registry()
    registry.register(
        "sample",
        Adapter(
            spec=SampleAdapter.spec,
            make_form=SampleAdapter.make_form,
            run=SampleAdapter.run,
            guide=None,
            analysis=Analysis(fn=analyze, options=lambda source, env: SampleParams()),
        ),
    )
    soccfg = load_soccfg()
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
    )
    ctrl.session_services.soc_connection.install_soc(soccfg, soccfg)
    window = MainWindow(ctrl, templates=lambda: {})
    ctrl.add_view(window)
    ctrl.setup_control.new_entry(EntryRequest("analyze-errors"))
    ctrl.setup_control.new_point(PointCreateRequest("point", "", CommandOrigin("gui")))
    ctrl.tab_control.new_tab("sample")
    window.resize(1500, 950)
    window.show()
    qapp.processEvents()
    ui = ComponentDriver(window)
    scenario = error_type.__name__ if error_type else "success"
    evidence = Path(os.environ.get("ANALYZE_ERROR_EVIDENCE", str(tmp_path))) / scenario
    try:
        ui.screenshot(evidence / "01-ready.png")
        ui.click("Run")
        ui.screenshot(evidence / "02-run-requested.png")
        ui.select_tab("Analysis")
        qt_until(
            lambda: ui.observe("Analyze").enabled, label="Run publishes analysis input"
        )
        ui.screenshot(evidence / "03-run-published.png")
        ui.click("Analyze")
        if error_type is not None:
            qt_until(lambda: bool(ui.dialogs()), label="Analyze error displayed")
            dialogs = ui.dialogs()
            assert any(
                dialog.title == "Analyze failed" and message in dialog.texts
                for dialog in dialogs
            ), dialogs
            ui.screenshot(evidence / "04-error-visible.png")
        else:
            qt_until(
                lambda: "Synthetic fit" in ui.legends().texts,
                label="Analyze return published",
            )
            ui.screenshot(evidence / "04-analysis-published.png")
    finally:
        ctrl.close()
        window.dispose()
        qapp.processEvents()
