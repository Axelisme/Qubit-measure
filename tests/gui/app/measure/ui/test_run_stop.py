"""Run → Stop → core return/failure through visible measure controls."""

from tests.program._profiles import load_soccfg
from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter
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


from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg, SampleResult


@pytest.fixture(params=["return", "error", "interrupt"])
def stop_window(
    request: pytest.FixtureRequest, qapp: QApplication, tmp_path: Path
) -> Iterator[tuple[MainWindow, bool]]:
    ending = str(request.param)
    fails = ending != "return"

    def acquire(
        cfg: SampleCfg, *, context: RunContext
    ) -> RunRecord[SampleCfg, SampleResult]:
        context.scope.stop.event.wait()
        if ending == "interrupt":
            raise KeyboardInterrupt("Synthetic core failed after Stop")
        if fails:
            raise RuntimeError("Synthetic core failed after Stop")
        return SampleAdapter.run(cfg, context=context)

    registry = Registry()
    registry.register(
        "synthetic-stop",
        Adapter(
            spec=SampleAdapter.spec,
            make_form=SampleAdapter.make_form,
            run=acquire,
            guide=None,
            analysis=SampleAdapter.analysis,
        ),
    )
    soccfg = load_soccfg()
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=soccfg, soccfg=soccfg)),
        registry,
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
    )
    ctrl.session_services.soc_connection.install_soc(soccfg, soccfg)
    window = MainWindow(ctrl, templates=lambda: {})
    ctrl.add_view(window)
    ctrl.setup_control.new_entry(EntryRequest("stop-scenario"))
    ctrl.setup_control.new_point(PointCreateRequest("point", "", CommandOrigin("gui")))
    window.show()
    qapp.processEvents()
    try:
        yield window, fails
    finally:
        # close drains; it does not cancel. Release the core's stop wait even
        # when a driver action or screenshot failed before the Stop click.
        ctrl.run_analyze_control.cancel_run()
        ctrl.close()
        window.deleteLater()
        qapp.processEvents()


def test_run_stop_displays_returned_result_or_original_core_error(
    stop_window: tuple[MainWindow, bool], tmp_path: Path
) -> None:
    window, fails = stop_window
    driver = ComponentDriver(window)
    driver.select_menu("New Tab ▾", ("synthetic-stop",))
    driver.screenshot(tmp_path / "01-ready.png")
    driver.click("Run")
    qt_until(lambda: driver.observe("Stop").enabled, label="Run admitted")
    driver.screenshot(tmp_path / "02-running.png")
    driver.click("Stop")
    driver.screenshot(tmp_path / "03-stop-requested.png")
    qt_until(
        lambda: any(w.text == "Run" and w.enabled for w in driver.visible().widgets),
        label="core terminal delivery",
    )
    assert driver.observe("Run").enabled
    driver.select_tab("Analysis")
    if fails:
        qt_until(lambda: bool(driver.dialogs()), label="core failure visible")
        assert any(
            dialog.title == "Run failed"
            and "Synthetic core failed after Stop" in "\n".join(dialog.texts)
            for dialog in driver.dialogs()
        )
        assert not driver.observe("Analyze").enabled
    else:
        qt_until(
            lambda: driver.observe("Analyze").enabled, label="returned result available"
        )
        assert not driver.dialogs()
    driver.screenshot(tmp_path / "04-terminal.png")
