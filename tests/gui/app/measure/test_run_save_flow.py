"""User-component Run and Save through the real measure-gui composition root."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import RunRequest
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.persistence_types import (
    AppPersistedState,
    PersistedSession,
    PersistedTab,
)
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui import _user_input
from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._app_launch import launched_measure_app
from tests.gui.app.measure._native_save_support import SavingAdapter


class PlottedSavingAdapter(SavingAdapter):
    """Persistable deterministic signal, with its curve drawn on the real host."""

    def run(
        self, req: RunRequest, cfg: SavingCfg, *, context: RunContext
    ) -> RunRecord[SavingCfg, SavingResult]:
        record = super().run(req, cfg, context=context)
        context.plots.subplots("signal")

        def draw(figure: Figure) -> None:
            axes = figure.axes[0]
            axes.plot(
                record.result.frequency, record.result.signal.real, label="Signal"
            )
            axes.legend()

        context.plots.draw("signal", draw)
        return record


@pytest.fixture
def measure_window(qapp, tmp_path: Path) -> Generator[QWidget]:
    registry = Registry()
    registry.register("synthetic", PlottedSavingAdapter)
    with launched_measure_app(qapp, tmp_path, registry=registry) as app:
        # Fixture bootstrap only: create an isolated entry/point before restoring tabs.
        app.controller.setup_control.new_entry(EntryRequest("component-flow"))
        app.controller.setup_control.new_point(
            PointCreateRequest("p", "Synthetic component flow", CommandOrigin("gui"))
        )
        app.controller.restore_persisted_state(
            AppPersistedState(
                session=PersistedSession(
                    tabs=(PersistedTab(adapter_name="synthetic", form_input={}),),
                    active_tab_index=0,
                )
            )
        )
        yield app.window


def test_run_curve_then_save_all_reports_native_save_in_bottom_status(
    measure_window: QWidget, tmp_path: Path
) -> None:
    flow = _user_input.ComponentDriver(measure_window)
    shots = tmp_path / "run-save"
    flow.screenshot(shots / "01-setup.png")
    assert flow.dialogs()[0].title == "Setup"
    flow.click("Close")
    flow.screenshot(shots / "02-ready.png")
    assert flow.observe("Run").enabled
    assert flow.legends().texts == ()
    flow.click("Run")
    flow.screenshot(shots / "03-run-requested.png")
    # The offline adapter returns after its host draw; terminal owner delivery restores Run.
    # qt_until's 30-second watchdog only guards a stuck worker/Qt delivery.
    qt_until(
        lambda: (
            flow.legends().texts == ("Signal",)
            and any(
                w.kind == "QPushButton" and w.text == "Run" and w.enabled and w.visible
                for w in flow.visible().widgets
            )
        ),
        label="synthetic Run result and idle button",
    )
    flow.screenshot(shots / "04-run-complete.png")
    assert flow.observe("Run").text == "Run"
    assert flow.legends().texts == ("Signal",)
    flow.select_tab("Data")
    flow.screenshot(shots / "05-data.png")
    assert flow.tabs("Data").current == "Data"
    assert flow.observe("Save All").enabled
    flow.click("Save All")
    flow.screenshot(shots / "06-saved.png")
    assert flow.status().text.startswith("Saved ")
    assert " for run " in flow.status().text
    assert "failed" not in flow.status().text
    assert flow.observe("✓ SAVED").visible
    assert flow.dialogs() == ()


def test_disabled_save_all_click_keeps_no_result_and_no_save_message(
    measure_window: QWidget, tmp_path: Path
) -> None:
    flow = _user_input.ComponentDriver(measure_window)
    shots = tmp_path / "disabled-save"
    flow.screenshot(shots / "01-setup.png")
    flow.click("Close")
    flow.screenshot(shots / "02-ready.png")
    flow.select_tab("Data")
    flow.screenshot(shots / "03-disabled-save.png")
    before = flow.status()
    button = flow.observe("Save All")
    assert button.visible and not button.enabled
    flow.click("Save All")
    flow.screenshot(shots / "04-after-disabled-click.png")
    assert flow.status() == before
    assert not flow.observe("Save All").enabled
    assert flow.dialogs() == ()
    flow.select_tab("Run")
    flow.screenshot(shots / "05-still-no-result.png")
    assert flow.legends().texts == ()
    assert flow.observe("Run").enabled
