"""SoC and Setup toolbar entry points share the host's run edit lock."""

import os
from collections.abc import Iterator
from pathlib import Path
from threading import Event

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.gui.app.measure.adapter import RunRequest
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._app_launch import LaunchedMeasureApp, launched_measure_app
from tests.gui.app.measure._native_save_support import SavingAdapter


@pytest.fixture
def soc_app(
    qapp: QApplication, tmp_path: Path
) -> Iterator[tuple[LaunchedMeasureApp, Event, Event]]:
    entered, release = Event(), Event()

    class HeldAdapter(SavingAdapter):
        def run(
            self, req: RunRequest, cfg: SavingCfg, *, context: RunContext
        ) -> RunRecord[SavingCfg, SavingResult]:
            entered.set()
            # The scenario releases the synthetic worker; teardown also releases on failure.
            if not release.wait(30):
                raise RuntimeError("scenario did not release synthetic run")
            return super().run(req, cfg, context=context)

    registry = Registry()
    registry.register("held", HeldAdapter)
    with launched_measure_app(qapp, tmp_path, registry=registry) as app:
        env = white_noise()
        app.controller.install_soc(env.soc, env.soccfg)
        app.controller.setup_control.new_entry(EntryRequest("synthetic"))
        app.controller.setup_control.new_point(
            PointCreateRequest("p", "Synthetic source", CommandOrigin("gui"))
        )
        app.controller.new_tab("held")
        try:
            yield app, entered, release
        finally:
            release.set()


def test_toolbar_soc_reopens_and_both_dialogs_follow_run_lock(
    soc_app: tuple[LaunchedMeasureApp, Event, Event], tmp_path: Path
) -> None:
    app, entered, release = soc_app
    ui = ComponentDriver(app.window)
    shots = Path(os.environ.get("SOC_UI_EVIDENCE", str(tmp_path))) / "measure"
    ui.screenshot(shots / "01-launch-setup.png")
    assert "IP address:" not in ui.dialogs()[0].texts
    ui.click("Close")
    ui.screenshot(shots / "02-toolbar.png")
    assert any(w.text == "SoC…" for w in ui.visible().widgets), (
        "SoC toolbar entry missing"
    )
    ui.click("SoC…")
    assert ui.dialogs()[0].title == "SoC"
    assert ui.observe(object_name="soc_ip").visible
    assert ui.observe("Connect").enabled
    ui.screenshot(shots / "03-soc.png")
    ui.click("Close")
    assert ui.dialogs() == ()
    ui.screenshot(shots / "04-closed.png")
    ui.click("SoC…")
    assert ui.observe(object_name="soc_ip").visible
    ui.screenshot(shots / "05-reopened.png")
    ui.click("Run")
    # The accepted run enters HeldAdapter; release below guarantees its terminal delivery.
    qt_until(entered.is_set, label="synthetic run entered")
    assert not ui.observe("Connect").enabled
    assert not ui.observe(object_name="soc_ip").enabled
    assert ui.observe("Close").enabled
    assert any(
        w.kind == "QPlainTextEdit" and w.enabled and bool(w.text)
        for w in ui.visible().widgets
    )
    ui.screenshot(shots / "06-running-soc.png")
    ui.click("Close")
    ui.click("SoC…")
    assert not ui.observe("Connect").enabled
    ui.screenshot(shots / "07-running-soc-reopened.png")
    ui.click("Close")
    ui.click("Setup…")
    assert ui.dialogs()[0].title == "Setup"
    assert "IP address:" not in ui.dialogs()[0].texts
    assert all(
        not ui.observe(caption).enabled for caption in ("New entry…", "Open", "Create")
    )
    assert ui.observe(object_name="working_points").enabled
    ui.select_list_row(ui.list_rows()[0].text)
    assert ui.list_rows()[0].selected
    assert ui.observe("Close").enabled
    ui.screenshot(shots / "08-running-setup.png")
    release.set()
    qt_until(
        lambda: any(w.text == "Run" and w.enabled for w in ui.visible().widgets),
        label="synthetic run delivered",
    )
    assert ui.observe("New entry…").enabled
    ui.screenshot(shots / "09-unlocked-setup.png")
    ui.click("Close")
    ui.click("SoC…")
    assert ui.observe("Connect").enabled
    ui.screenshot(shots / "10-unlocked-soc.png")
    ui.click("Close")
