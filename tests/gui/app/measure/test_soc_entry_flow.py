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
from zcu_tools.gui.app.measure.adapter import Adapter
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.ui.arb_waveform_dialog import ArbWaveformDialog

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

    def HeldAdapter_run(
        cfg: SavingCfg, *, context: RunContext
    ) -> RunRecord[SavingCfg, SavingResult]:
        entered.set()
        if not release.wait(30):
            raise RuntimeError("scenario did not release synthetic run")
        return SavingAdapter.run(cfg, context=context)

    HeldAdapter = Adapter(
        spec=SavingAdapter.spec,
        make_form=SavingAdapter.make_form,
        run=HeldAdapter_run,
        guide=None,
        analysis=SavingAdapter.analysis,
        post_analysis=SavingAdapter.post_analysis,
    )

    registry = Registry()
    registry.register("held", HeldAdapter)
    with launched_measure_app(qapp, tmp_path, registry=registry) as app:
        env = white_noise()
        app.controller.session_services.soc_connection.install_soc(env.soc, env.soccfg)
        app.controller.setup_control.new_entry(EntryRequest("synthetic"))
        app.controller.setup_control.new_point(
            PointCreateRequest("p", "Synthetic source", CommandOrigin("gui"))
        )
        app.controller.new_tab("held")
        try:
            yield app, entered, release
        finally:
            release.set()


def arb_driver(app: LaunchedMeasureApp) -> ComponentDriver:
    """Scope visible controls to ARB when Inspect remains open behind it."""
    return ComponentDriver(next(iter(app.window.findChildren(ArbWaveformDialog))))


@pytest.mark.parametrize(
    ("opening", "action", "run_editable"),
    [
        ("Setup…", "New entry…", False),
        ("SoC…", "Connect", False),
        ("Devices…", "Add Device", True),
        ("Inspect…", "New", True),
        ("Arb Waveforms…", "Save", True),
    ],
)
def test_run_only_locks_setup_soc_and_preserves_remote_control_policy(
    soc_app: tuple[LaunchedMeasureApp, Event, Event],
    tmp_path: Path,
    opening: str,
    action: str,
    run_editable: bool,
) -> None:
    app, entered, release = soc_app
    ui = ComponentDriver(app.window)
    shots = Path(os.environ.get("SOC_UI_EVIDENCE", str(tmp_path))) / opening.rstrip("…")
    ui.click("Close")
    ui.click("Inspect…" if opening == "Arb Waveforms…" else opening)
    if opening == "Inspect…":
        ui.select_tab("Modules")
    if opening == "Arb Waveforms…":
        ui.click("Arb Waveforms…")
    ui.screenshot(shots / "01-editable.png")
    assert ui.observe(action).enabled
    ui.click("Run")
    qt_until(entered.is_set, label="synthetic run entered")
    ui.screenshot(shots / "02-running.png")
    assert ui.observe(action).enabled is run_editable

    # A remote owner is an external collaborator, not a shortcut for local UI commands.
    authority = app.controller.control_identity
    authority.connect("scenario", "agent", "owner")
    ui.screenshot(shots / "03-remote.png")
    assert not ui.observe(action).enabled
    authority.release("scenario")
    ui.screenshot(shots / "04-local-running.png")
    assert ui.observe(action).enabled is run_editable

    if opening == "Arb Waveforms…":
        arb_driver(app).click("Close")
    ui.click("Close")
    # Opening a Run-locked window must not replace the registry's Local authority.
    ui.click("SoC…")
    assert not ui.observe("Connect").enabled
    ui.screenshot(shots / "05-running-soc.png")
    ui.click("Close")
    ui.click("Inspect…" if opening == "Arb Waveforms…" else opening)
    if opening == "Inspect…":
        ui.select_tab("Modules")
    if opening == "Arb Waveforms…":
        ui.click("Arb Waveforms…")
    ui.screenshot(shots / "06-reopened-running.png")
    assert ui.observe(action).enabled is run_editable
    authority.acquire("scenario")
    release.set()
    qt_until(
        lambda: any(w.text == "Run" for w in ui.visible().widgets),
        label="synthetic run delivered",
    )
    ui.screenshot(shots / "07-finished-remote.png")
    assert not ui.observe(action).enabled
    # Local return after terminal delivery restores edits, never before authority returns.
    ui.click(object_name="control_indicator")
    ui.screenshot(shots / "08-finished-local.png")
    assert ui.observe(action).enabled
    if opening == "Arb Waveforms…":
        arb_driver(app).click("Close")
    ui.click("Close")


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
