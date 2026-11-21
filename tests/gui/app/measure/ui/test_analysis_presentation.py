"""Analyze publishes the same operator summary in the window and MCP."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.native_persistence import save_run
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    AnalysisSavingAdapter_analyze,
    NativeSaveSession,
    PaneParams,
)
from tests.gui.app.measure.remote._form_rpc import FormRpc
from tests.mcp.measure._support import make_client


def _summary_lines(source: RunRecord[SavingCfg, SavingResult]) -> tuple[str, ...]:
    if source.cfg is None or source.cfg.gain == 0.0:
        return ()
    return (f"Source gain: {source.cfg.gain:g}", "Applied setting: 0 MHz")


@pytest.fixture
def live_registry() -> Registry:
    """Reuse the native UI fixture with an authored, source-bound summary."""
    registry = Registry()
    registry.register(
        "saving",
        Adapter(
            spec=AnalysisSavingAdapter.spec,
            make_form=AnalysisSavingAdapter.make_form,
            run=AnalysisSavingAdapter.run,
            guide=None,
            analysis=Analysis(
                fn=AnalysisSavingAdapter_analyze,
                options=lambda source, env: PaneParams(),
                summary_lines=_summary_lines,
            ),
        ),
    )
    return registry


def _native_sources(tmp_path: Path, session: NativeSaveSession) -> tuple[Path, Path]:
    source_path = tmp_path / "source.h5"
    quiet_path = tmp_path / "summary-free.h5"
    for path, cfg in ((source_path, SavingCfg()), (quiet_path, SavingCfg(gain=0.0))):
        source = RunRecord(
            cfg,
            SavingCore().run(cfg, context=None),
            metadata=session.metadata.model_dump(mode="json"),
        )
        save_run(source, path, spec=SavingCore.AXES_SPEC)
    return source_path, quiet_path


def _rpc_result(
    rpc: FormRpc, method: str, params: dict[str, object]
) -> dict[str, object]:
    reply = rpc.call(method, params)
    assert reply["ok"], reply
    return reply["result"]


def test_analysis_shows_mcp_summary_lines_for_the_same_run(
    live_ui: tuple[NativeSaveSession, Controller, MainWindow, str],
    tmp_path: Path,
    qapp: QApplication,
) -> None:
    session, ctrl, window, tab_id = live_ui
    source_path, quiet_path = _native_sources(tmp_path, session)
    ctrl.run_analyze_control.load_tab_result(tab_id, str(source_path))
    window.show()
    qapp.processEvents()
    ui = ComponentDriver(window)
    ui.screenshot(tmp_path / "01-run.png")
    ui.select_tab("Analysis")
    ui.screenshot(tmp_path / "02-analysis-ready.png")
    ui.press_key(Qt.Key.Key_Space, "Analyze")
    qt_until(
        lambda: ui.observe(object_name="analysis.summary_lines").visible,
        label="local analysis summary visible",
    )
    ui.screenshot(tmp_path / "03-local-analysis.png")

    remote = RemoteControlAdapter(ctrl, ControlOptions(port=0), QtOwnerScheduler())
    with ThreadPoolExecutor() as worker:
        connected = worker.submit(FormRpc, remote)
        qt_until(connected.done)
        rpc = connected.result()

        client = make_client(
            tmp_path, lambda method, params: _rpc_result(rpc, method, params)
        )
        try:
            completed = worker.submit(client.call, "tab_analyze", {"tab": tab_id})
            qt_until(completed.done)
            reply = completed.result()
            ui.screenshot(tmp_path / "04-mcp-analysis.png")
            assert reply.data["status"] == "finished", reply.data
            primary = reply.data["analysis"]["primary"]
            assert primary["summary"] == {"mean": 3.0}
            presentation = primary["presentation"]
            assert presentation["run_id"] == session.metadata.run_id
            assert presentation.get("summary_lines") == [
                "Source gain: 1",
                "Applied setting: 0 MHz",
            ]
            assert (
                ui.observe(object_name="analysis.summary_lines").text.splitlines()
                == (presentation["summary_lines"])
            )

            loaded = worker.submit(
                _rpc_result,
                rpc,
                "tab.load_data",
                {"tab_id": tab_id, "data_path": str(quiet_path)},
            )
            qt_until(loaded.done)
            assert loaded.result()["data_path"] == str(quiet_path)
            ui.screenshot(tmp_path / "05-loaded-summary-free.png")
            assert not ui.observe(object_name="analysis.summary_lines").visible

            analyzed = worker.submit(client.call, "tab_analyze", {"tab": tab_id})
            qt_until(analyzed.done)
            quiet_reply = analyzed.result()
            ui.screenshot(tmp_path / "06-analyzed-summary-free.png")
            assert quiet_reply.data["status"] == "finished", quiet_reply.data
            assert (
                "summary_lines"
                not in quiet_reply.data["analysis"]["primary"]["presentation"]
            )
            assert not ui.observe(object_name="analysis.summary_lines").visible
        finally:
            client.context.session.close()
            remote.on_client_close(rpc.link, on_owner_thread=True)
