"""The fixed tab tools read the same live GUI session without changing focus."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui._qt_waits import qt_until

from ._helpers import Fixture, call, mcp_client, open_client

pytestmark = pytest.mark.uses_wall_clock


@pytest.fixture
def live_gui(tmp_path, qapp):
    fx = Fixture(active_label="ctx001", project_root=str(tmp_path / "project"))
    fx.start()
    try:
        yield fx
    finally:
        fx.stop()


def test_existing_tab_reads_and_failed_open_share_gui_state(
    live_gui: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    # The real render view follows State focus; this fixture's view is a stub.
    view = live_gui.view
    assert view is not None
    view.get_view_snapshot.side_effect = lambda: {
        **view.get_view_snapshot.return_value,
        "active_tab_id": live_gui.state.active_tab_id,
    }
    sock = open_client(live_gui.service.port)
    bridge, invoke = mcp_client(live_gui.service.port, tmp_path)
    try:
        invoke("connect", {"port": live_gui.service.port})
        tab = invoke(
            "rpc_call", {"method": "tab.new", "params": {"adapter_name": "fake"}}
        )["tab_id"]
        live_gui.ctrl.run_analyze_control.start_run(tab)
        qt_until(
            lambda: live_gui.ctrl.active_operation_count() == 0,
            label="sample Run setup terminal",
        )
        assert live_gui.state.get_tab(tab).run.result is not None
        # Run setup ends before the read-only observation and focus baseline.
        # The fixture's render-view snapshot is a static stub; State owns focus.
        assert live_gui.state.active_tab_id == tab
        focused = call(sock, "tab.new", {"adapter_name": "fake"})["result"]["tab_id"]
        assert call(sock, "tab.set_active", {"tab_id": focused})["ok"]
        before = call(sock, "tab.list_all")["result"]
        assert before["active_tab_id"] == focused
        assert live_gui.state.active_tab_id == focused

        with pytest.raises(GuiRpcError):
            invoke(
                "rpc_call", {"method": "tab.snapshot", "params": {"tab_id": "missing"}}
            )["tabs"][0]
        assert live_gui.state.active_tab_id == focused

        overview = invoke(
            "rpc_call", {"method": "tab.snapshot", "params": {"tab_id": tab}}
        )["tabs"][0]
        assert overview["adapter_name"] == "fake"
        assert overview["interaction"]["has_run_result"] is True
        assert overview["interaction"]["is_running"] is False
        assert overview["tab_id"] == tab
        assert overview["result_state"]["available"] is True
        assert live_gui.state.active_tab_id == focused

        invoke("rpc_call", {"method": "context.snapshot"})
        with pytest.raises(GuiRpcError, match="Could not read the data file"):
            invoke(
                "rpc_call",
                {
                    "method": "tab.open_file",
                    "params": {
                        "adapter_name": "fake",
                        "data_path": str(tmp_path / "missing.h5"),
                    },
                },
            )
        after = call(sock, "tab.list_all")["result"]
        assert {entry["tab_id"] for entry in after["tabs"]} == {
            entry["tab_id"] for entry in before["tabs"]
        }
        assert after["active_tab_id"] == focused
        assert live_gui.state.active_tab_id == focused
    finally:
        bridge.disconnect()
        sock.close()
