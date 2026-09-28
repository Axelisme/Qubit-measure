"""The fixed tab tools read the same live GUI session without changing focus."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._helpers import Fixture, call, mcp_client, open_client

pytestmark = pytest.mark.uses_wall_clock


@pytest.fixture
def live_gui(qapp):
    fx = Fixture()
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
    live_gui.view.get_view_snapshot.side_effect = lambda: {
        **live_gui.view.get_view_snapshot.return_value,
        "active_tab_id": live_gui.state.active_tab_id,
    }
    sock = open_client(live_gui.service.port)
    bridge, invoke = mcp_client(live_gui.service.port, tmp_path)
    try:
        invoke("connect", {"port": live_gui.service.port})
        tab = invoke("tab_open", {"experiment": "fake"})["tab"]
        # The fixture's render-view snapshot is a static stub; State owns focus.
        assert live_gui.state.active_tab_id == tab
        focused = call(sock, "tab.new", {"adapter_name": "fake"})["result"]["tab_id"]
        assert call(sock, "tab.set_active", {"tab_id": focused})["ok"]
        before = call(sock, "tab.list_all")["result"]
        assert before["active_tab_id"] == focused
        assert live_gui.state.active_tab_id == focused

        with pytest.raises(GuiRpcError):
            invoke("tab_get", {"tab": "missing", "include": ["summary"]})
        with pytest.raises(GuiRpcError):
            invoke("tab_live", {"tab": "missing"})
        assert live_gui.state.active_tab_id == focused

        overview = invoke(
            "tab_get", {"tab": tab, "include": ["summary", "analyze_params"]}
        )
        assert overview["summary"]["experiment"] == "fake"
        assert overview["summary"]["state"]["has_result"] is False
        assert any(
            entry["name"] == "threshold"
            for entry in overview["analyze_params"]["primary"]["definitions"]
        )
        live = invoke("tab_live", {"tab": tab})
        assert live["running"] is False
        assert live["reason"] == "no_run"
        assert live["operation_state"]["tab_id"] == tab
        assert live["operation_state"]["result_state"]["available"] is False
        assert live_gui.state.active_tab_id == focused

        with pytest.raises(GuiRpcError):
            invoke(
                "tab_open",
                {"experiment": "fake", "from_file": str(tmp_path / "missing.h5")},
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
