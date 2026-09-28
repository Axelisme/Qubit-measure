"""Real MCP/socket analysis uses GUI parameters and pane-owned writeback."""

from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.fixture()
def fx(qapp):
    fixture = Fixture(active_label="ctx001")
    fixture.state.set_context(
        replace(fixture.state.session_env, md=MetaDict(), ml=ModuleLibrary())
    )
    fixture.start()
    yield fixture
    fixture.stop()


def test_mcp_analysis_returns_actual_params_and_replaces_old_draft(fx, tmp_path):
    tab = fx.ctrl.new_tab("fake")
    run = fx.ctrl.start_run(tab)
    sock = open_client(fx.service.port)
    try:
        assert (
            call(sock, "operation.await", {"operation_id": run, "timeout": 2})[
                "result"
            ]["status"]
            == "finished"
        )
    finally:
        sock.close()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke("rpc_call", {"method": "context.snapshot"})
        invoke("tab_get", {"tab": tab})
        with pytest.raises(GuiRpcError, match="threshold") as error:
            invoke("tab_analyze", {"tab": tab, "params": {"threshold": "bad"}})
        assert error.value.code == "invalid_params"
        assert "float" in str(error.value)
        first = invoke("tab_analyze", {"tab": tab, "params": {"threshold": 0.3}})
        assert first["status"] == "finished"
        assert first["params"] == {"threshold": 0.3}
        assert first["invalidated"] == []
        assert Path(first["figure"]).is_file()
        previous = fx.state.get_tab(tab).analysis.writeback_draft
        assert previous is not None and previous.is_active
        second = invoke("tab_analyze", {"tab": tab})
        assert second["status"] == "finished"
        assert second["params"] == {"threshold": 0.3}
        assert second["invalidated"] == ["analysis.writeback"]
        assert not previous.is_active
        current = fx.state.get_tab(tab).analysis
        assert current.writeback_draft is not previous
        assert second["summary"] == current.result.to_summary_dict()
    finally:
        bridge.disconnect()
