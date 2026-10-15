"""Invalid Form resource publication reaches socket/MCP before Run admission."""

from pathlib import Path

import pytest
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.form import FormErrors
from zcu_tools.gui.form_adapter import Form

from tests.gui.app.measure._synthetic import SampleAdapter

from ._helpers import Fixture, call, mcp_client, observe_run_inputs, open_client

pytestmark = pytest.mark.uses_wall_clock


class FiniteCfg(ConfigBase):
    """Synthetic numeric input uses the existing finite-number constraint."""

    gain: float = Field(default=0.1, allow_inf_nan=False)


@pytest.fixture
def live_cfg_resource(qapp):
    fx = Fixture()
    tab_id = "tab-live"
    resource = fx.prepare_tab(tab_id, SampleAdapter(), Form(FiniteCfg, {}))
    fx.start()
    try:
        yield fx, tab_id, resource
    finally:
        fx.stop()


@pytest.mark.parametrize("text", ["nan", "inf"])
def test_invalid_resource_publishes_complete_socket_and_mcp_cfg_then_blocks_run(
    live_cfg_resource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    # The MCP connect status orientation is unrelated to cfg observation.
    monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    fx, tab_id, resource = live_cfg_resource
    resource.write({("gain",): text})
    assert isinstance(resource.result(), FormErrors)

    sock = open_client(fx.service.port)
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        tab = call(sock, "tab.get_cfg", {"tab_id": tab_id}, rid="tab")
        assert tab["ok"]
        observed = tab["result"]
        assert observed["schema"]["properties"]["gain"]
        assert observed["input"]["gain"] == text
        assert observed["errors"][0]["path"] == ["gain"]
        assert observed["errors"][0]["message"]

        invoke("connect", {"port": fx.service.port})
        assert (
            invoke("rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": tab_id}})
            == observed
        )
        observe_run_inputs(
            fx,
            tab_id,
            lambda method, params: call(sock, method, params)["result"],
        )
        before = fx.ctrl.get_tab_snapshot(tab_id)
        rejected = call(sock, "tab.run_start", {"tab_id": tab_id}, rid="run")
        assert not rejected["ok"]
        assert rejected["error"]["reason"] == "not_valid"
        after = fx.ctrl.get_tab_snapshot(tab_id)
        assert not after.interaction.is_running
        assert after.run.result is before.run.result
    finally:
        bridge.disconnect()
        sock.close()
