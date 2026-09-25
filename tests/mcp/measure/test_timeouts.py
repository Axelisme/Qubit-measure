"""Timeout policy at the context/bridge and assembled-tool seams."""

from pathlib import Path
from typing import Any

import pytest
from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


@pytest.mark.parametrize(
    ("tool", "arguments", "method"),
    [
        ("gui_tab_get_cfg", {"tab_id": "t"}, "tab.get_cfg"),
        ("gui_soc_connect", {"kind": "mock"}, "soc.connect"),
    ],
)
def test_generated_and_manual_tools_use_method_timeout_plus_transport_slack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    arguments: dict[str, Any],
    method: str,
) -> None:
    client = make_client(tmp_path)
    calls: list[tuple[str, dict[str, Any], float]] = []

    def send_rpc_raw(
        method: str, params: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        calls.append((method, params, timeout_seconds))
        return {"ok": True, "result": {"soc": {"is_mock": True}}}

    monkeypatch.setattr(client.context.bridge, "send_rpc_raw", send_rpc_raw)
    client.call(tool, arguments)
    sent_method, sent_params, timeout = calls[0]
    assert (sent_method, sent_params) == (method, arguments)
    assert METHOD_SPECS[method].timeout_seconds < timeout < 30.0


@pytest.mark.parametrize("method", ["operation.await", "notify.await"])
def test_dynamic_wait_requires_an_explicit_transport_timeout(
    tmp_path: Path, method: str
) -> None:
    client = make_client(tmp_path)
    with pytest.raises(ValueError, match="requires explicit timeout_seconds"):
        client.context.send_gui_rpc(method, {})
    assert client.transport.sent == []


def test_handler_timeout_is_distinct_from_transport_timeout(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.transport.replies["state.has_soc"] = {
        "ok": False,
        "error": {
            "code": "timeout",
            "message": "handler did not finish",
        },
    }
    with pytest.raises(GuiRpcError) as error:
        client.context.send_gui_rpc("state.has_soc", {})
    assert (error.value.code, error.value.reason) == ("timeout", "gui_handler_timeout")
    assert client.transport.is_open


@pytest.mark.parametrize("tool", [None, "gui_op_wait"])
def test_transport_timeout_is_not_downgraded_to_a_bounded_wait_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str | None,
) -> None:
    client = make_client(tmp_path)

    def send_rpc_raw(
        method: str, params: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        raise GuiTransportTimeoutError(method, timeout_seconds)

    monkeypatch.setattr(client.context.bridge, "send_rpc_raw", send_rpc_raw)
    with pytest.raises(GuiRpcError) as error:
        if tool is None:
            client.context.send_gui_rpc("state.has_soc", {})
        else:
            client.call(tool, {"handle": 5, "timeout": 0.05})
    assert (error.value.code, error.value.reason) == (
        "timeout",
        "gui_transport_timeout",
    )


def test_soc_connect_reconciles_a_late_reply_through_bounded_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path)
    calls: list[tuple[str, float]] = []

    def send_rpc_raw(
        method: str, params: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        calls.append((method, timeout_seconds))
        if method == "soc.connect":
            return {
                "ok": False,
                "error": {"code": "timeout", "message": "late connect reply"},
            }
        if method == "state.has_soc":
            return {"ok": True, "result": {"value": True}}
        if method == "soc.info":
            return {
                "ok": True,
                "result": {"description": "connected after timeout", "is_mock": False},
            }
        if method == "resources.versions":
            return {"ok": True, "result": {"versions": {}}}
        raise AssertionError(method)

    monkeypatch.setattr(client.context.bridge, "send_rpc_raw", send_rpc_raw)
    result = client.call(
        "gui_soc_connect", {"kind": "remote", "ip": "192.0.2.1", "port": 8888}
    )
    assert result["soc"] == {"description": "connected after timeout", "is_mock": False}
    assert "warning" in result
    primary = [
        (method, timeout) for method, timeout in calls if method != "resources.versions"
    ]
    assert [method for method, _ in primary] == [
        "soc.connect",
        "state.has_soc",
        "soc.info",
    ]
    assert METHOD_SPECS["soc.connect"].timeout_seconds < primary[0][1] < 30.0
    assert all(0 < timeout < primary[0][1] for _, timeout in primary[1:])
