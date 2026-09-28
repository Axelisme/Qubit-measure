"""Public synchronous SoC tools use the GUI's current hardware state."""

from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


def test_soc_info_disconnected_does_not_request_a_soc_cfg(tmp_path: Path) -> None:
    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "state.has_soc" and params == {}
        return {"value": False}

    client = make_client(tmp_path, reply)
    assert client.call("soc_info", {}) == {
        "connected": False,
        "address": None,
        "port": None,
        "description": None,
        "is_mock": False,
    }
    assert "soc.info" not in [method for method, _ in client.transport.sent]


def test_soc_info_opt_in_cfg_and_sync_connect_reads_same_gui_state(
    tmp_path: Path,
) -> None:
    methods: list[tuple[str, dict[str, Any]]] = []

    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        methods.append((method, params))
        if method == "soc.connect":
            assert params == {"kind": "remote", "ip": "192.0.2.1", "port": 8888}
            return {"soc": {"description": "connected", "is_mock": False}}
        if method == "state.has_soc":
            return {"value": True}
        if method == "soc.info":
            result: dict[str, Any] = {
                "description": "generator/readout channel table",
                "is_mock": False,
                "address": "192.0.2.1",
                "port": 8888,
            }
            if params["include_cfg"]:
                result["cfg"] = {"gens": [{"fs": 6000.0}]}
            return result
        raise AssertionError(method)

    client = make_client(tmp_path, reply)
    expected = {
        "connected": True,
        "description": "generator/readout channel table",
        "is_mock": False,
        "address": "192.0.2.1",
        "port": 8888,
    }
    assert (
        client.call("soc_connect", {"address": "192.0.2.1", "port": 8888}) == expected
    )
    assert client.call("soc_info", {"include_cfg": True}) == {
        **expected,
        "cfg": {"gens": [{"fs": 6000.0}]},
    }
    assert methods == [
        ("soc.connect", {"kind": "remote", "ip": "192.0.2.1", "port": 8888}),
        ("state.has_soc", {}),
        ("soc.info", {"include_cfg": False}),
        ("state.has_soc", {}),
        ("soc.info", {"include_cfg": True}),
    ]


def test_soc_connect_failure_does_not_retry_or_report_a_new_state(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    client.transport.replies["soc.connect"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "busy",
            "message": "SoC connect is busy",
        },
    }
    with pytest.raises(GuiRpcError, match="busy") as caught:
        client.call("soc_connect", {"address": "192.0.2.1", "port": 8888})
    assert caught.value.reason == "busy"
    assert [method for method, _ in client.transport.sent].count("soc.connect") == 1
    assert "soc.info" not in [method for method, _ in client.transport.sent]
