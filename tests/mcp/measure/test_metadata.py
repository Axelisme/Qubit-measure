"""Public MetaDict tools use GUI-owned compact reads and ordered writes."""

from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


def test_md_get_uses_gui_summary_unless_keys_are_explicit(tmp_path: Path) -> None:
    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "context.md_get":
            assert params == {"summaries": True}
            return {
                "keys": ["freq", "matrix"],
                "values": {"freq": 5.0, "matrix": "2 × 2 matrix"},
            }
        if method == "context.md_get_attr":
            assert params == {"key": "matrix"}
            return {"key": "matrix", "value": [[1, 2], [3, 4]]}
        raise AssertionError(method)

    client = make_client(tmp_path, reply)
    assert client.call("md_get", {}) == {
        "values": {"freq": 5.0, "matrix": "2 × 2 matrix"}
    }
    assert client.call("md_get", {"keys": ["matrix"]}) == {
        "values": {"matrix": [[1, 2], [3, 4]]}
    }
    assert client.call("md_get", {"keys": []}) == {"values": {}}
    assert [method for method, _ in client.transport.sent].count(
        "context.md_get_attr"
    ) == 1


def test_md_set_returns_owner_receipts_in_order(tmp_path: Path) -> None:
    sent: list[str] = []

    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "context.md_set_attr"
        assert params["receipt"] is True
        key = params["key"]
        sent.append(key)
        return {"before": None, "after": params["value"]}

    client = make_client(tmp_path, reply)
    assert client.call("md_set", {"values": {"a": 1, "b": [2]}}) == {
        "a": {"before": None, "after": 1},
        "b": {"before": None, "after": [2]},
    }
    assert sent == ["a", "b"]


def test_md_set_failure_exposes_confirmed_prefix_without_retry(tmp_path: Path) -> None:
    sent: list[str] = []

    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "context.md_set_attr"
        key = params["key"]
        sent.append(key)
        if key == "bad":
            return {
                "ok": False,
                "error": {
                    "code": "invalid_params",
                    "reason": "invalid_md_key",
                    "message": "bad key",
                },
            }
        return {"before": 1, "after": params["value"]}

    client = make_client(tmp_path)
    client.transport.replies["context.md_set_attr"] = lambda params: (
        reply("context.md_set_attr", params)
        if params["key"] == "bad"
        else {"ok": True, "result": reply("context.md_set_attr", params)}
    )
    with pytest.raises(GuiRpcError, match="bad.*a.*before.*after") as caught:
        client.call("md_set", {"values": {"a": 2, "bad": 3, "unreached": 4}})
    assert caught.value.reason == "invalid_md_key"
    assert sent == ["a", "bad"]
