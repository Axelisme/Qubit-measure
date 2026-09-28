"""Public context selection and creation on the shared GUI state."""

from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


def test_contexts_list_and_use_gui_labels(tmp_path: Path) -> None:
    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "context.labels":
            return {"labels": ["base", "copy"]}
        if method == "context.active":
            return {"label": "base"}
        if method == "context.use":
            assert params == {"label": "copy"}
            return {"label": "copy", "has_active_context": True}
        raise AssertionError(method)

    client = make_client(tmp_path, reply)
    assert client.call("contexts", {}) == {"active": "base", "labels": ["base", "copy"]}
    assert client.call("context_use", {"label": "copy"}) == {"label": "copy"}
    assert ("context.use", {"label": "copy"}) in client.transport.sent


def test_context_create_distinguishes_current_default_from_empty(
    tmp_path: Path,
) -> None:
    sent: list[dict[str, Any]] = []

    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "context.new":
            sent.append(params)
            return {"label": params["label"], "has_active_context": True}
        raise AssertionError(method)

    client = make_client(tmp_path, reply)
    assert client.call("context_create", {"label": "copy"}) == {"label": "copy"}
    assert client.call("context_create", {"label": "empty", "clone_from": None}) == {
        "label": "empty"
    }
    assert sent == [
        {"label": "copy", "bind_device": None, "clone_from": "current"},
        {"label": "empty", "bind_device": None, "clone_from": None},
    ]


def test_context_use_unknown_label_keeps_gui_error(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.transport.replies["context.use"] = {
        "ok": False,
        "error": {
            "code": "invalid_params",
            "reason": "unknown_context",
            "message": "unknown context label 'ghost'; available: ['base']",
        },
    }
    with pytest.raises(GuiRpcError, match="available: \\['base'\\]"):
        client.call("context_use", {"label": "ghost"})
