"""Library edit uses one application command without editor orchestration."""

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


@pytest.mark.parametrize("valid", [False, True])
def test_library_edit_delegates_once_and_reports_committed_prefix(tmp_path, valid):
    edits = [
        {"path": "length", "value": 0.25},
        {"path": "missing", "value": 1},
        {"path": "length", "value": 0.9},
    ]

    def respond(method, params):
        if method == "context.ml_get":
            if "name" in params:
                return {"cfg": {"length": 0.9 if valid else 0.25}}
            return {"modules": [], "waveforms": [{"name": "seed"}]}
        if method == "context.ml_edit":
            return {
                "valid": valid,
                "applied": 3 if valid else 1,
                "errors": []
                if valid
                else [{"path": "missing", "message": "unknown field"}],
            }
        raise AssertionError(method)

    client = make_client(tmp_path, respond)
    result = client.call("ml_edit", {"name": "seed", "edits": edits})
    commands = [
        (name, args)
        for name, args in client.transport.sent
        if name == "context.ml_edit"
    ]
    assert commands == [
        ("context.ml_edit", {"kind": "waveform", "name": "seed", "edits": edits})
    ]
    assert result["applied"] == (3 if valid else 1)
    assert result["skipped"] == ([] if valid else [2])
    assert result["failed"] == (
        None if valid else {"index": 1, "path": "missing", "message": "unknown field"}
    )
    assert result["cfg"]["length"] == (0.9 if valid else 0.25)
    assert not any(
        name.startswith("editor.") or name == "context.snapshot"
        for name, _ in client.transport.sent
    )


def test_library_edit_stale_is_not_retried_or_repaired(tmp_path):
    def respond(method, params):
        if method == "context.ml_get":
            return {"modules": [], "waveforms": [{"name": "seed"}]}
        raise AssertionError(method)

    client = make_client(tmp_path, respond)
    client.transport.replies["context.ml_edit"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_version",
            "message": "context changed",
        },
    }
    with pytest.raises(GuiRpcError) as error:
        client.call(
            "ml_edit", {"name": "seed", "edits": [{"path": "length", "value": 0.25}]}
        )
    assert error.value.reason == "stale_version"
    calls = [name for name, _ in client.transport.sent]
    assert calls.count("context.ml_edit") == 1
    assert calls.count("context.ml_get") == 1
    assert not any(
        name.startswith("editor.") or name == "context.snapshot" for name in calls
    )
