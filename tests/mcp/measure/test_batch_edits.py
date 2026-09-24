"""Batch tools validate before dispatch and retain ordered partial progress."""

from pathlib import Path
from typing import Any

import pytest

from ._support import make_client


@pytest.mark.parametrize(
    ("tool", "method", "arguments", "expected_params", "expected_result"),
    [
        (
            "gui_editor_set",
            "editor.set_field",
            {
                "editor_id": "ed1",
                "edits": [
                    {"path": "reps", "value": 100},
                    {"path": "sweep.gain.expts", "value": 5},
                ],
            },
            [
                {"editor_id": "ed1", "path": "reps", "value": 100},
                {"editor_id": "ed1", "path": "sweep.gain.expts", "value": 5},
            ],
            {"applied": 2, "valid": True},
        ),
        (
            "gui_context_md_write",
            "context.md_set_attr",
            {
                "attrs": [
                    {"key": "r_f", "value": 5000.0},
                    {"key": "q_f", "value": 200.0},
                ]
            },
            [{"key": "r_f", "value": 5000.0}, {"key": "q_f", "value": 200.0}],
            {"applied": 2},
        ),
    ],
)
def test_batch_edits_forward_in_order_without_trailing_product_reads(
    tmp_path: Path,
    tool: str,
    method: str,
    arguments: dict[str, Any],
    expected_params: list[dict[str, Any]],
    expected_result: dict[str, Any],
) -> None:
    client = make_client(tmp_path, lambda method, params: {"valid": True})
    assert client.call(tool, arguments) == expected_result
    sent = [
        (
            name,
            {key: value for key, value in params.items() if key != "expected_versions"},
        )
        for name, params in client.transport.sent
        if name != "resources.versions"
    ]
    assert sent == [(method, params) for params in expected_params]


@pytest.mark.parametrize("tool", ["gui_editor_set", "gui_context_md_write"])
def test_failed_batch_reports_progress_and_does_not_apply_later_items(
    tmp_path: Path,
    tool: str,
) -> None:
    editor = tool == "gui_editor_set"
    method = "editor.set_field" if editor else "context.md_set_attr"
    key = "path" if editor else "key"
    client = make_client(tmp_path)

    def reply(params: dict[str, Any]) -> dict[str, Any]:
        if params[key] == "bad":
            return {
                "ok": False,
                "error": {"code": "invalid_params", "message": "unknown path 'bad'"},
            }
        return {"ok": True, "result": {}}

    client.transport.replies[method] = reply
    items = [
        {key: name, "value": index} for index, name in enumerate(["ok", "bad", "never"])
    ]
    arguments = {"editor_id": "ed1", "edits": items} if editor else {"attrs": items}
    with pytest.raises(RuntimeError) as error:
        client.call(tool, arguments)
    assert [
        (name, params[key])
        for name, params in client.transport.sent
        if name != "resources.versions"
    ] == [(method, "ok"), (method, "bad")]
    message = str(error.value)
    assert "'bad'" in message
    if editor:
        assert "edits[1]" in message
        assert "1 edit(s) already applied" in message
    else:
        assert "applied_count=1" in message
        assert "failed_index=1" in message


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (
            "gui_editor_set",
            {
                "editor_id": "ed1",
                "edits": [{"path": "ok", "value": 1}, {"path": "reps"}],
            },
        ),
        ("gui_context_md_write", {"attrs": []}),
        (
            "gui_context_md_write",
            {"attrs": [{"key": "ok", "value": 1}, {"key": "bad"}]},
        ),
    ],
)
def test_malformed_batch_fails_before_any_rpc(
    tmp_path: Path, tool: str, arguments: dict[str, Any]
) -> None:
    client = make_client(tmp_path)
    with pytest.raises(ValueError):
        client.call(tool, arguments)
    assert client.transport.sent == []
