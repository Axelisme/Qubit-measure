"""Writeback tools use one GUI-owned preview or batch command."""

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


@pytest.mark.parametrize(
    "stage,pane", [("primary", "analysis"), ("post", "post_analysis")]
)
def test_preview_projects_complete_gui_values(tmp_path, stage, pane):
    client = make_client(tmp_path)
    client.transport.replies["tab.writeback_preview"] = {
        "ok": True,
        "result": {
            "destination_context": {"active_label": "ctx"},
            "items": [
                {
                    "id": "md-1",
                    "kind": "metadict",
                    "target_name": "freq",
                    "description": "Frequency",
                    "current": [1, 2],
                    "proposed": [3, 4],
                }
            ],
        },
    }
    result = client.call("writeback", {"tab": "t", "stage": stage})
    assert result == {
        "destination": {"active_label": "ctx"},
        "items": [
            {
                "id": "md-1",
                "kind": "md",
                "target": "freq",
                "description": "Frequency",
                "current": [1, 2],
                "proposed": [3, 4],
            }
        ],
    }
    assert [
        entry for entry in client.transport.sent if entry[0].startswith("tab.")
    ] == [
        ("tab.writeback_preview", {"tab_id": "t", "subtab_id": pane}),
    ]


@pytest.mark.parametrize("write", [[], [{"id": "md-1", "value": None}]])
def test_write_is_one_command_without_hidden_reads(tmp_path, write):
    client = make_client(tmp_path)
    reply = {
        "written": [
            {
                "id": "ml-1",
                "kind": "module",
                "target": "same",
                "before": {},
                "after": {"gain": 0.5},
            },
            {
                "id": "wf-1",
                "kind": "waveform",
                "target": "same",
                "before": None,
                "after": {"length": 2},
            },
        ]
    }
    client.transport.replies["tab.writeback_write"] = {"ok": True, "result": reply}
    assert client.call("writeback", {"tab": "t", "write": write}) == reply
    domain_calls = [
        entry
        for entry in client.transport.sent
        if entry[0]
        not in (
            "wire.version",
            "rpc.catalog",
            "resources.versions",
        )
    ]
    assert domain_calls == [
        (
            "tab.writeback_write",
            {
                "tab_id": "t",
                "subtab_id": "analysis",
                "write": write,
            },
        )
    ]


def test_write_failure_is_not_retried(tmp_path):
    client = make_client(tmp_path)
    client.transport.replies["tab.writeback_write"] = {
        "ok": False,
        "error": {"code": "precondition_failed", "message": "draft unavailable"},
    }
    with pytest.raises(GuiRpcError, match="draft unavailable"):
        client.call("writeback", {"tab": "t", "write": [{"id": "md-1"}]})
    assert (
        sum(method == "tab.writeback_write" for method, _ in client.transport.sent) == 1
    )


def test_invalid_stage_fails_before_transport(tmp_path):
    client = make_client(tmp_path)
    with pytest.raises(ValueError, match="stage"):
        client.call("writeback", {"tab": "t", "stage": "run"})
    assert client.transport.sent == []
