"""Fixed tool tables retain their own live GUI session and error contract."""

from pathlib import Path

import pytest

from ._support import make_client


def test_fixed_tool_contract_is_available_without_gui_access(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    try:
        assert set(client.tools) == {
            "connect",
            "status",
            "wait",
            "cancel",
            "rpc_list",
            "rpc_describe",
            "rpc_call",
            "tab_analyze",
            "tab_interact",
            "tab_close",
            "answer",
            "apply_writeback",
            "device_setup",
            "recipe_guide",
            "workflow_open",
            "workflow_close",
            "workflow_status",
            "workflow_start",
            "workflow_pause",
            "workflow_resume",
            "workflow_stop",
            "workflow_form",
            "workflow_describe_form",
            "workflow_edit_form",
            "workflow_tunables",
            "workflow_records",
            "workflow_figures",
            "workflow_figure",
            "workflow_answer",
        }
        assert all(callable(entry["handler"]) for entry in client.tools.values())
        assert client.transport.sent == []
    finally:
        client.context.session.close()


def test_fixed_rpc_tools_keep_their_own_transport_and_catalog(tmp_path: Path) -> None:
    first = make_client(tmp_path)
    second = make_client(tmp_path)
    for client, value in ((first, "A"), (second, "B")):
        client.transport.replies["adapter.list"] = {
            "ok": True,
            "result": {"adapters": [value]},
        }
        client.transport.replies["rpc.catalog"] = {
            "ok": True,
            "result": {
                "methods": [
                    {
                        "method": "adapter.list",
                        "description": f"Catalog {value}",
                        "params": {"type": "object", "properties": {}},
                        "timeout_seconds": 5.0,
                        "exposure": "rpc",
                        "tool_names": [],
                        "operation_key": None,
                    }
                ],
            },
        }

    assert first.call("rpc_call", {"method": "adapter.list"}) == {"adapters": ["A"]}
    assert second.call("rpc_call", {"method": "adapter.list"}) == {"adapters": ["B"]}
    assert (
        first.call("rpc_describe", {"method": "adapter.list"})["description"]
        == "Catalog A"
    )
    assert (
        second.call("rpc_describe", {"method": "adapter.list"})["description"]
        == "Catalog B"
    )
    assert [method for method, _ in first.transport.sent].count("adapter.list") == 1
    assert [method for method, _ in second.transport.sent].count("adapter.list") == 1


def test_gui_errors_preserve_wire_code_and_reason(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.transport.replies["adapter.list"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "invalid_data_file",
            "message": "invalid input",
        },
    }
    with pytest.raises(RuntimeError, match="invalid input") as error:
        client.call("rpc_call", {"method": "adapter.list"})
    assert (
        getattr(error.value, "code", None),
        getattr(error.value, "reason", None),
    ) == (
        "precondition_failed",
        "invalid_data_file",
    )
