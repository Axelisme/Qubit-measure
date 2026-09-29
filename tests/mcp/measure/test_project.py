"""Project tool reads and writes through the GUI owner."""

from pathlib import Path
from typing import Any

import pytest

from ._support import make_client


def test_project_read_and_partial_update_use_one_gui_owner_request(
    tmp_path: Path,
) -> None:
    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "project.info":
            assert params == {}
            return {
                "chip_name": "chip-a",
                "qub_name": "q1",
                "res_name": "res",
                "result_dir": "result/chip-a/q1",
                "database_path": "Database/chip-a/q1",
            }
        if method == "project.apply":
            assert params == {"chip_name": "chip-b"}
            return {
                "chip_name": "chip-b",
                "qub_name": "q1",
                "res_name": "res",
                "result_dir": "result/chip-b/q1",
                "database_path": "Database/chip-b/q1",
                "params_path": "result/chip-b/q1/params.json",
                "scope_id": "new-scope",
            }
        raise AssertionError(method)

    client = make_client(tmp_path, reply)
    assert client.call("project", {}) == {
        "chip": "chip-a",
        "qubit": "q1",
        "resonator": "res",
        "result_dir": "result/chip-a/q1",
        "database_path": "Database/chip-a/q1",
    }
    assert client.call("project", {"chip": "chip-b"}) == {
        "chip": "chip-b",
        "qubit": "q1",
        "resonator": "res",
        "result_dir": "result/chip-b/q1",
        "database_path": "Database/chip-b/q1",
    }
    assert ("project.apply", {"chip_name": "chip-b"}) in client.transport.sent
    # The update is one GUI owner request: only the explicit read above asked
    # for project.info, so the tool never reads the project to merge it itself.
    assert [method for method, _ in client.transport.sent].count("project.info") == 1


def test_project_no_project_errors_without_implicit_defaults(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    client.transport.replies["project.info"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "no_project",
            "message": "No project applied",
        },
    }
    client.transport.replies["project.apply"] = {
        "ok": False,
        "error": {
            "code": "invalid_params",
            "reason": "missing_project_fields",
            "message": "chip, qubit, resonator required",
        },
    }
    from zcu_tools.mcp.measure.session import GuiRpcError

    with pytest.raises(GuiRpcError, match="No project applied"):
        client.call("project", {})
    with pytest.raises(GuiRpcError, match="required"):
        client.call("project", {"chip": "only-chip"})
    assert ("project.apply", {"chip_name": "only-chip"}) in client.transport.sent
