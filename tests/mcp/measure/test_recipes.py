"""Recipe promises through the shipped tool table and GUI transport seam."""

from copy import deepcopy
from typing import Any

import pytest
from zcu_tools.mcp.core.reply import ToolReply

from ._support import make_client


@pytest.mark.parametrize("reuse_tab_id", [None, "kept"])
def test_lookback_missing_frequency_does_not_run_a_blind_default(
    tmp_path, reuse_tab_id
):
    tab = reuse_tab_id or "new"
    publication = {
        "cfg_ref": {"cfg_id": "cfg", "revision": "2"},
        "status": "Valid",
        "source_basis": [],
        "diagnostics": [],
        "tree": {
            "kind": "section",
            "valid": True,
            "children": {
                "modules": {
                    "kind": "section",
                    "valid": True,
                    "children": {
                        "readout": {
                            "kind": "reference",
                            "ref": None,
                            "valid": True,
                            "children": {},
                        },
                    },
                },
            },
        },
    }

    def respond(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "context.snapshot":
            return {"label": "sample", "md": {}, "ml": {"modules": {}, "waveforms": {}}}
        if method == "tab.new":
            return {"tab_id": tab}
        if method == "tab.snapshot":
            return {
                "tabs": [
                    {
                        "tab_id": tab,
                        "adapter_name": "lookback",
                        "interaction": {
                            "is_running": False,
                            "is_analyzing": False,
                            "is_saving_data": False,
                        },
                    }
                ]
            }
        if method in ("tab.get_cfg", "tab.reset_cfg", "tab.edit_cfg"):
            return deepcopy(publication)
        raise AssertionError(f"Missing frequency must not start work: {method}")

    client = make_client(tmp_path, respond)
    try:
        arguments = {} if reuse_tab_id is None else {"reuse_tab_id": reuse_tab_id}
        reply = client.call("lookback", arguments)
        assert isinstance(reply, ToolReply)
        assert reply.data["status"] == "needs_parameters"
        assert reply.data["tab"] == tab
        assert [item["parameter"] for item in reply.data["missing"]] == [
            "frequency_mhz"
        ]
        assert reply.is_error is False
        methods = [method for method, _ in client.transport.sent]
        assert "tab.run_start" not in methods
        assert ("tab.new" in methods) is (reuse_tab_id is None)
        assert ("tab.reset_cfg" in methods) is (reuse_tab_id is not None)
    finally:
        client.context.session.close()
