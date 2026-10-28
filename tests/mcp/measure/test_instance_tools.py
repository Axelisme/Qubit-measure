"""Fixed tool tables retain their own live GUI session and error contract."""

import json
import sys
from io import BytesIO, TextIOWrapper
from pathlib import Path

import pytest
from zcu_tools.gui import logging_setup
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.server import main
from zcu_tools.resources.entry.registry import ComponentRegistry, component_registry
from zcu_tools.resources.entry.schema import ComponentSchema

from tests._lab import SyntheticLab

from ._script_recipe_support import Inputs, Report
from ._support import make_client


def test_lab_registers_components_and_exposes_recipes_without_gui_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @recipe("synthetic_lab_recipe", "Return a synthetic report without a GUI")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        return Report(count=inputs.count)

    def register(registry: ComponentRegistry) -> None:
        registry.register("synthetic_mcp_component", ComponentSchema)

    lab = SyntheticLab[None](register_components=register, recipes=(synthetic,))
    monkeypatch.setattr(logging_setup, "setup_gui_logging", lambda **_kwargs: None)
    with (
        TextIOWrapper(
            BytesIO(b'{"jsonrpc":"2.0","id":7,"method":"tools/list"}\n'),
            encoding="utf-8",
        ) as incoming,
        TextIOWrapper(BytesIO(), encoding="utf-8") as outgoing,
    ):
        monkeypatch.setattr(sys, "stdin", incoming)
        monkeypatch.setattr(sys, "stdout", outgoing)
        try:
            main(lab, gui_module="synthetic.measure")
            outgoing.seek(0)
            response = json.loads(outgoing.read())
            tools = {tool["name"]: tool for tool in response["result"]["tools"]}
            assert tools["synthetic_lab_recipe"]["inputSchema"]["required"] == ["count"]
            assert component_registry.get("synthetic_mcp_component") is ComponentSchema
        finally:
            if "synthetic_mcp_component" in component_registry.kinds():
                component_registry.unregister("synthetic_mcp_component")


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
