"""Guide and editing-context behavior through the shipped MCP tool table."""

from dataclasses import dataclass, field
from typing import Any

import pytest
from zcu_tools.mcp.measure import server


@dataclass
class TabRpc:
    guides: list[str] = field(default_factory=list)
    next_tab: int = 0

    def __call__(
        self,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if method == "tab.new":
            self.next_tab += 1
            return {"tab_id": f"tab-{self.next_tab}"}
        if method == "tab.snapshot":
            return {"tabs": [{"editor_id": f"editor-{params['tab_id']}"}]}
        if method == "tab.get_cfg":
            return {"tree": {"gain": 0.25}}
        if method == "adapter.guide":
            name = str(params["adapter_name"])
            self.guides.append(name)
            return {"guide": f"Guide for {name}"}
        raise AssertionError(f"Unexpected RPC: {method}")


@pytest.fixture
def tab_rpc(monkeypatch: pytest.MonkeyPatch) -> TabRpc:
    rpc = TabRpc()
    monkeypatch.setattr(server, "send_gui_rpc", rpc)
    return rpc


@pytest.mark.parametrize("skip_guide", [None, False, True])
def test_open_returns_editing_context_and_requested_guide(
    tab_rpc: TabRpc, skip_guide: bool | None
) -> None:
    arguments: dict[str, Any] = {"adapter_name": "onetone"}
    if skip_guide is not None:
        arguments["skip_guide"] = skip_guide

    result = server.TOOLS["gui_tab_open"]["handler"](arguments)

    expected: dict[str, Any] = {
        "tab_id": "tab-1",
        "adapter": "onetone",
        "editor_id": "editor-tab-1",
        "tree": {"gain": 0.25},
    }
    if skip_guide:
        expected["guide_omitted"] = True
        assert tab_rpc.guides == []
    else:
        expected["guide"] = "Guide for onetone"
        assert tab_rpc.guides == ["onetone"]
    assert result == expected


@pytest.mark.parametrize("adapters", [("onetone", "onetone"), ("onetone", "rabi")])
def test_each_open_returns_its_guide_regardless_of_prior_calls(
    tab_rpc: TabRpc, adapters: tuple[str, str]
) -> None:
    open_tab = server.TOOLS["gui_tab_open"]["handler"]

    results = [open_tab({"adapter_name": name}) for name in adapters]

    assert [result["guide"] for result in results] == [
        f"Guide for {name}" for name in adapters
    ]
    assert [result["tab_id"] for result in results] == ["tab-1", "tab-2"]
    assert all("guide_omitted" not in result for result in results)
    assert tab_rpc.guides == list(adapters)
