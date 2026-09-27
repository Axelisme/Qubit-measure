"""Public read/open/screenshot contracts on a GUI-owned measure session."""

from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


def test_experiments_and_guide_read_live_adapter_descriptions(tmp_path: Path) -> None:
    guide = {
        "behavior": "Measure a decay. Inspect the result before calibration.",
        "expects_md": [],
        "expects_ml": [],
        "typical_writeback": [],
        "recommended": [],
    }

    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "adapter.list":
            return {"adapters": ["ramsey", "t1"]}
        if method == "adapter.guide":
            assert params == {"adapter_name": "ramsey"}
            return {"guide": guide}
        raise AssertionError(method)

    client = make_client(tmp_path, reply)
    assert client.call("experiments", {"prefix": "ram"}) == {
        "experiments": [{"name": "ramsey", "summary": "Measure a decay."}]
    }
    assert client.call("guide", {"experiment": "ramsey"}) == guide
    assert [method for method, _ in client.transport.sent].count("adapter.guide") == 2


def test_tab_open_failed_load_closes_new_tab_without_soc(tmp_path: Path) -> None:
    tab = "new-tab"

    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.new":
            return {
                "tab_id": tab,
                "__agent_write_versions": {f"tab:{tab}": [0, 1]},
            }
        if method == "tab.snapshot":
            return {"tabs": [{"tab_id": tab, "adapter_name": "ramsey"}]}
        if method == "context.snapshot":
            return {"label": None}
        if method == "tab.get_analyze_result":
            return {"summary": None}
        if method == "tab.close":
            return {"ok": True}
        raise AssertionError(method)

    client = make_client(tmp_path, reply)
    client.transport.replies["resources.versions"] = {
        "ok": True,
        "result": {
            "versions": {
                f"tab:{tab}": 1,
                f"tab:{tab}:result": 0,
                f"tab:{tab}:analyze": 0,
                "context": 0,
            }
        },
    }
    client.transport.replies["tab.load_data"] = {
        "ok": False,
        "error": {
            "code": "invalid_params",
            "reason": "incompatible_data",
            "message": "wrong experiment",
        },
    }

    with pytest.raises(GuiRpcError, match="wrong experiment"):
        client.call("tab_open", {"experiment": "ramsey", "from_file": "old.h5"})
    methods = [method for method, _ in client.transport.sent]
    assert (
        methods.index("tab.new")
        < methods.index("tab.load_data")
        < methods.index("tab.close")
    )
    assert "soc.connect" not in methods


def test_tab_get_summary_reads_explicit_tab_without_changing_focus(
    tmp_path: Path,
) -> None:
    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "tab.snapshot"
        assert params == {"tab_id": "old-tab"}
        return {
            "tabs": [
                {
                    "tab_id": "old-tab",
                    "adapter_name": "ramsey",
                    "interaction": {
                        "is_running": False,
                        "is_analyzing": False,
                        "has_run_result": True,
                        "has_analyze_result": False,
                    },
                    "result_source_path": "old.h5",
                }
            ]
        }

    client = make_client(tmp_path, reply)
    assert client.call("tab_get", {"tab": "old-tab", "include": ["summary"]}) == {
        "summary": {
            "experiment": "ramsey",
            "state": {
                "running": False,
                "analyzing": False,
                "has_result": True,
                "has_analysis": False,
                "has_post": False,
            },
            "source_file": "old.h5",
        }
    }
    assert not any(method == "tab.set_active" for method, _ in client.transport.sent)


def test_screenshot_returns_png_path_without_changing_focus(tmp_path: Path) -> None:
    png = b"\x89PNG\r\n\x1a\nimage"

    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "view.screenshot"
        path = Path(params["out_path"])
        path.write_bytes(png)
        return {"saved_to": str(path), "bytes": len(png)}

    client = make_client(tmp_path, reply)
    result = client.call("screenshot", {"target": "window"})
    path = Path(result["path"])
    assert path.is_file() and path.read_bytes() == png
    assert path.is_absolute()
    assert not any(method == "tab.set_active" for method, _ in client.transport.sent)
