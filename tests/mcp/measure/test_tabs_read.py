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


@pytest.mark.parametrize(
    "reason", ["invalid_data_file", "cleanup_failed", "stale_version"]
)
def test_tab_open_from_file_preserves_error_without_retry(
    tmp_path: Path, reason: str
) -> None:
    client = make_client(tmp_path)
    client.transport.replies["tab.open_file"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": reason,
            "message": "opening failed",
            "data": {"stale": ["context"]},
        },
    }
    with pytest.raises(GuiRpcError) as caught:
        client.call("tab_open", {"experiment": "ramsey", "from_file": "old.h5"})
    assert caught.value.reason == reason
    assert [
        item
        for item in client.transport.sent
        if item[0] not in ("wire.version", "rpc.catalog")
    ] == [("tab.open_file", {"adapter_name": "ramsey", "data_path": "old.h5"})]


@pytest.mark.parametrize("backfill", ["applied", "not_applied"])
def test_tab_open_from_file_is_one_application_operation(
    tmp_path: Path, backfill: str
) -> None:
    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "tab.open_file"
        assert params == {"adapter_name": "ramsey", "data_path": "saved.h5"}
        return {"tab_id": "loaded-tab", "cfg_backfill": backfill}

    client = make_client(tmp_path, reply)
    assert client.call(
        "tab_open", {"experiment": "ramsey", "from_file": "saved.h5"}
    ) == {
        "tab": "loaded-tab",
        "experiment": "ramsey",
        "cfg_backfill": backfill,
    }
    assert [
        item
        for item in client.transport.sent
        if item[0] not in ("wire.version", "rpc.catalog")
    ] == [("tab.open_file", {"adapter_name": "ramsey", "data_path": "saved.h5"})]


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
                        "has_post_analyze_result": False,
                    },
                    "result_source_path": "old.h5",
                    "result_state": {
                        "revision": 4,
                        "available": True,
                        "source_path": "old.h5",
                    },
                    "analysis_state": {
                        "revision": 2,
                        "available": False,
                        "has_figure": False,
                        "has_writeback_draft": False,
                    },
                    "post_analysis_state": {
                        "revision": 2,
                        "available": False,
                        "has_figure": False,
                        "has_writeback_draft": False,
                    },
                    "save_paths": {
                        "data_path": "next.h5",
                        "analysis_image_path": "analysis.png",
                        "post_analysis_image_path": "post.png",
                    },
                }
            ]
        }

    client = make_client(tmp_path, reply)
    result = client.call("tab_get", {"tab": "old-tab", "include": ["summary"]})
    assert (
        result["operation_state"]
        == reply("tab.snapshot", {"tab_id": "old-tab"})["tabs"][0]
    )
    assert result["summary"] == {
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
    assert not any(method == "tab.set_active" for method, _ in client.transport.sent)


@pytest.mark.parametrize("arguments", [{}, {"target": "window"}])
def test_screenshot_returns_png_path_without_changing_focus(
    tmp_path: Path, arguments: dict[str, str]
) -> None:
    png = b"\x89PNG\r\n\x1a\nimage"

    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "view.screenshot"
        path = Path(params["out_path"])
        path.write_bytes(png)
        return {"saved_to": str(path), "bytes": len(png)}

    client = make_client(tmp_path, reply)
    result = client.call("screenshot", arguments)
    path = Path(result["path"])
    assert path.is_file() and path.read_bytes() == png
    assert path.is_absolute()
    assert not any(method == "tab.set_active" for method, _ in client.transport.sent)
    client.context.session.cleanup_pngs()
    assert not path.exists()


@pytest.mark.parametrize(
    "target", ["setup", "device", "predictor", "inspect", "arb_waveform"]
)
def test_screenshot_dialog_targets_use_gui_path_without_inline_png(
    tmp_path: Path, target: str
) -> None:
    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "dialog.screenshot" and params["name"] == target
        path = Path(params["out_path"])
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return {"saved_to": str(path), "bytes": path.stat().st_size}

    client = make_client(tmp_path, reply)
    path = Path(client.call("screenshot", {"target": target})["path"])
    assert path.is_file() and path.read_bytes().startswith(b"\x89PNG")
    client.context.session.cleanup_pngs()
    assert not path.exists()


def test_tab_get_analyze_params_includes_definitions_and_current_values(
    tmp_path: Path,
) -> None:
    definitions = [{"name": "gain", "type": "float", "label": "Gain"}]

    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert params == {"tab_id": "old-tab"}
        if method == "tab.snapshot":
            return {
                "tabs": [
                    {
                        "tab_id": "old-tab",
                        "adapter_name": "ramsey",
                        "interaction": {"has_run_result": True},
                    }
                ]
            }
        if method == "tab.get_analyze_params":
            return {"analyze_params": {"gain": 1.25}, "definitions": definitions}
        if method == "tab.get_post_analyze_params":
            return {"post_analyze_params": None, "definitions": []}
        raise AssertionError(method)

    client = make_client(tmp_path, reply)
    assert client.call(
        "tab_get", {"tab": "old-tab", "include": ["analyze_params"]}
    ) == {
        "analyze_params": {
            "primary": {"definitions": definitions, "values": {"gain": 1.25}},
            "post": {"definitions": [], "values": None},
        }
    }
    assert not any(method == "tab.set_active" for method, _ in client.transport.sent)


def test_tab_get_marks_only_unfinished_cfg_and_artifact_owners(tmp_path: Path) -> None:
    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert params == {"tab_id": "old-tab"}
        if method == "tab.snapshot":
            return {
                "tabs": [
                    {
                        "tab_id": "old-tab",
                        "adapter_name": "ramsey",
                        "interaction": {"has_run_result": True},
                        "save_paths": {
                            "data_path": "data.h5",
                            "analysis_image_path": "analysis.png",
                            "post_analysis_image_path": "post.png",
                        },
                    }
                ]
            }
        if method == "tab.get_cfg":
            return {"tree": {"frequency": {"raw": "5", "resolved": 5}}}
        raise AssertionError(method)

    client = make_client(tmp_path, reply)
    result = client.call("tab_get", {"tab": "old-tab", "include": ["cfg", "artifacts"]})
    assert (
        result["operation_state"]
        == reply("tab.snapshot", {"tab_id": "old-tab"})["tabs"][0]
    )
    assert result["cfg"] == {"frequency": {"raw": "5", "resolved": 5}}
    assert result["artifacts"][0] == {
        "key": "data",
        "kind": "data",
        "default_path": "data.h5",
    }
    assert result["partial"] == {
        "cfg": "06-cfg-library owns aggregate type/choice/lock projection",
        "artifacts": "09-save-lifecycle owns status/last_saved_path",
    }


def test_tab_live_without_run_does_not_capture_figure(tmp_path: Path) -> None:
    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "tab.snapshot" and params == {"tab_id": "old-tab"}
        return {
            "tabs": [
                {
                    "tab_id": "old-tab",
                    "adapter_name": "ramsey",
                    "interaction": {"is_running": False, "has_run_result": False},
                }
            ]
        }

    client = make_client(tmp_path, reply)
    assert client.call("tab_live", {"tab": "old-tab"}) == {
        "running": False,
        "reason": "no_run",
        "operation_state": reply("tab.snapshot", {"tab_id": "old-tab"})["tabs"][0],
    }
    assert not any(method == "tab.set_active" for method, _ in client.transport.sent)


def test_tab_live_uses_gui_run_operation_elapsed_and_figure_path(
    tmp_path: Path,
) -> None:
    png = b"\x89PNG\r\n\x1a\nfigure"

    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.snapshot":
            assert params == {"tab_id": "live-tab"}
            return {
                "tabs": [
                    {
                        "tab_id": "live-tab",
                        "adapter_name": "ramsey",
                        "interaction": {"is_running": True, "has_run_result": False},
                    }
                ]
            }
        if method == "operation.active":
            return {"operations": [{"op": 7, "tab": "live-tab", "kind": "run"}]}
        if method == "operation.progress":
            assert params == {"operation_id": 7}
            return {
                "active": True,
                "elapsed_s": 3.25,
                "bars": [{"format": "Run 1/10", "percent": 10.0, "eta_s": 4.5}],
            }
        if method == "tab.get_figure":
            assert params["tab_id"] == "live-tab" and params["subtab_id"] == "run"
            path = Path(params["out_path"])
            path.write_bytes(png)
            return {"saved_to": str(path), "bytes": len(png)}
        raise AssertionError(method)

    client = make_client(tmp_path, reply)
    result = client.call("tab_live", {"tab": "live-tab"})
    assert (
        result["operation_state"]
        == reply("tab.snapshot", {"tab_id": "live-tab"})["tabs"][0]
    )
    assert result["running"] is True
    assert result["progress"] == [{"label": "Run 1/10", "percent": 10.0}]
    assert result["elapsed_s"] == 3.25 and result["eta_s"] == 4.5
    assert Path(result["figure"]).read_bytes() == png
    assert not any(method == "tab.set_active" for method, _ in client.transport.sent)
    client.context.session.cleanup_pngs()
    assert not Path(result["figure"]).exists()


def test_cfg_only_read_calls_only_its_resource(tmp_path: Path) -> None:
    def reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method == "tab.get_cfg" and params == {"tab_id": "old-tab"}
        return {"tree": {"frequency": {"value": 5.0}}}

    client = make_client(tmp_path, reply)
    client.context.session.ensure_connected()
    client.transport.sent.clear()
    result = client.call("tab_get", {"tab": "old-tab", "include": ["cfg"]})
    assert result["cfg"] == {"frequency": {"value": 5.0}}
    assert client.transport.sent == [("tab.get_cfg", {"tab_id": "old-tab"})]


def test_tab_get_rejects_invalid_include_items_before_read(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    with pytest.raises(ValueError, match="include must be"):
        client.call("tab_get", {"tab": "old-tab", "include": [{}]})
    assert not client.transport.sent


def test_screenshot_rejects_invalid_target_type_before_read(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    with pytest.raises(ValueError, match="screenshot target"):
        client.call("screenshot", {"target": []})
    assert not client.transport.sent
