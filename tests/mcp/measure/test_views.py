"""MCP figure files, screenshots, resource diagnostics, and overview products."""

import base64
import tempfile
from pathlib import Path
from typing import Any

import pytest

from ._support import make_client


@pytest.mark.parametrize("explicit_path", [False, True])
@pytest.mark.parametrize("pane", ["run", "analysis", "post_analysis"])
def test_figure_forwards_a_file_destination_and_the_requested_pane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_path: bool,
    pane: str,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client = make_client(
        tmp_path, lambda method, params: {"bytes": 1234, "saved_to": params["out_path"]}
    )
    arguments = {"tab_id": "t", "subtab_id": pane}
    if explicit_path:
        arguments["out_path"] = str(tmp_path / "custom.png")
    expected = str(
        tmp_path / ("custom.png" if explicit_path else f"measure_fig_t_{pane}.png")
    )
    assert client.call("gui_tab_get_figure", arguments) == {
        "bytes": 1234,
        "saved_to": expected,
    }
    assert [
        call for call in client.transport.sent if call[0] != "resources.versions"
    ] == [
        ("tab.get_figure", {"tab_id": "t", "subtab_id": pane, "out_path": expected}),
    ]


@pytest.mark.parametrize("target", ["setup", "device", "window"])
@pytest.mark.parametrize("explicit_path", [False, True])
def test_screenshot_decodes_png_into_its_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    explicit_path: bool,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    raw = b"PNGDATA"
    client = make_client(
        tmp_path,
        lambda method, params: {
            "png_b64": base64.b64encode(raw).decode("ascii"),
            "bytes": len(raw),
        },
    )
    arguments = {"target": target}
    if explicit_path:
        arguments["out_path"] = str(tmp_path / "custom.png")
    default_name = (
        "measure_window.png" if target == "window" else f"measure_dialog_{target}.png"
    )
    destination = tmp_path / ("custom.png" if explicit_path else default_name)
    assert client.call("gui_screenshot", arguments) == {
        "bytes": len(raw),
        "saved_to": str(destination),
    }
    assert destination.read_bytes() == raw
    expected_rpc = (
        ("view.screenshot", {})
        if target == "window"
        else ("dialog.screenshot", {"name": target})
    )
    assert [
        call for call in client.transport.sent if call[0] != "resources.versions"
    ] == [expected_rpc]


def test_debug_versions_returns_the_full_resource_table(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    versions = {"context": 3, "tab:t:cfg": 7, "soc": 1}
    client.transport.replies["resources.versions"] = {
        "ok": True,
        "result": {"versions": versions},
    }
    assert client.call("gui_debug_resource_versions", {}) == versions


@pytest.mark.parametrize("has_project", [False, True])
@pytest.mark.parametrize("has_soc", [False, True])
def test_overview_projects_gui_state_and_only_reads_available_products(
    tmp_path: Path,
    has_project: bool,
    has_soc: bool,
) -> None:
    project = {
        "chip_name": "Q5_2D",
        "qub_name": "Q1",
        "res_name": "R1",
        "result_dir": "/r",
        "database_path": "/db",
    }
    replies: dict[str, dict[str, Any]] = {
        "state.has_project": {"value": has_project},
        "state.has_context": {"value": True},
        "state.has_active_context": {"value": False},
        "state.has_soc": {"value": has_soc},
        "state.hardware_gate": {"active": []},
        "project.info": project,
        "context.active": {"label": "default"},
        "soc.info": {"is_mock": True},
        "run.running_tab": {"tab_id": "t2"},
        "view.snapshot": {"active_tab_id": "t1"},
        "tab.snapshot": {
            "tabs": [
                {
                    "tab_id": "t1",
                    "adapter_name": "Freq",
                    "interaction": {"is_running": False},
                },
                {
                    "tab_id": "t2",
                    "adapter_name": "Rabi",
                    "interaction": {"is_running": True},
                },
            ]
        },
    }
    client = make_client(tmp_path, lambda method, params: replies[method])
    assert client.call("gui_overview", {}) == {
        "state": {
            "has_project": has_project,
            "has_context": True,
            "has_active_context": False,
            "has_soc": has_soc,
        },
        "project": project if has_project else None,
        "context": "default",
        "soc": {"connected": has_soc, "is_mock": True if has_soc else None},
        "hardware_gate": {"active": []},
        "tabs": [
            {"tab_id": "t1", "adapter": "Freq", "is_running": False},
            {"tab_id": "t2", "adapter": "Rabi", "is_running": True},
        ],
        "running_tab": "t2",
        "active_tab": "t1",
    }
    methods = [method for method, _ in client.transport.sent]
    assert ("project.info" in methods) == has_project
    assert ("soc.info" in methods) == has_soc
