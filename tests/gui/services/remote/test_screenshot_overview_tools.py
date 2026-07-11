"""MCP screenshot, debug, and overview tool tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.remote.handlers.state_project import (
    _h_state_hardware_gate,
)
from zcu_tools.gui.session.events import GatePresence
from zcu_tools.mcp.measure import server as mcp_server


@pytest.fixture(autouse=True)
def _clear_mcp_policy_state():
    mcp_server._SESSION.clear_policy_state()
    yield
    mcp_server._SESSION.clear_policy_state()


# ---------------------------------------------------------------------------
# Figure/screenshot consolidation (WIRE 24) — run/analyze replies NO LONGER fold
# figure_path; looking at a plot is a separate gui_tab_get_current_figure call.
# ---------------------------------------------------------------------------


def test_get_current_figure_omitted_out_path_writes_temp_file(monkeypatch):
    """Omitting out_path must drive the wire in out_path mode (synthesised temp
    path) and return {saved_to, bytes} with NO inline base64."""
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        # Mirror the wire out_path branch: write nothing here, just echo saved_to.
        return {"bytes": 1234, "saved_to": params["out_path"]}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_get_current_figure"]["handler"](
        {"tab_id": "fake-freq-1"}
    )

    expected_path = str(Path(gettempdir()) / "measure_fig_fake-freq-1.png")
    assert out == {"bytes": 1234, "saved_to": expected_path}
    assert "png_b64" not in out
    # The convenience layer forwarded an out_path so the wire never returns base64.
    assert calls == [
        ("tab.get_current_figure", {"tab_id": "fake-freq-1", "out_path": expected_path})
    ]


def test_get_current_figure_explicit_out_path_is_forwarded(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {"bytes": 1234, "saved_to": params["out_path"]}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_get_current_figure"]["handler"](
        {"tab_id": "t1", "out_path": "/tmp/custom.png"}
    )

    assert out == {"bytes": 1234, "saved_to": "/tmp/custom.png"}
    assert calls == [
        ("tab.get_current_figure", {"tab_id": "t1", "out_path": "/tmp/custom.png"})
    ]


def test_screenshot_dialog_omitted_out_path_writes_temp_file(monkeypatch):
    """target=<dialog name> still works: decode + write a per-dialog temp PNG and
    return {saved_to, bytes} with NO inline base64 (the old dialog behaviour)."""
    import base64
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    raw = b"PNGDATA"
    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {"png_b64": base64.b64encode(raw).decode("ascii"), "bytes": len(raw)}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_screenshot"]["handler"]({"target": "setup"})

    expected_path = str(Path(gettempdir()) / "measure_dialog_setup.png")
    assert out == {"bytes": len(raw), "saved_to": expected_path}
    assert "png_b64" not in out
    assert Path(expected_path).read_bytes() == raw
    # A dialog target forwards dialog.screenshot with only 'name'.
    assert calls == [("dialog.screenshot", {"name": "setup"})]


def test_screenshot_schema_includes_arb_waveform_dialog():
    from zcu_tools.mcp.measure import server as mcp_server

    target = mcp_server.TOOLS["gui_screenshot"]["inputSchema"]["properties"]["target"]
    assert "arb_waveform" in target["enum"]
    assert "arb_waveform" in target["description"]
    assert "arb_waveform" in mcp_server.TOOLS["gui_screenshot"]["description"]


def test_screenshot_dialog_explicit_out_path(monkeypatch, tmp_path):
    import base64

    from zcu_tools.mcp.measure import server as mcp_server

    raw = b"X"
    target = tmp_path / "shot.png"

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds, method, params
        return {"png_b64": base64.b64encode(raw).decode("ascii"), "bytes": len(raw)}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_screenshot"]["handler"](
        {"target": "device", "out_path": str(target)}
    )

    assert out == {"bytes": len(raw), "saved_to": str(target)}
    assert target.read_bytes() == raw


def test_screenshot_window_writes_window_png(monkeypatch):
    """target='window' drives the view.screenshot wire method and writes the single
    measure_window.png temp file (the whole-window grab; no per-name)."""
    import base64
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    raw = b"WINDOWPNG"
    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {"png_b64": base64.b64encode(raw).decode("ascii"), "bytes": len(raw)}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_screenshot"]["handler"]({"target": "window"})

    expected_path = str(Path(gettempdir()) / "measure_window.png")
    assert out == {"bytes": len(raw), "saved_to": expected_path}
    assert "png_b64" not in out
    assert Path(expected_path).read_bytes() == raw
    # The window target forwards view.screenshot with NO params.
    assert calls == [("view.screenshot", {})]


def test_debug_versions_dumps_resource_table(monkeypatch):
    """gui_debug_resource_versions returns the full resources.versions table verbatim."""
    from zcu_tools.mcp.measure import server as mcp_server

    table = {"context": 3, "tab:t1:cfg": 7, "soc": 1}

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del params, timeout_seconds
        assert method == "resources.versions"
        return {"versions": table}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_debug_resource_versions"]["handler"]({})
    # P1 flattened the reply: the {versions: ...} wrapper is dropped, the table is
    # returned verbatim as a flat {resource_key: int} map.
    assert out == table


def test_debug_operations_dumps_handle_cache(monkeypatch):
    """gui_debug_operations dumps ONLY the mcp-side _OP_BY_KEY cache, reshaped to
    {handles: {key: {operation_id}}}. The live device enumeration is no longer
    duplicated here (gui_device_list_operations owns it), so the tool makes NO wire
    call (it never touches device.active_operations)."""
    from zcu_tools.mcp.measure import server as mcp_server

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del params, timeout_seconds
        # P4: no device enumeration here — any wire call is a regression.
        raise AssertionError(f"gui_debug_operations must not call the wire: {method}")

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    mcp_server._SESSION.operation_handles.update({"tab:t1": 42, "device:flux": 7})
    out = mcp_server.TOOLS["gui_debug_operations"]["handler"]({})
    assert out == {
        "handles": {
            "tab:t1": {"operation_id": 42},
            "device:flux": {"operation_id": 7},
        }
    }


def test_server_instructions_present_three_tiers():
    """_SERVER_INSTRUCTIONS groups the tools under the three tiers (light check)."""
    from zcu_tools.mcp.measure import server as mcp_server

    text = mcp_server._SERVER_INSTRUCTIONS
    assert "RECOMMENDED" in text
    assert "ON-DEMAND" in text
    assert "DEV" in text
    # All three DEV tools are named in the instructions (gui_debug_screenshot was
    # renamed to gui_screenshot in Phase 171 P1; gui_debug_versions renamed to
    # gui_debug_resource_versions in Phase 171 polish).
    assert "gui_screenshot" in text
    assert "gui_debug_resource_versions" in text
    assert "gui_debug_operations" in text


def _overview_fake_send(*, has_soc: bool):
    """A send_gui_rpc stub answering the read RPCs _assemble_overview fans out
    over, parameterised by SoC presence (soc.info only valid while connected)."""
    flags = {
        "state.has_project": {"value": True},
        "state.has_context": {"value": True},
        "state.has_active_context": {"value": False},
        "state.has_soc": {"value": has_soc},
        "state.hardware_gate": {"active": []},
        "project.info": {
            "chip_name": "Q5_2D",
            "qub_name": "Q1",
            "res_name": "R1",
            "result_dir": "/r",
            "database_path": "/db",
        },
        "context.active": {"label": "default"},
        "soc.info": {"description": "desc", "is_mock": True, "cfg": {}},
        "run.running_tab": {"tab_id": "t2"},
        "view.snapshot": {"active_tab_id": "t1", "context_label": "default"},
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

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del params, timeout_seconds
        if method not in flags:
            raise AssertionError(f"unexpected overview RPC: {method}")
        return flags[method]

    return fake_send


def test_overview_assembles_from_read_rpcs_with_project(monkeypatch):
    """gui_overview packs state / project / context / soc / tabs / running_tab /
    active_tab from existing reads; with a project applied, project uses the full
    wire shape {chip_name, qub_name, res_name, result_dir, database_path} — the
    overview is the single orientation SSOT, folding in the project paths so the
    retired gui_project_info tool has no separate surface."""
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "send_gui_rpc", _overview_fake_send(has_soc=True))
    out = mcp_server.TOOLS["gui_overview"]["handler"]({})

    assert out == {
        "state": {
            "has_project": True,
            "has_context": True,
            "has_active_context": False,
            "has_soc": True,
        },
        "project": {
            "chip_name": "Q5_2D",
            "qub_name": "Q1",
            "res_name": "R1",
            "result_dir": "/r",
            "database_path": "/db",
        },
        "context": "default",
        "soc": {"connected": True, "is_mock": True},
        "hardware_gate": {"active": []},
        "tabs": [
            {"tab_id": "t1", "adapter": "Freq", "is_running": False},
            {"tab_id": "t2", "adapter": "Rabi", "is_running": True},
        ],
        "running_tab": "t2",
        "active_tab": "t1",
    }


def test_hardware_gate_rpc_projects_presence_without_monotonic_epoch() -> None:
    ctrl = MagicMock()
    ctrl.get_hardware_gate_presence.return_value = (
        GatePresence(
            kind="run",
            origin_kind="agent",
            note="run T1 (tab t1)",
            active_for_seconds=1.25,
        ),
    )
    adapter = cast(Any, SimpleNamespace(ctrl=ctrl))

    assert _h_state_hardware_gate(adapter, {}) == {
        "active": [
            {
                "kind": "run",
                "origin_kind": "agent",
                "note": "run T1 (tab t1)",
                "active_for_seconds": 1.25,
            }
        ]
    }


def test_overview_project_is_null_when_no_project(monkeypatch):
    """project.info fast-fails no_project without a project, so the overview
    reports project=None and never queries project.info."""
    from zcu_tools.mcp.measure import server as mcp_server

    queried: list[str] = []
    base = _overview_fake_send(has_soc=True)

    def tracking(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        queried.append(method)
        if method == "state.has_project":
            return {"value": False}
        return base(method, params, timeout_seconds)

    monkeypatch.setattr(mcp_server, "send_gui_rpc", tracking)
    out = mcp_server.TOOLS["gui_overview"]["handler"]({})

    assert out["project"] is None
    assert "project.info" not in queried


def test_overview_skips_soc_info_when_not_connected(monkeypatch):
    """soc.info fast-fails without a SoC, so is_mock stays None and soc.info is
    never queried when has_soc is False."""
    from zcu_tools.mcp.measure import server as mcp_server

    queried: list[str] = []
    base = _overview_fake_send(has_soc=False)

    def tracking(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        queried.append(method)
        return base(method, params, timeout_seconds)

    monkeypatch.setattr(mcp_server, "send_gui_rpc", tracking)
    out = mcp_server.TOOLS["gui_overview"]["handler"]({})

    assert out["soc"] == {"connected": False, "is_mock": None}
    assert "soc.info" not in queried


def test_connect_folds_overview_into_reply(monkeypatch):
    """gui_bridge_connect returns {note, overview} so attaching alone gives the
    picture. (gui_connect was renamed to gui_bridge_connect; the handler is the
    same tool_gui_connect.)"""
    from zcu_tools.mcp.measure import server as mcp_server

    ports: list[int] = []
    monkeypatch.setattr(mcp_server, "resolve_connect_port", lambda cfg, req: 9911)
    monkeypatch.setattr(
        mcp_server._BRIDGE,
        "connect",
        lambda port, token=None: ports.append(port) or "ok",
    )
    monkeypatch.setattr(mcp_server, "_assemble_overview", lambda: {"sentinel": True})
    initialize_events = MagicMock()
    monkeypatch.setattr(
        mcp_server._SESSION, "initialize_event_stream", initialize_events
    )

    out = mcp_server.TOOLS["gui_bridge_connect"]["handler"]({})
    assert out == {"note": "ok", "overview": {"sentinel": True}}
    assert ports == [9911]
    initialize_events.assert_called_once_with()
