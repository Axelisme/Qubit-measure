"""Public operation tools use the live GUI, including GUI-origin handles."""

from pathlib import Path
from typing import Any

from ._support import make_client


def test_status_indexes_gui_origin_operations_without_an_agent_start(tmp_path: Path) -> None:
    replies: dict[str, dict[str, Any]] = {
        "state.has_project": {"value": True},
        "project.info": {"chip_name": "chip", "qub_name": "qubit", "res_name": "res"},
        "state.has_context": {"value": True},
        "state.has_active_context": {"value": True},
        "context.active": {"label": "bias"},
        "state.has_soc": {"value": True},
        "soc.info": {"is_mock": True},
        "device.list": {"devices": [{"name": "flux", "status": "connected"}]},
        "predictor.info": {"loaded": False},
        "tab.snapshot": {"tabs": [{"tab_id": "gui-tab", "adapter_name": "ramsey", "interaction": {"is_running": False}}]},
        "operation.active": {"operations": [{"op": 31, "tab": "gui-tab", "kind": "analyze"}, {"op": 32, "tab": None, "kind": "device"}]},
    }
    client = make_client(tmp_path, lambda method, params: replies[method])

    assert client.call("status", {}) == {
        "project": {"chip": "chip", "qubit": "qubit", "resonator": "res"},
        "soc": {"connected": True, "mock": True},
        "context": {"active": "bias"},
        "devices": [{"name": "flux", "connected": True}],
        "predictor": {"loaded": False},
        "ready": {"can_run": True, "missing": []},
        "tabs": [{"tab": "gui-tab", "experiment": "ramsey", "running": False}],
        "running": [{"op": 31, "tab": "gui-tab", "kind": "analyze"}, {"op": 32, "tab": None, "kind": "device"}],
    }
    assert ("operation.active", {}) in client.transport.sent
