"""Setup progress, uncertainty and native-only working values through public tools."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.core.reply import ToolReply

from tests.gui.app.measure.remote._helpers import Fixture, mcp_client

from ._support import MeasureClient, make_client


class SetupGui:
    """Recording native collaborator; no setup tool orchestration lives here."""

    def __init__(self) -> None:
        self.has_soc = False
        self.is_mock = False
        self.loaded = False
        self.operation_status = "finished"
        self.wait_reason = "completed"
        self.device: dict[str, Any] = {
            "name": "fake_flux",
            "type_name": "FakeDevice",
            "address": "",
            "status": "connected",
            "error": None,
            "unit": "none",
            "info": {
                "type": "FakeDevice",
                "value": 0.25,
                "rampstep": 0.1,
                "output": False,
            },
            "fields": [{"name": "value", "settable": True, "type": "float"}],
        }
        self.guide = {"behavior": "native guide", "recommended": "use adapter defaults"}

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "state.has_soc":
            return {"value": self.has_soc}
        if method == "soc.info":
            return {"is_mock": self.is_mock, "cfg": {"board": "mock"}}
        if method == "device.list":
            return {
                "devices": [
                    {"name": self.device["name"], "status": self.device["status"]}
                ]
            }
        if method == "device.snapshot":
            return {"snapshot": deepcopy(self.device)}
        if method == "predictor.info":
            return {"loaded": self.loaded}
        if method == "simulation.initialize":
            self.has_soc = self.is_mock = self.loaded = True
            return {"operation_id": 40}
        if method == "device.setup":
            self.device["info"].update(params["updates"])
            return {"operation_id": 41}
        if method == "operation.await":
            return {"reason": self.wait_reason, "status": self.operation_status}
        if method == "operation.progress":
            return {"active": False}
        if method == "adapter.guide":
            return {"guide": self.guide}
        raise AssertionError(f"Unexpected GUI method: {method}")


@pytest.fixture
def setup_client(
    tmp_path: Path, clients: list[MeasureClient]
) -> tuple[MeasureClient, SetupGui]:
    gui = SetupGui()
    client = make_client(tmp_path, gui)
    clients.append(client)
    return client, gui


def test_simulation_verifies_native_readiness_and_repeats_explicit_initialization(
    setup_client,
):
    client, gui = setup_client
    for _ in range(2):
        result = client.call("simulation_initialize", {"timeout": 0})
        assert isinstance(result, ToolReply)
        assert not result.is_error
        assert result.data["status"] == "finished"
        assert result.data["operation"]["status"] == "finished"
        assert result.data["verification"]["ready"] is True
        assert result.data["after"]["soc"]["cfg"] == {"board": "mock"}
        assert all(
            step["status"] == "completed" for step in result.data["steps"].values()
        )
    assert gui.device["info"]["value"] == 0.25
    assert [method for method, _ in client.transport.sent].count(
        "simulation.initialize"
    ) == 2
    assert ("soc.info", {"include_cfg": True}) in client.transport.sent


def test_set_value_preserves_native_value_and_unrelated_controls(setup_client):
    client, gui = setup_client
    result = client.call(
        "device_set_value", {"name": "fake_flux", "value": 1, "unit": "native"}
    )
    assert result.data["status"] == "finished"
    assert not result.is_error
    assert result.data["before"]["device"]["info"]["value"] == 0.25
    assert result.data["after"]["device"]["info"] == gui.device["info"]
    assert result.data["verification"]["actual"] == 1
    assert result.data["verification"]["exact_match"] is True
    assert result.data["requested"] == {
        "name": "fake_flux",
        "value": 1,
        "unit": "native",
    }
    assert (
        "device.setup",
        {"name": "fake_flux", "updates": {"value": 1}},
    ) in client.transport.sent
    assert gui.device["info"]["output"] is False
    assert gui.device["info"]["rampstep"] == 0.1
    client.call("device_set_value", {"name": "fake_flux", "value": 1, "unit": "native"})
    assert [method for method, _ in client.transport.sent].count("device.setup") == 2


@pytest.mark.parametrize("unit", ["A", "V"])
def test_physical_device_uses_exact_native_unit_without_conversion(setup_client, unit):
    client, gui = setup_client
    gui.device.update(type_name="YOKOGS200", unit=unit)
    result = client.call(
        "device_set_value", {"name": "fake_flux", "value": 1.5, "unit": unit}
    )
    assert result.data["status"] == "finished"
    assert result.data["verification"]["actual"] == 1.5
    assert result.data["after"]["device"]["unit"] == unit


@pytest.mark.parametrize("invalid", [True, "1", float("inf"), float("nan")])
def test_bad_value_is_rejected_before_gui_access(setup_client, invalid):
    client, _ = setup_client
    with pytest.raises(ValueError):
        client.call(
            "device_set_value",
            {"name": "fake_flux", "value": invalid, "unit": "native"},
        )
    assert client.transport.sent == []


@pytest.mark.parametrize("tool", ["simulation_initialize", "device_set_value"])
@pytest.mark.parametrize("invalid", [True, "0", -1, 301, float("inf"), float("nan")])
def test_bad_timeout_is_rejected_before_gui_access(setup_client, tool, invalid):
    client, _ = setup_client
    arguments = (
        {}
        if tool == "simulation_initialize"
        else {"name": "fake_flux", "value": 1, "unit": "native"}
    )
    with pytest.raises(ValueError):
        client.call(tool, {**arguments, "timeout": invalid})
    assert client.transport.sent == []


@pytest.mark.parametrize(
    "defect", ["disconnected", "wrong_unit", "no_value", "non_fake_native"]
)
def test_device_precondition_failure_retains_before_without_mutation(
    setup_client, defect
):
    client, gui = setup_client
    if defect == "disconnected":
        gui.device["status"] = "disconnected"
    elif defect == "no_value":
        gui.device["fields"] = []
    elif defect == "non_fake_native":
        gui.device["type_name"] = "YOKOGS200"
    unit = "A" if defect == "wrong_unit" else "native"
    result = client.call(
        "device_set_value", {"name": "fake_flux", "value": 1, "unit": unit}
    )
    assert result.is_error
    assert result.data["before"]["device"] == gui.device
    assert result.data["after"]["device"] is None
    assert result.data["steps"]["start"]["status"] == "not_started"
    assert "device.setup" not in [method for method, _ in client.transport.sent]


@pytest.mark.parametrize(
    "tool,method,arguments",
    [
        ("simulation_initialize", "simulation.initialize", {}),
        (
            "device_set_value",
            "device.setup",
            {"name": "fake_flux", "value": 1, "unit": "native"},
        ),
    ],
)
@pytest.mark.parametrize("failure", ["stale", "transport_timeout", "handler_timeout"])
def test_start_failure_distinguishes_rejection_from_unknown_without_retry(
    setup_client, tool, method, arguments, failure
):
    client, gui = setup_client

    def rejected(params):
        if failure == "transport_timeout":
            gui(method, params)
            raise GuiTransportTimeoutError(method, 0.01)
        return {
            "ok": False,
            "error": {
                "code": "precondition_failed" if failure == "stale" else "timeout",
                "reason": "stale_version" if failure == "stale" else None,
                "message": "wire failure",
            },
        }

    client.transport.replies[method] = rejected
    result = client.call(tool, arguments)
    assert result.is_error
    assert result.data["op"] is None
    assert result.data["steps"]["start"]["status"] == (
        "failed" if failure == "stale" else "unknown"
    )
    assert result.data["steps"]["wait"]["status"] == "not_started"
    assert [sent for sent, _ in client.transport.sent].count(method) == 1
    if failure == "stale":
        assert result.data["error"]["reason"] == "stale_version"
        assert result.data["steps"]["post_read"]["status"] == "not_started"
    else:
        assert result.data["status"] == "unknown"


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("simulation_initialize", {}),
        ("device_set_value", {"name": "fake_flux", "value": 1, "unit": "native"}),
    ],
)
def test_wait_timeout_keeps_operation_running_and_reads_partial_facts(
    setup_client, tool, arguments
):
    client, gui = setup_client
    gui.wait_reason = "timeout"
    result = client.call(tool, {**arguments, "timeout": 0})
    assert result.data["status"] == "running"
    assert result.data["operation"]["status"] == "running"
    assert result.data["steps"]["wait"]["status"] == "running"
    assert result.data["steps"]["post_read"]["status"] == "completed"
    assert result.data["op"] is not None
    if tool == "simulation_initialize":
        assert result.data["verification"]["ready"] is False
    gui.wait_reason = "completed"
    assert (
        client.call("wait", {"op": result.data["op"], "timeout": 0})["status"]
        == "finished"
    )


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("simulation_initialize", {}),
        ("device_set_value", {"name": "fake_flux", "value": 1, "unit": "native"}),
    ],
)
def test_native_failed_operation_is_not_hidden_by_completed_reads(
    setup_client, tool, arguments
):
    client, gui = setup_client
    gui.operation_status = "failed"
    client.transport.replies["operation.await"] = {
        "ok": True,
        "result": {
            "reason": "completed",
            "status": "failed",
            "error": {"message": "partial ramp", "reason": "failed"},
        },
    }
    result = client.call(tool, arguments)
    assert result.is_error
    assert result.data["status"] == "failed"
    assert result.data["operation"]["error"]["message"] == "partial ramp"
    assert result.data["steps"]["wait"]["status"] == "completed"
    assert result.data["steps"]["post_read"]["status"] == "completed"
    assert result.data["after"] != result.data["before"]


@pytest.mark.parametrize(
    "tool,pre_read,arguments",
    [
        ("simulation_initialize", "state.has_soc", {}),
        (
            "device_set_value",
            "device.snapshot",
            {"name": "unknown", "value": 1, "unit": "native"},
        ),
    ],
)
def test_missing_precondition_stops_before_start_and_retains_null_facts(
    setup_client, tool, pre_read, arguments
):
    client, _ = setup_client
    client.transport.replies[pre_read] = {
        "ok": False,
        "error": {"code": "precondition_failed", "message": "missing setup resource"},
    }
    result = client.call(tool, arguments)
    assert result.is_error
    assert result.data["steps"]["pre_read"]["status"] == "failed"
    assert result.data["steps"]["start"]["status"] == "not_started"
    assert result.data["op"] is None
    assert all(value is None for value in result.data["after"].values())
    assert not any(
        method in {"simulation.initialize", "device.setup"}
        for method, _ in client.transport.sent
    )


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("simulation_initialize", {}),
        ("device_set_value", {"name": "fake_flux", "value": 1, "unit": "native"}),
    ],
)
def test_wait_query_failure_keeps_handle_and_does_not_infer_completion(
    setup_client, tool, arguments
):
    client, _ = setup_client
    client.transport.replies["operation.await"] = {
        "ok": False,
        "error": {
            "code": "invalid_params",
            "reason": "unknown_op",
            "message": "evicted operation",
        },
    }
    result = client.call(tool, arguments)
    assert result.is_error
    assert result.data["status"] == "unknown"
    assert result.data["op"] is not None
    assert result.data["steps"]["wait"]["status"] == "unknown"
    assert result.data["steps"]["post_read"]["status"] == "completed"
    assert result.data["error"]["reason"] == "unknown_op"
    assert [method for method, _ in client.transport.sent].count("operation.await") == 1


def test_post_read_failure_keeps_native_success_and_already_read_prefix(setup_client):
    client, _ = setup_client
    reads = 0

    def predictor(params):
        nonlocal reads
        reads += 1
        if reads == 1:
            return {"ok": True, "result": {"loaded": False}}
        return {
            "ok": False,
            "error": {
                "code": "internal_error",
                "reason": "query_failed",
                "message": "snapshot failed",
            },
        }

    client.transport.replies["predictor.info"] = predictor
    result = client.call("simulation_initialize", {})
    assert result.is_error
    assert result.data["status"] == "failed"
    assert result.data["operation"]["status"] == "finished"
    assert result.data["after"]["soc"]["is_mock"] is True
    assert result.data["after"]["devices"][0]["name"] == "fake_flux"
    assert result.data["after"]["predictor"] is None
    assert result.data["steps"]["post_read"]["status"] == "failed"


def test_connection_loss_after_receipt_retains_handle_without_reconnecting(
    setup_client,
):
    client, gui = setup_client

    def disconnect_after_start(params):
        response = gui("device.setup", params)
        client.transport.close()
        return {"ok": True, "result": response}

    client.transport.replies["device.setup"] = disconnect_after_start
    result = client.call(
        "device_set_value", {"name": "fake_flux", "value": 1, "unit": "native"}
    )
    assert result.is_error
    assert result.data["status"] == "unknown"
    assert result.data["op"] is not None
    assert result.data["steps"]["start"]["status"] == "completed"
    assert result.data["steps"]["wait"]["status"] == "unknown"
    assert [method for method, _ in client.transport.sent].count("device.setup") == 1
    assert "operation.await" not in [method for method, _ in client.transport.sent]


def test_setup_workflows_use_real_gui_coordinator_and_device_owner(
    qapp, tmp_path, request
):
    fixture = Fixture(empty_project=True, headless=True)
    port = fixture.start()
    _, call = mcp_client(port, tmp_path, request=request)
    try:
        initial = call("simulation_initialize", {})
        assert initial["status"] == "finished", initial
        assert initial["verification"]["ready"] is True
        assert initial["before"]["soc"] is None
        before = initial["after"]["devices"][0]
        value = call(
            "device_set_value", {"name": "fake_flux", "value": 0.5, "unit": "native"}
        )
        assert value["status"] == "finished", value
        assert value["verification"]["actual"] == 0.5
        assert (
            value["after"]["device"]["info"]["rampstep"] == before["info"]["rampstep"]
        )
        repeated = call("simulation_initialize", {})
        assert repeated["verification"]["ready"] is True
        assert repeated["after"]["devices"][0]["info"]["value"] == 0.5
    finally:
        fixture.stop()


def test_native_finished_is_retained_when_mock_verification_fails(setup_client):
    client, _ = setup_client
    client.transport.replies["soc.info"] = {
        "ok": True,
        "result": {"is_mock": False, "cfg": {}},
    }
    result = client.call("simulation_initialize", {})
    assert result.is_error
    assert result.data["status"] == "failed"
    assert result.data["operation"]["status"] == "finished"
    assert result.data["verification"]["ready"] is False
    assert result.data["after"]["soc"]["is_mock"] is False


def test_value_quantization_is_observed_not_invented_as_a_failure(setup_client):
    client, gui = setup_client

    def quantized(params):
        gui("device.setup", params)
        gui.device["info"]["value"] = 1.01
        return {"ok": True, "result": {"operation_id": 42}}

    client.transport.replies["device.setup"] = quantized
    result = client.call(
        "device_set_value", {"name": "fake_flux", "value": 1, "unit": "native"}
    )
    assert not result.is_error
    assert result.data["status"] == "finished"
    assert result.data["verification"]["actual"] == 1.01
    assert result.data["verification"]["exact_match"] is False


@pytest.mark.parametrize(
    "recipe,adapter",
    [
        ("time_rabi", "twotone/rabi/len_rabi"),
        ("amplitude_rabi", "twotone/rabi/amp_rabi"),
    ],
)
def test_recipe_guide_queries_authoritative_adapter_and_returns_native_object(
    setup_client, recipe, adapter
):
    client, gui = setup_client
    assert client.call("recipe_guide", {"recipe": recipe}) == {
        "recipe": recipe,
        "adapter": adapter,
        "guide": gui.guide,
    }
    assert ("adapter.guide", {"adapter_name": adapter}) in client.transport.sent
    assert not any(
        method in {"simulation.initialize", "device.setup", "operation.await"}
        for method, _ in client.transport.sent
    )


def test_unknown_recipe_is_rejected_without_querying_gui(setup_client):
    client, _ = setup_client
    with pytest.raises(ValueError, match="unknown recipe"):
        client.call("recipe_guide", {"recipe": "unknown"})
    assert client.transport.sent == []


@pytest.mark.parametrize(
    "code,reason",
    [
        ("invalid_params", "unknown_adapter"),
        ("precondition_failed", "stale_version"),
        ("timeout", None),
    ],
)
def test_recipe_guide_retains_native_query_errors(setup_client, code, reason):
    client, _ = setup_client
    client.transport.replies["adapter.guide"] = {
        "ok": False,
        "error": {"code": code, "reason": reason, "message": "query failure"},
    }
    with pytest.raises(RuntimeError) as error:
        client.call("recipe_guide", {"recipe": "time_rabi"})
    assert getattr(error.value, "code", None) == code
    assert getattr(error.value, "reason", None) == (
        "gui_handler_timeout" if code == "timeout" else reason
    )
