"""Setup progress, uncertainty and native-only working values through public tools."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure import tools_operation
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import MeasureClient, WireTransport, make_client


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
            "unit": "native",
            "info": {
                "type": "FakeDevice",
                "value": 0.25,
                "rampstep": 0.1,
                "output": "off",
                "address": "",
                "value_unit": "native",
            },
            "fields": [{"name": "value", "settable": True, "type": "float"}],
        }
        self.guide = {
            "markdown": "# Native guide\n\nUse adapter defaults.\n",
            "resource_dir": None,
        }

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        response: dict[str, Any]
        if method == "state.has_soc":
            response = {"value": self.has_soc}
        elif method == "soc.info":
            response = {"is_mock": self.is_mock, "cfg": {"board": "mock"}}
        elif method == "device.list":
            response = {
                "devices": [
                    {"name": self.device["name"], "status": self.device["status"]}
                ]
            }
        elif method == "device.snapshot":
            response = {"snapshot": deepcopy(self.device)}
        elif method == "predictor.info":
            response = {"loaded": self.loaded}
        elif method == "device.setup":
            self.device["info"] = deepcopy(params["cfg"])
            response = {"operation_id": 41}
        elif method == "operation.await":
            response = {"reason": self.wait_reason, "status": self.operation_status}
        elif method == "operation.progress":
            response = {"active": False}
        elif method == "adapter.guide":
            response = {"guide": self.guide}
        else:
            raise AssertionError(f"Unexpected GUI method: {method}")
        return response


@pytest.fixture
def setup_client(
    tmp_path: Path, clients: list[MeasureClient]
) -> tuple[MeasureClient, SetupGui]:
    gui = SetupGui()
    client = make_client(tmp_path, gui)
    clients.append(client)
    return client, gui


@pytest.mark.parametrize("prior_connection", [False, True])
@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (
            "device_setup",
            {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
        ),
        ("recipe_guide", {"recipe": "pipeline"}),
    ],
)
def test_setup_requires_existing_connection_without_attaching_replacement_gui(
    tmp_path, clients, monkeypatch, prior_connection, tool, arguments
):
    gui = SetupGui()
    client = make_client(tmp_path, gui, port_is_open=lambda port: True)
    clients.append(client)
    if prior_connection:
        assert (
            client.call("recipe_guide", {"recipe": "pipeline"})["recipe"] == "pipeline"
        )
    client.context.bridge.disconnect()
    replacement = WireTransport(gui)
    replacement.replies["rpc.catalog"] = deepcopy(
        client.transport.replies["rpc.catalog"]
    )
    connects: list[int] = []

    def connect(port: int, token: str | None = None) -> str:
        connects.append(port)
        client.context.bridge.set_transport(replacement)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    previous_calls = list(client.transport.sent)
    if tool == "recipe_guide":
        with pytest.raises(GuiRpcError):
            client.call(tool, arguments)
    else:
        result = client.call(tool, arguments)
        assert result.is_error
        assert result.data["op"] is None
        assert result.data["steps"]["start"]["status"] == "not_started"
    assert connects == []
    assert replacement.sent == []
    assert client.transport.sent == previous_calls


@pytest.mark.parametrize(
    ("tool", "arguments", "mutation"),
    [
        (
            "device_setup",
            {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
            "device.setup",
        ),
    ],
)
def test_internal_progress_timeout_retains_setup_handle_and_before_facts(
    setup_client, tool, arguments, mutation
):
    client, gui = setup_client
    gui.wait_reason = "timeout"

    def progress(params):
        raise GuiTransportTimeoutError("operation.progress", 0.01)

    client.transport.replies["operation.progress"] = progress
    result = client.call(
        tool,
        arguments,
    )
    assert result.is_error
    assert result.data["op"] is not None
    assert result.data["status"] == "unknown"
    assert result.data["operation"] is None
    assert result.data["steps"]["pre_read"]["status"] == "completed"
    assert result.data["steps"]["start"]["status"] == "completed"
    assert result.data["steps"]["wait"]["status"] == "unknown"
    assert result.data["error"]["code"] == "timeout"
    assert result.data["error"]["reason"] == "gui_transport_timeout"
    if tool == "device_setup":
        assert result.data["before"]["device"]["info"]["value"] == 0.25
    else:
        assert result.data["before"]["soc"] is None
    assert [method for method, _ in client.transport.sent].count(mutation) == 1


@pytest.mark.parametrize(
    ("tool", "arguments", "mutation", "post_read"),
    [
        (
            "device_setup",
            {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
            "device.setup",
            "device.snapshot",
        ),
    ],
)
def test_running_operation_with_post_read_failure_is_an_error_reply(
    setup_client, tool, arguments, mutation, post_read
):
    client, gui = setup_client
    gui.wait_reason = "timeout"
    reads = 0

    def read(params):
        nonlocal reads
        reads += 1
        if reads == 1:
            return {"ok": True, "result": gui(post_read, params)}
        return {
            "ok": False,
            "error": {
                "code": "internal_error",
                "reason": "query_failed",
                "message": "post-read failed while operation is running",
            },
        }

    client.transport.replies[post_read] = read
    result = client.call(
        tool,
        arguments,
    )
    assert result.is_error
    assert result.data["op"] is not None
    assert result.data["operation"]["status"] == "running"
    assert result.data["steps"]["wait"]["status"] == "running"
    assert result.data["steps"]["post_read"]["status"] == "failed"
    assert result.data["error"]["reason"] == "query_failed"
    assert [method for method, _ in client.transport.sent].count(mutation) == 1


def test_set_value_preserves_native_value_and_unrelated_controls(setup_client):
    client, gui = setup_client
    result = client.call(
        "device_setup",
        {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
    )
    assert result.data["status"] == "finished"
    assert not result.is_error
    assert result.data["before"]["device"]["info"]["value"] == 0.25
    assert result.data["after"]["device"]["info"] == gui.device["info"]
    assert result.data["after"]["device"]["info"]["value"] == 1
    assert result.data["requested"] == {
        "name": "fake_flux",
        "cfg": {**result.data["before"]["device"]["info"], "value": 1},
    }
    assert (
        "device.setup",
        {"name": "fake_flux", "cfg": result.data["requested"]["cfg"]},
    ) in client.transport.sent
    assert gui.device["info"]["output"] == "off"
    assert gui.device["info"]["rampstep"] == 0.1
    client.call(
        "device_setup",
        {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
    )
    assert [method for method, _ in client.transport.sent].count("device.setup") == 2


def test_physical_device_uses_exact_native_unit_without_conversion(setup_client):
    client, gui = setup_client
    gui.device.update(type_name="YOKOGS200", unit="V")
    gui.device["info"].update(
        type="YOKOGS200", mode="voltage", value_unit="V", rampstep=1e-3
    )
    result = client.call(
        "device_setup", {"name": "fake_flux", "cfg": {"value": 1.5, "value_unit": "V"}}
    )
    assert result.data["status"] == "finished"
    assert result.data["after"]["device"]["info"]["value"] == 1.5
    assert result.data["after"]["device"]["unit"] == "V"


@pytest.mark.parametrize("defect", ["disconnected", "info-null"])
def test_device_precondition_failure_retains_before_without_mutation(
    setup_client, defect
):
    client, gui = setup_client
    if defect == "disconnected":
        gui.device["status"] = "memory_only"
        client.transport.replies["device.setup"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "request_rejected",
                "message": "not connected",
            },
        }
    else:
        gui.device["info"] = None
    result = client.call(
        "device_setup",
        {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
    )
    assert result.is_error
    assert result.data["before"]["device"] == gui.device
    assert result.data["after"]["device"] is None
    assert result.data["op"] is None
    if defect == "disconnected":
        assert result.data["steps"]["start"]["status"] == "failed"
        assert result.data["error"]["code"] == "precondition_failed"
        assert result.data["error"]["request_rejected"] is True
        assert result.data["requested"]["cfg"] == {**gui.device["info"], "value": 1}
        assert [method for method, _ in client.transport.sent].count(
            "device.setup"
        ) == 1
    else:
        assert result.data["steps"]["pre_read"]["status"] == "failed"
        assert result.data["requested"]["cfg"] is None
        assert not any(method == "device.setup" for method, _ in client.transport.sent)


@pytest.mark.parametrize(
    "tool,method,arguments",
    [
        (
            "device_setup",
            "device.setup",
            {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
        ),
    ],
)
@pytest.mark.parametrize(
    "failure", ["not_holder", "transport_timeout", "handler_timeout"]
)
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
                "code": "precondition_failed" if failure == "not_holder" else "timeout",
                "reason": "not_holder" if failure == "not_holder" else None,
                "message": "wire failure",
            },
        }

    client.transport.replies[method] = rejected
    result = client.call(tool, arguments)
    assert result.is_error
    assert result.data["op"] is None
    assert result.data["steps"]["start"]["status"] == (
        "failed" if failure == "not_holder" else "unknown"
    )
    assert result.data["steps"]["wait"]["status"] == "not_started"
    assert [sent for sent, _ in client.transport.sent].count(method) == 1
    if failure == "not_holder":
        assert result.data["error"]["reason"] == "not_holder"
        assert result.data["steps"]["post_read"]["status"] == "not_started"
    else:
        assert result.data["status"] == "unknown"


@pytest.mark.parametrize(
    "tool,arguments",
    [
        (
            "device_setup",
            {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
        ),
    ],
)
def test_wait_timeout_keeps_operation_running_and_reads_partial_facts(
    setup_client, tool, arguments
):
    client, gui = setup_client
    gui.wait_reason = "timeout"
    result = client.call(
        tool,
        arguments,
    )
    assert result.data["status"] == "running"
    assert result.data["operation"]["status"] == "running"
    assert result.data["steps"]["wait"]["status"] == "running"
    assert result.data["steps"]["post_read"]["status"] == "completed"
    assert result.data["op"] is not None
    gui.wait_reason = "completed"
    assert (
        client.call("wait", {"op": result.data["op"], "timeout": 0})["status"]
        == "finished"
    )


@pytest.mark.parametrize(
    "tool,arguments",
    [
        (
            "device_setup",
            {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
        ),
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
        (
            "device_setup",
            "device.snapshot",
            {"name": "unknown", "cfg": {"value": 1, "value_unit": "native"}},
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
        (
            "device_setup",
            {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
        ),
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
        "device_setup",
        {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
    )
    assert result.is_error
    assert result.data["status"] == "unknown"
    assert result.data["op"] is not None
    assert result.data["steps"]["start"]["status"] == "completed"
    assert result.data["steps"]["wait"]["status"] == "unknown"
    assert [method for method, _ in client.transport.sent].count("device.setup") == 1
    assert "operation.await" not in [method for method, _ in client.transport.sent]


def test_value_quantization_is_observed_not_invented_as_a_failure(setup_client):
    client, gui = setup_client

    def quantized(params):
        gui("device.setup", params)
        gui.device["info"]["value"] = 1.01
        return {"ok": True, "result": {"operation_id": 42}}

    client.transport.replies["device.setup"] = quantized
    result = client.call(
        "device_setup",
        {"name": "fake_flux", "cfg": {"value": 1, "value_unit": "native"}},
    )
    assert not result.is_error
    assert result.data["status"] == "finished"
    assert result.data["after"]["device"]["info"]["value"] == 1.01


@pytest.mark.parametrize(
    "recipe,adapter",
    [
        ("pipeline", "lookback"),
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
        ("precondition_failed", "source_unavailable"),
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
        client.call("recipe_guide", {"recipe": "pipeline"})
    assert getattr(error.value, "code", None) == code
    assert getattr(error.value, "reason", None) == (
        "gui_handler_timeout" if code == "timeout" else reason
    )


class CompleteSetupGui(SetupGui):
    """Native full-cfg collaborator; the tool owns merge and wait orchestration."""

    def __init__(self) -> None:
        super().__init__()
        self.device["unit"] = "native"
        self.device["info"].update(address="", output="off", value_unit="native")

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "device.setup":
            self.device["info"] = deepcopy(params["cfg"])
            return {"operation_id": 41}
        return super().__call__(method, params)


@pytest.fixture
def complete_setup_client(
    tmp_path: Path, clients: list[MeasureClient]
) -> tuple[MeasureClient, CompleteSetupGui]:
    gui = CompleteSetupGui()
    client = make_client(tmp_path, gui)
    clients.append(client)
    return client, gui


def set_yoko(gui: CompleteSetupGui, mode: str) -> None:
    gui.device.update(type_name="YOKOGS200", unit="A" if mode == "current" else "V")
    gui.device["info"] = {
        "type": "YOKOGS200",
        "address": "",
        "mode": mode,
        "value_unit": gui.device["unit"],
        "value": 0.001 if mode == "current" else 0.25,
        "output": "off",
        "rampstep": 1e-6 if mode == "current" else 1e-3,
    }


def setup_calls(client: MeasureClient) -> list[dict[str, Any]]:
    return [
        params for method, params in client.transport.sent if method == "device.setup"
    ]


@pytest.mark.parametrize("mode", ["current", "voltage"])
def test_device_setup_yoko_value_requires_patch_unit_before_rpc(
    complete_setup_client, mode
):
    client, gui = complete_setup_client
    set_yoko(gui, mode)
    before = deepcopy(gui.device)
    result = client.call("device_setup", {"name": "fake_flux", "cfg": {"value": 0.001}})
    assert result.is_error
    assert "value_unit" in result.data["error"]["message"]
    assert result.data["before"]["device"] == before
    assert result.data["requested"] == {"name": "fake_flux", "cfg": None}
    assert result.data["op"] is None
    assert setup_calls(client) == []


@pytest.mark.parametrize(
    "kind,value,unit",
    [("current", 0.001, "A"), ("voltage", 1.5, "V"), ("fake", 1, "native")],
    ids=["current-A", "voltage-V", "fake-native"],
)
def test_device_setup_sends_complete_cfg_with_explicit_value_unit(
    complete_setup_client, kind, value, unit
):
    client, gui = complete_setup_client
    if kind != "fake":
        set_yoko(gui, kind)
    before = deepcopy(gui.device)
    candidate = {**before["info"], "value": value, "value_unit": unit}
    result = client.call(
        "device_setup",
        {"name": "fake_flux", "cfg": {"value": value, "value_unit": unit}},
    )
    assert not result.is_error
    assert result.data["status"] == "finished"
    assert setup_calls(client) == [{"name": "fake_flux", "cfg": candidate}]
    assert result.data["requested"] == {"name": "fake_flux", "cfg": candidate}
    assert result.data["before"]["device"] == before
    assert result.data["after"]["device"]["info"] == candidate
    gui.device["info"]["value"] = 7
    assert result.data["before"]["device"] == before
    assert result.data["after"]["device"]["info"] == candidate


@pytest.mark.parametrize(
    "patch", [{"output": "on"}, {"rampstep": 1e-6}], ids=["output", "rampstep"]
)
def test_device_setup_nonvalue_patch_needs_no_unit(complete_setup_client, patch):
    client, gui = complete_setup_client
    set_yoko(gui, "current")
    candidate = {**gui.device["info"], **patch}
    result = client.call("device_setup", {"name": "fake_flux", "cfg": patch})
    assert not result.is_error
    assert setup_calls(client) == [{"name": "fake_flux", "cfg": candidate}]


@pytest.mark.parametrize(
    "kind,patch",
    [
        ("current", {"value": 0.001, "value_unit": "V"}),
        ("fake", {"unknown": 2}),
        ("fake", {"value_unit": "A"}),
        ("rf", {"freq_Hz": "invalid"}),
    ],
    ids=[
        "mode-unit-conflict",
        "unknown-field",
        "fake-invalid-unit",
        "rf-invalid-frequency",
    ],
)
def test_device_setup_retains_native_validation_rejection(
    complete_setup_client, kind, patch
):
    client, gui = complete_setup_client
    if kind == "current":
        set_yoko(gui, kind)
    elif kind == "rf":
        gui.device.update(type_name="RohdeSchwarzSGS100A")
        gui.device["info"] = {
            "type": "RohdeSchwarzSGS100A",
            "address": "",
            "freq_Hz": 1e9,
            "power_dBm": -120,
            "output": "off",
            "IQ": "off",
        }
    before = deepcopy(gui.device)
    native = {
        "code": "invalid_params",
        "reason": "model_rejected",
        "message": "native validator diagnostic",
    }
    client.transport.replies["device.setup"] = {"ok": False, "error": native}
    result = client.call("device_setup", {"name": "fake_flux", "cfg": patch})
    assert result.is_error
    assert result.data["error"] == {
        **native,
        "message": "GUI Error (invalid_params): native validator diagnostic",
        "request_rejected": True,
    }
    assert result.data["steps"]["start"]["status"] == "failed"
    assert result.data["before"]["device"] == before
    assert result.data["after"]["device"] is None
    assert result.data["op"] is None
    assert setup_calls(client) == [
        {"name": "fake_flux", "cfg": {**before["info"], **patch}}
    ]


@pytest.mark.parametrize("kind", ["rf-valid"])
def test_device_setup_nonflux_cfg_uses_native_rpc_contract(complete_setup_client, kind):
    client, gui = complete_setup_client
    gui.device.update(type_name="RohdeSchwarzSGS100A")
    gui.device["info"] = {
        "type": "RohdeSchwarzSGS100A",
        "address": "",
        "freq_Hz": 1e9,
        "power_dBm": -120,
        "output": "off",
        "IQ": "off",
    }
    candidate = {**gui.device["info"], "freq_Hz": 5e9, "power_dBm": 10}
    result = client.call(
        "device_setup", {"name": "fake_flux", "cfg": {"freq_Hz": 5e9, "power_dBm": 10}}
    )
    assert not result.is_error
    assert result.data["after"]["device"]["info"] == candidate
    assert setup_calls(client) == [{"name": "fake_flux", "cfg": candidate}]


@pytest.mark.parametrize("outcome", ["finished", "failed", "cancelled"])
def test_device_setup_reports_native_terminal_outcome_and_snapshots(
    complete_setup_client, outcome
):
    client, gui = complete_setup_client
    before = deepcopy(gui.device)
    native = {"reason": "completed", "status": outcome, "feedback": {"native": [1, 2]}}
    if outcome != "finished":
        native["error"] = {"reason": "driver_outcome", "message": "partial ramp"}
    client.transport.replies["operation.await"] = {"ok": True, "result": native}
    result = client.call("device_setup", {"name": "fake_flux", "cfg": {"value": 1}})
    assert result.data["status"] == outcome
    assert result.is_error == (outcome != "finished")
    assert result.data["operation"]["status"] == outcome
    assert result.data["operation"]["feedback"] == native["feedback"]
    if outcome != "finished":
        assert result.data["operation"]["error"] == native["error"]
    assert result.data["before"]["device"] == before
    assert result.data["after"]["device"]["info"]["value"] == 1
    assert result.data["steps"]["wait"]["status"] == "completed"


@pytest.mark.parametrize("bound", [300], ids=["bounded-300"])
def test_device_setup_timeout_retains_handle_without_cancellation(
    complete_setup_client, bound, monkeypatch
):
    client, gui = complete_setup_client
    # Control elapsed time without changing the real wait/handle/cancel logic.
    monkeypatch.setattr(
        tools_operation, "time", SimpleNamespace(monotonic=lambda: 100.0)
    )
    gui.wait_reason = "timeout"
    result = client.call("device_setup", {"name": "fake_flux", "cfg": {"value": 1}})
    assert not result.is_error
    assert result.data["status"] == "running"
    assert result.data["op"] is not None
    assert result.data["steps"]["wait"]["status"] == "running"
    waits = [
        params
        for method, params in client.transport.sent
        if method == "operation.await"
    ]
    assert len(waits) == 1 and waits[0]["timeout"] == bound
    assert result.data["operation"]["elapsed_s"] == 0
    gui.wait_reason = "completed"
    assert (
        client.call("wait", {"op": result.data["op"], "timeout": 0})["status"]
        == "finished"
    )
    assert [
        method for method, _ in client.transport.sent if method.startswith("operation.")
    ] == ["operation.await", "operation.progress", "operation.await"]
