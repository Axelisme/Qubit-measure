"""Read-only simulator source and error contracts through RPC and MCP.

Real context captures supply concrete qubits. Independent inputs never require
an installed model, a point, hardware, a dialog or a dialog cache.
"""

from copy import deepcopy
from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.resources.entry import component_registry

import zcu_lab.components as components
from tests.gui._predictor_fakes import FailingPredictorService

from ._helpers import call, mcp_client, open_client


@pytest.mark.parametrize("method", ["predictor.predict", "predictor.calibrate"])
@pytest.mark.parametrize(
    "failure",
    [
        ValueError("solver invariant"),
        ValueError("Spectrum must be finite"),
        RuntimeError("solver failed"),
    ],
)
def test_numerical_failures_are_unexpected_with_traceback(
    simulator, caplog, method, failure
):
    _, adapter, sock = simulator
    service = FailingPredictorService(failure)
    adapter.predictor_control = PredictorControlFacet(service)
    with caplog.at_level("ERROR"):
        reply = call(
            sock,
            method,
            {"params": COMPLETE, "device_value": 0.1, "frequency_mhz": 1707.0},
        )
    assert not reply["ok"]
    assert reply["error"]["code"] == "controller_error", reply
    assert len(service.requests) == 1
    assert any(
        record.exc_info is not None and record.exc_info[1] is failure
        for record in caplog.records
    )


COMPLETE = {
    "model": "fluxonium",
    "EJ": 5.0,
    "EC": 1.0,
    "EL": 0.5,
    "flux_half": 0.0,
    "flux_period": 1.0,
    "flux_bias": 0.0,
}
MISSING = ["EC", "EL", "flux_half", "flux_period", "flux_bias"]


@pytest.fixture
def simulator(qapp, tmp_path: Path):
    before_kinds = component_registry.kinds()
    added = []
    for kind, schema in (
        ("qubit/fluxonium", components.QubitSchema),
        ("qubit/transmon", components.TransmonSchema),
    ):
        if kind not in before_kinds:
            component_registry.register(kind, schema)
            added.append(kind)
    controller = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        Registry(),
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
        params_for=components.params_for,
    )
    adapter = RemoteControlAdapter(
        controller, ControlOptions(port=0), controller.owner_scheduler
    )
    adapter.start()
    sock = open_client(adapter.port)
    try:
        yield controller, adapter, sock
    finally:
        sock.close()
        adapter.stop()
        assert vars(controller)["_background_svc"].quiesce()
        for kind in added:
            component_registry.unregister(kind)
        assert component_registry.kinds() == before_kinds


def prepare_point(controller: Controller) -> None:
    """Publish two concrete components with different completeness at one point."""
    controller.setup_control.new_entry(EntryRequest("queries"))
    controller.context_control.add_component(
        "Q1",
        "qubit/fluxonium",
        {
            "EJ": 5.0,
            "EC": 1.0,
            "EL": 0.5,
            "flux_half": 0.0,
            "flux_period": 1.0,
            "ext": {"flux_bias": 0.0},
        },
        setup=True,
    )
    controller.context_control.add_component(
        "Q2", "qubit/fluxonium", {"EJ": 4.1}, setup=True
    )
    controller.context_control.add_component(
        "QT",
        "qubit/transmon",
        {
            "EJmax": 20.0,
            "EC": 0.3,
            "d": 0.0,
            "flux_half": 0.0,
            "flux_period": 1.0,
            "ext": {"flux_bias": 0.0},
        },
        setup=True,
    )
    controller.setup_control.new_point(
        PointCreateRequest("first", "", CommandOrigin("gui"))
    )


def test_qubit_queries_and_calibration_preserve_current_point(simulator) -> None:
    controller, _, sock = simulator
    prepare_point(controller)
    before = controller.context_control.read_context().document()
    info = call(sock, "predictor.info", {"qubit": "Q2"})["result"]
    assert info["params"]["EJ"] == 4.1
    assert info["missing"] == MISSING
    freq = call(sock, "predictor.predict", {"qubit": "Q1", "device_value": 0.1})[
        "result"
    ]["freq_mhz"]
    assert freq == pytest.approx(1707.3748654723192, abs=1e-7)
    result = call(
        sock,
        "predictor.calibrate",
        {
            "qubit": "Q1",
            "device_value": 0.101,
            "frequency_mhz": freq,
        },
    )["result"]
    assert result["flux_bias_after"] == pytest.approx(-0.001, abs=1e-4)
    assert (
        call(sock, "predictor.info", {"qubit": "Q1"})["result"]["params"]["flux_bias"]
        == 0.0
    )
    assert controller.context_control.read_context().document() == before


def test_direct_prediction_and_calibration_without_a_point(simulator) -> None:
    _, _, sock = simulator
    params = deepcopy(COMPLETE)
    freq_reply = call(
        sock, "predictor.predict", {"params": params, "device_value": 0.1}
    )
    assert freq_reply["ok"], freq_reply
    assert freq_reply["result"]["freq_mhz"] == pytest.approx(
        1707.3748654723192, abs=1e-7
    )
    result = call(
        sock,
        "predictor.calibrate",
        {
            "params": params,
            "device_value": 0.101,
            "frequency_mhz": freq_reply["result"]["freq_mhz"],
        },
    )
    assert result["ok"], result
    assert result["result"]["flux_bias_before"] == 0.0
    assert result["result"]["flux_bias_after"] == pytest.approx(-0.001, abs=1e-4)
    assert (
        call(sock, "predictor.info", {"params": params})["result"]["params"][
            "flux_bias"
        ]
        == 0.0
    )
    assert params == COMPLETE


def test_info_reports_partial_direct_parameters(simulator) -> None:
    _, _, sock = simulator
    reply = call(sock, "predictor.info", {"params": {"model": "fluxonium", "EJ": 4.1}})
    assert reply["ok"], reply
    assert reply["result"] == {
        "model": "fluxonium",
        "params": {
            "EJ": 4.1,
            "EC": None,
            "EL": None,
            "flux_half": None,
            "flux_period": None,
            "flux_bias": None,
        },
        "missing": MISSING,
    }


@pytest.mark.parametrize(
    "method", ["predictor.info", "predictor.predict", "predictor.calibrate"]
)
@pytest.mark.parametrize("source", [{}, {"qubit": "Q1", "params": COMPLETE}])
def test_sources_are_exclusive_and_required(simulator, method, source) -> None:
    _, _, sock = simulator
    reply = call(sock, method, {**source, "device_value": 0.1, "frequency_mhz": 1707.0})
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "invalid_predictor_source"
    assert (
        "qubit" in reply["error"]["message"] and "params" in reply["error"]["message"]
    )


@pytest.mark.parametrize("method", ["predictor.predict", "predictor.calibrate"])
@pytest.mark.parametrize("source", ["qubit", "params"])
def test_missing_fields_are_named_without_mutation(simulator, method, source) -> None:
    controller, _, sock = simulator
    prepare_point(controller)
    before = controller.context_control.read_context().document()
    inputs = (
        {"qubit": "Q2"}
        if source == "qubit"
        else {"params": {"model": "fluxonium", "EJ": 4.1}}
    )
    reply = call(sock, method, {**inputs, "device_value": 0.1, "frequency_mhz": 1707.0})
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "missing_predictor_params"
    assert reply["error"]["data"]["missing"] == MISSING
    assert all(name in reply["error"]["message"] for name in MISSING)
    assert controller.context_control.read_context().document() == before


@pytest.mark.parametrize("qubit", ["unknown", "Q1.EJ", "qubit", ""])
def test_qubit_requires_a_concrete_component_name(simulator, qubit) -> None:
    controller, _, sock = simulator
    prepare_point(controller)
    reply = call(sock, "predictor.info", {"qubit": qubit})
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "invalid_predictor_qubit"


@pytest.mark.parametrize(
    "params,field",
    [
        ({"EJ": 5.0}, "model"),
        ({"model": "unknown"}, "model"),
        ({**COMPLETE, "EC": "1"}, "EC"),
        ({**COMPLETE, "EL": True}, "EL"),
        ({**COMPLETE, "flux_period": 0.0}, "flux_period"),
        ({**COMPLETE, "EJ": -1.0}, "EJ"),
        ({**COMPLETE, "unexpected": 1.0}, "unexpected"),
    ],
)
def test_invalid_direct_parameters_have_a_named_error(simulator, params, field) -> None:
    _, _, sock = simulator
    reply = call(sock, "predictor.predict", {"params": params, "device_value": 0.1})
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "invalid_predictor_params"
    assert field in reply["error"]["message"]


def test_queries_use_selected_point_not_setup_or_another_qubit(simulator) -> None:
    controller, _, sock = simulator
    prepare_point(controller)
    first = controller.context_control.read_context()
    controller.context_control.set_parameter(first.target, "Q1.ext.flux_bias", 0.02)
    controller.setup_control.new_point(
        PointCreateRequest("second", "", CommandOrigin("gui"))
    )
    controller.context_control.set_setup_parameter("Q1.ext.flux_bias", 0.04)
    assert (
        call(sock, "predictor.info", {"qubit": "Q1"})["result"]["params"]["flux_bias"]
        == 0.0
    )
    controller.setup_control.use_point(PointUseRequest("first"))
    before = controller.context_control.read_context().document()
    assert (
        call(sock, "predictor.info", {"qubit": "Q1"})["result"]["params"]["flux_bias"]
        == 0.02
    )
    actual = call(sock, "predictor.predict", {"qubit": "Q1", "device_value": 0.1})[
        "result"
    ]
    independent = call(
        sock,
        "predictor.predict",
        {"params": {**COMPLETE, "flux_bias": 0.02}, "device_value": 0.1},
    )
    assert independent["ok"], independent
    assert independent["result"] == actual
    assert controller.context_control.read_context().document() == before


def test_mcp_rpc_preserves_read_only_sources_and_named_errors(
    simulator, tmp_path: Path, request
) -> None:
    controller, adapter, _ = simulator
    prepare_point(controller)
    before = controller.context_control.read_context().document()
    bridge, invoke = mcp_client(adapter.port, tmp_path, request=request)
    try:
        invoke("connect", {"port": adapter.port})
        direct = invoke(
            "rpc_call",
            {
                "method": "predictor.predict",
                "params": {"params": COMPLETE, "device_value": 0.1},
            },
        )
        assert direct["freq_mhz"] == pytest.approx(1707.3748654723192, abs=1e-7)
        assert (
            invoke(
                "rpc_call",
                {
                    "method": "predictor.predict",
                    "params": {"qubit": "Q1", "device_value": 0.1},
                },
            )
            == direct
        )
        calibrated = invoke(
            "rpc_call",
            {
                "method": "predictor.calibrate",
                "params": {
                    "params": COMPLETE,
                    "device_value": 0.101,
                    "frequency_mhz": direct["freq_mhz"],
                },
            },
        )
        assert calibrated["flux_bias_after"] == pytest.approx(-0.001, abs=1e-4)
        assert (
            invoke("rpc_call", {"method": "predictor.info", "params": {"qubit": "Q2"}})[
                "missing"
            ]
            == MISSING
        )
        with pytest.raises(GuiRpcError) as rejected:
            invoke("rpc_call", {"method": "predictor.info", "params": {}})
        assert rejected.value.reason == "invalid_predictor_source"
        assert controller.context_control.read_context().document() == before
    finally:
        bridge.disconnect()


def test_transmon_direct_info_and_missing_prediction(simulator) -> None:
    _, _, sock = simulator
    direct = {"model": "transmon", "EJmax": 20.0, "d": 0.0}
    reply = call(sock, "predictor.info", {"params": direct})
    assert reply["ok"], reply
    assert reply["result"] == {
        "model": "transmon",
        "params": {
            "EJmax": 20.0,
            "EC": None,
            "d": 0.0,
            "flux_half": None,
            "flux_period": None,
            "flux_bias": None,
        },
        "missing": ["EC", "flux_half", "flux_period", "flux_bias"],
    }
    failed = call(sock, "predictor.predict", {"params": direct, "device_value": 0.0})
    assert failed["error"]["reason"] == "missing_predictor_params"
    assert failed["error"]["data"]["missing"] == reply["result"]["missing"]
    invalid = call(sock, "predictor.info", {"params": {**direct, "EL": 1.0}})
    assert invalid["error"]["reason"] == "invalid_predictor_params"


@pytest.mark.parametrize(
    "model, qubit, reachable",
    [("fluxonium", "Q1", 1707.3748654723192), ("transmon", "QT", 1707.3748654723192)],
)
@pytest.mark.parametrize("source", ["qubit", "params"])
@pytest.mark.parametrize("target_kind", ["reachable", "below", "above"])
def test_mcp_calibration_reports_actual_frequency_and_clamping_read_only(
    simulator, tmp_path, request, model, qubit, reachable, source, target_kind
):
    controller, adapter, _ = simulator
    prepare_point(controller)
    before = controller.context_control.read_context().document()
    direct = (
        COMPLETE
        if model == "fluxonium"
        else {
            "model": "transmon",
            "EJmax": 20.0,
            "EC": 0.3,
            "d": 0.0,
            "flux_half": 0.0,
            "flux_period": 1.0,
            "flux_bias": 0.0,
        }
    )
    selection = {"qubit": qubit} if source == "qubit" else {"params": deepcopy(direct)}
    target = (
        reachable
        if target_kind == "reachable"
        else (-100.0 if target_kind == "below" else 20000.0)
    )
    bridge, invoke = mcp_client(adapter.port, tmp_path, request=request)
    try:
        invoke("connect", {"port": adapter.port})
        result = invoke(
            "rpc_call",
            {
                "method": "predictor.calibrate",
                "params": {**selection, "device_value": 0.101, "frequency_mhz": target},
            },
        )
        assert "clamped" in result, "MCP must project model reachability"
        assert result["clamped"] is (target_kind != "reachable")
        assert result["flux_bias_before"] == 0.0
        actual = invoke(
            "rpc_call",
            {
                "method": "predictor.predict",
                "params": {
                    "params": {**direct, "flux_bias": result["flux_bias_after"]},
                    "device_value": 0.101,
                },
            },
        )
        assert actual["freq_mhz"] == pytest.approx(result["frequency_mhz"], abs=1e-6)
        if target_kind == "reachable":
            assert result["frequency_mhz"] == pytest.approx(target, abs=1e-6)
        assert controller.context_control.read_context().document() == before
        assert selection == (
            {"qubit": qubit} if source == "qubit" else {"params": direct}
        )
    finally:
        bridge.disconnect()
