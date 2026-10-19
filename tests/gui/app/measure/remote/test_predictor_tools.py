"""Read-only predictor sources and error contracts through public RPC and MCP.

Entry-owned JSON files require no selected point. Inline inputs require no entry,
hardware, dialog, cache or installed model.
"""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import EntryRequest
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui._predictor_fakes import FailingPredictorService

from ._helpers import call, mcp_client, open_client

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
    controller = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        Registry(),
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
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
        controller.close()


def prepare_files(controller: Controller) -> Path:
    """Create authored predictor JSONs in an opened entry without a point."""
    controller.setup_control.new_entry(EntryRequest("queries"))
    entry = controller.context_control.get_entry_info()
    assert entry is not None
    directory = entry.result_path / "predictors"
    directory.mkdir()
    for filename, inputs in (
        ("complete.json", COMPLETE),
        ("partial.json", {"model": "fluxonium", "EJ": 4.1}),
        (
            "transmon.json",
            {
                "model": "transmon",
                "EJmax": 20.0,
                "EC": 0.3,
                "d": 0.0,
                "flux_half": 0.0,
                "flux_period": 1.0,
                "flux_bias": 0.0,
            },
        ),
    ):
        document = {
            "format": "zcu.predictor",
            "version": 1,
            "model": inputs["model"],
            "params": {key: value for key, value in inputs.items() if key != "model"},
            "flux_unit": None,
            "source": {"run_id": None, "created_at": None},
        }
        (directory / filename).write_text(json.dumps(document), encoding="utf-8")
    return directory


def test_filename_queries_and_calibration_preserve_files_without_a_point(
    simulator,
) -> None:
    controller, _, sock = simulator
    directory = prepare_files(controller)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    info = call(sock, "predictor.info", {"filename": "partial.json"})
    assert info["ok"], info
    assert info["result"]["params"]["EJ"] == 4.1
    assert info["result"]["missing"] == MISSING
    prediction = call(
        sock, "predictor.predict", {"filename": "complete.json", "device_value": 0.1}
    )
    assert prediction["ok"], prediction
    freq = prediction["result"]["freq_mhz"]
    assert freq == pytest.approx(1707.3748654723192, abs=1e-7)
    calibration = call(
        sock,
        "predictor.calibrate",
        {"filename": "complete.json", "device_value": 0.101, "frequency_mhz": freq},
    )
    assert calibration["ok"], calibration
    assert calibration["result"]["flux_bias_after"] == pytest.approx(-0.001, abs=1e-4)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    assert controller.context_control.current_point_label() is None


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
@pytest.mark.parametrize(
    "source", [{}, {"filename": "complete.json", "params": COMPLETE}]
)
def test_sources_are_exclusive_and_required(simulator, method, source) -> None:
    _, _, sock = simulator
    reply = call(sock, method, {**source, "device_value": 0.1, "frequency_mhz": 1707.0})
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "invalid_predictor_source"
    assert (
        "filename" in reply["error"]["message"]
        and "params" in reply["error"]["message"]
    )


@pytest.mark.parametrize("method", ["predictor.predict", "predictor.calibrate"])
@pytest.mark.parametrize("source", ["filename", "params"])
def test_missing_fields_are_named_without_mutation(simulator, method, source) -> None:
    controller, _, sock = simulator
    directory = prepare_files(controller)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    inputs = (
        {"filename": "partial.json"}
        if source == "filename"
        else {"params": {"model": "fluxonium", "EJ": 4.1}}
    )
    reply = call(sock, method, {**inputs, "device_value": 0.1, "frequency_mhz": 1707.0})
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "missing_predictor_params"
    assert reply["error"]["data"]["missing"] == MISSING
    assert all(name in reply["error"]["message"] for name in MISSING)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


@pytest.mark.parametrize(
    "filename",
    ["../complete.json", "/complete.json", "complete", "", "nested/complete.json"],
)
def test_filename_requires_an_entry_local_json_name(simulator, filename) -> None:
    controller, _, sock = simulator
    prepare_files(controller)
    reply = call(sock, "predictor.info", {"filename": filename})
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "invalid_predictor_file"


def test_filename_without_an_entry_has_failed_precondition(simulator) -> None:
    _, _, sock = simulator
    reply = call(sock, "predictor.info", {"filename": "complete.json"})
    assert not reply["ok"]
    assert reply["error"]["code"] == "precondition_failed"
    assert reply["error"]["reason"] == "no_entry"


def test_missing_file_has_failed_precondition(simulator) -> None:
    controller, _, sock = simulator
    directory = prepare_files(controller)
    before = set(directory.iterdir())
    reply = call(sock, "predictor.info", {"filename": "missing.json"})
    assert not reply["ok"]
    assert reply["error"]["code"] == "precondition_failed"
    assert reply["error"]["reason"] == "predictor_file_unavailable"
    assert set(directory.iterdir()) == before


@pytest.mark.parametrize("content", ["{broken", '{"format": "wrong"}'])
def test_malformed_file_has_invalid_params_without_rewriting(
    simulator, content
) -> None:
    controller, _, sock = simulator
    directory = prepare_files(controller)
    path = directory / "broken.json"
    path.write_text(content, encoding="utf-8")
    reply = call(sock, "predictor.info", {"filename": path.name})
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "invalid_predictor_file"
    assert path.read_text(encoding="utf-8") == content


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


def test_filename_queries_follow_the_opened_entry(simulator) -> None:
    controller, _, sock = simulator
    directory = prepare_files(controller)
    original = (directory / "complete.json").read_text(encoding="utf-8")
    first = call(sock, "predictor.info", {"filename": "complete.json"})
    assert first["ok"], first
    assert first["result"]["params"]["flux_bias"] == 0.0
    controller.setup_control.new_entry(EntryRequest("another"))
    entry = controller.context_control.get_entry_info()
    assert entry is not None
    other = entry.result_path / "predictors"
    other.mkdir()
    document = json.loads(original)
    document["params"]["flux_bias"] = 0.04
    (other / "complete.json").write_text(json.dumps(document), encoding="utf-8")
    second = call(sock, "predictor.info", {"filename": "complete.json"})
    assert second["ok"], second
    assert second["result"]["params"]["flux_bias"] == 0.04
    assert (directory / "complete.json").read_text(encoding="utf-8") == original


def test_mcp_rpc_preserves_read_only_file_sources_and_named_errors(
    simulator, tmp_path: Path, request
) -> None:
    controller, adapter, _ = simulator
    directory = prepare_files(controller)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
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
        try:
            loaded = invoke(
                "rpc_call",
                {
                    "method": "predictor.predict",
                    "params": {"filename": "complete.json", "device_value": 0.1},
                },
            )
        except GuiRpcError as exc:
            pytest.fail(f"MCP must accept an entry-local predictor filename: {exc}")
        assert loaded == direct
        calibrated = invoke(
            "rpc_call",
            {
                "method": "predictor.calibrate",
                "params": {
                    "filename": "complete.json",
                    "device_value": 0.101,
                    "frequency_mhz": direct["freq_mhz"],
                },
            },
        )
        assert calibrated["flux_bias_after"] == pytest.approx(-0.001, abs=1e-4)
        assert (
            invoke(
                "rpc_call",
                {"method": "predictor.info", "params": {"filename": "partial.json"}},
            )["missing"]
            == MISSING
        )
        with pytest.raises(GuiRpcError) as rejected:
            invoke(
                "rpc_call",
                {"method": "predictor.info", "params": {"filename": "missing.json"}},
            )
        assert rejected.value.reason == "predictor_file_unavailable"
        assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
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
    "model,filename", [("fluxonium", "complete.json"), ("transmon", "transmon.json")]
)
@pytest.mark.parametrize("source", ["filename", "params"])
@pytest.mark.parametrize("target_kind", ["reachable", "below", "above"])
def test_mcp_calibration_reports_actual_frequency_and_clamping_read_only(
    simulator, tmp_path, request, model, filename, source, target_kind
):
    controller, adapter, _ = simulator
    directory = prepare_files(controller)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
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
    selection = (
        {"filename": filename} if source == "filename" else {"params": deepcopy(direct)}
    )
    target = (
        1707.3748654723192
        if target_kind == "reachable"
        else (-100.0 if target_kind == "below" else 20000.0)
    )
    bridge, invoke = mcp_client(adapter.port, tmp_path, request=request)
    try:
        invoke("connect", {"port": adapter.port})
        try:
            result = invoke(
                "rpc_call",
                {
                    "method": "predictor.calibrate",
                    "params": {
                        **selection,
                        "device_value": 0.101,
                        "frequency_mhz": target,
                    },
                },
            )
        except GuiRpcError as exc:
            pytest.fail(f"MCP must calibrate the supplied {source} source: {exc}")
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
        assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
        assert selection == (
            {"filename": filename} if source == "filename" else {"params": direct}
        )
    finally:
        bridge.disconnect()
