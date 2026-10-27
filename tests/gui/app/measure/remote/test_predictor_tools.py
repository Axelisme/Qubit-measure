"""Read-only predictor source, coordinate and error contracts through RPC/MCP."""

import json
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

COMPLETE = {"model": "fluxonium", "EJ": 5.0, "EC": 1.0, "EL": 0.5}
TRANSMON = {"model": "transmon", "EJmax": 20.0, "EC": 0.3, "d": 0.0}
FLUX = {
    "offset": 0.25,
    "period": {"coil": -2.0, "bias": 4.0},
    "unit": {"coil": "A", "bias": "V"},
}
VALUES = {"coil": -0.4, "bias": 0.6}  # 0.6 Phi0.
MISSING = ["EC", "EL"]


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
    """Create synthetic v2 predictor sources in an opened entry without a point."""
    controller.setup_control.new_entry(EntryRequest("queries"))
    entry = controller.context_control.get_entry_info()
    assert entry is not None
    directory = entry.result_path / "predictors"
    directory.mkdir()
    for filename, inputs in (
        ("complete.json", COMPLETE),
        ("partial.json", {"model": "fluxonium", "EJ": 4.1}),
        ("transmon.json", TRANSMON),
    ):
        document = {
            "format": "zcu.predictor",
            "version": 2,
            "model": inputs["model"],
            "params": {key: value for key, value in inputs.items() if key != "model"},
            "flux": FLUX,
            "source": {"run_id": None, "created_at": None},
        }
        (directory / filename).write_text(json.dumps(document), encoding="utf-8")
    return directory


@pytest.mark.parametrize("with_flux", [False, True])
def test_info_preserves_partial_energy_and_reports_optional_flux(
    simulator, with_flux: bool
) -> None:
    _, _, sock = simulator
    source: dict[str, object] = {"params": {"model": "fluxonium", "EJ": 4.1}}
    if with_flux:
        source["flux"] = FLUX
    reply = call(sock, "predictor.info", source)
    assert reply["ok"], reply
    assert reply["result"] == {
        "model": "fluxonium",
        "params": {"EJ": 4.1, "EC": None, "EL": None},
        "missing": MISSING,
        "flux": FLUX if with_flux else None,
    }


def test_filename_and_inline_prediction_agree_without_writes_or_a_point(
    simulator,
) -> None:
    controller, _, sock = simulator
    directory = prepare_files(controller)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    replies = [
        call(sock, "predictor.predict", {**source, "device_values": VALUES})
        for source in (
            {"filename": "complete.json"},
            {"params": COMPLETE, "flux": FLUX},
        )
    ]
    assert all(reply["ok"] for reply in replies), replies
    assert replies[0]["result"] == replies[1]["result"]
    assert replies[0]["result"]["freq_mhz"] == pytest.approx(
        1707.3748654723192, abs=1e-7
    )
    info = call(sock, "predictor.info", {"filename": "complete.json"})
    assert info["result"]["flux"] == FLUX
    assert controller.context_control.current_point_label() is None
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


@pytest.mark.parametrize(
    "model,filename,direct",
    [("fluxonium", "complete.json", COMPLETE), ("transmon", "transmon.json", TRANSMON)],
)
@pytest.mark.parametrize("target_kind", ["reachable", "below", "above"])
def test_mcp_calibration_returns_flux_correction_and_actual_frequency_without_writes(
    simulator, tmp_path: Path, request, model, filename, direct, target_kind
) -> None:
    controller, adapter, _ = simulator
    directory = prepare_files(controller)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    values = {"coil": -0.402, "bias": 0.6}
    target = (
        1707.3748654723192
        if target_kind == "reachable"
        else (-100.0 if target_kind == "below" else 20000.0)
    )
    bridge, invoke = mcp_client(adapter.port, tmp_path, request=request)
    try:
        invoke("connect", {"port": adapter.port})
        results = [
            invoke(
                "rpc_call",
                {
                    "method": "predictor.calibrate",
                    "params": {
                        **source,
                        "device_values": values,
                        "frequency_mhz": target,
                    },
                },
            )
            for source in ({"filename": filename}, {"params": direct, "flux": FLUX})
        ]
        assert results[0] == results[1]
        result = results[0]
        assert result["clamped"] is (target_kind != "reachable")
        corrected = {**FLUX, "offset": 0.25 + result["offset_correction"]}
        actual = invoke(
            "rpc_call",
            {
                "method": "predictor.predict",
                "params": {
                    "params": direct,
                    "flux": corrected,
                    "device_values": values,
                },
            },
        )
        assert actual["freq_mhz"] == pytest.approx(result["frequency_mhz"], abs=1e-6)
        if target_kind == "reachable":
            assert result["frequency_mhz"] == pytest.approx(target, abs=1e-6)
            if model == "fluxonium":
                assert result["offset_correction"] == pytest.approx(-0.001, abs=1e-7)
        assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
        assert FLUX["offset"] == 0.25
    finally:
        bridge.disconnect()


@pytest.mark.parametrize(
    "method", ["predictor.info", "predictor.predict", "predictor.calibrate"]
)
@pytest.mark.parametrize(
    "source",
    [
        {},
        {"filename": "complete.json", "params": COMPLETE},
        {"filename": "complete.json", "flux": FLUX},
    ],
)
def test_sources_are_exclusive_and_filename_flux_cannot_be_overridden(
    simulator, method, source
) -> None:
    _, _, sock = simulator
    reply = call(
        sock, method, {**source, "device_values": VALUES, "frequency_mhz": 1707.0}
    )
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "invalid_predictor_source"


@pytest.mark.parametrize("method", ["predictor.predict", "predictor.calibrate"])
def test_inline_queries_require_explicit_flux(simulator, method) -> None:
    _, _, sock = simulator
    reply = call(
        sock,
        method,
        {"params": COMPLETE, "device_values": VALUES, "frequency_mhz": 1707.0},
    )
    assert not reply["ok"]
    assert reply["error"]["reason"] == "missing_predictor_flux"
    assert "flux" in reply["error"]["message"]


@pytest.mark.parametrize("method", ["predictor.predict", "predictor.calibrate"])
@pytest.mark.parametrize("filename", [False, True])
def test_queries_name_a_missing_device_instead_of_reading_cache(
    simulator, method, filename
) -> None:
    controller, _, sock = simulator
    prepare_files(controller)
    source = (
        {"filename": "complete.json"}
        if filename
        else {"params": COMPLETE, "flux": FLUX}
    )
    reply = call(
        sock,
        method,
        {**source, "device_values": {"coil": -0.4}, "frequency_mhz": 1707.0},
    )
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert "bias" in reply["error"]["message"]


@pytest.mark.parametrize(
    "method", ["predictor.info", "predictor.predict", "predictor.calibrate"]
)
@pytest.mark.parametrize(
    "flux",
    [None, {}, {**FLUX, "offset": True}, {**FLUX, "unit": {}}, {**FLUX, "extra": 1}],
)
def test_inline_flux_uses_v2_validation(simulator, method, flux) -> None:
    _, _, sock = simulator
    reply = call(
        sock,
        method,
        {
            "params": COMPLETE,
            "flux": flux,
            "device_values": VALUES,
            "frequency_mhz": 1707.0,
        },
    )
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"]["reason"] == "invalid_predictor_flux"


@pytest.mark.parametrize("method", ["predictor.predict", "predictor.calibrate"])
@pytest.mark.parametrize("source_kind", ["filename", "params"])
def test_missing_energy_fields_are_named_without_source_mutation(
    simulator, method, source_kind
) -> None:
    controller, _, sock = simulator
    directory = prepare_files(controller)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    source = (
        {"filename": "partial.json"}
        if source_kind == "filename"
        else {"params": {"model": "fluxonium", "EJ": 4.1}, "flux": FLUX}
    )
    reply = call(
        sock, method, {**source, "device_values": VALUES, "frequency_mhz": 1707.0}
    )
    assert not reply["ok"]
    assert reply["error"]["reason"] == "missing_predictor_params"
    assert reply["error"]["data"]["missing"] == MISSING
    assert all(name in reply["error"]["message"] for name in MISSING)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


@pytest.mark.parametrize(
    "params,field",
    [
        ({"EJ": 5.0}, "model"),
        ({"model": "unknown"}, "model"),
        ({**COMPLETE, "EC": "1"}, "EC"),
        ({**COMPLETE, "EL": True}, "EL"),
        ({**COMPLETE, "EJ": -1.0}, "EJ"),
        ({**COMPLETE, "unexpected": 1.0}, "unexpected"),
    ],
)
def test_invalid_direct_parameters_have_a_named_error(simulator, params, field) -> None:
    _, _, sock = simulator
    reply = call(
        sock,
        "predictor.predict",
        {"params": params, "flux": FLUX, "device_values": VALUES},
    )
    assert not reply["ok"]
    assert reply["error"]["reason"] == "invalid_predictor_params"
    assert field in reply["error"]["message"]


@pytest.mark.parametrize("value", [True, "1", None, float("inf")])
def test_invalid_device_values_name_the_device(simulator, value) -> None:
    _, _, sock = simulator
    reply = call(
        sock,
        "predictor.predict",
        {"params": COMPLETE, "flux": FLUX, "device_values": {**VALUES, "bias": value}},
    )
    assert not reply["ok"]
    assert reply["error"]["code"] == "invalid_params"
    assert "bias" in reply["error"]["message"]


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
) -> None:
    _, adapter, sock = simulator
    service = FailingPredictorService(failure)
    adapter.predictor_control = PredictorControlFacet(service)
    with caplog.at_level("ERROR"):
        reply = call(
            sock,
            method,
            {
                "params": COMPLETE,
                "flux": FLUX,
                "device_values": VALUES,
                "frequency_mhz": 1707.0,
            },
        )
    assert not reply["ok"]
    assert reply["error"]["code"] == "controller_error"
    assert len(service.requests) == 1
    assert any(
        record.exc_info is not None and record.exc_info[1] is failure
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "filename",
    ["../complete.json", "/complete.json", "complete", "", "nested/complete.json"],
)
def test_filename_requires_an_entry_local_json_name(simulator, filename) -> None:
    controller, _, sock = simulator
    prepare_files(controller)
    reply = call(sock, "predictor.info", {"filename": filename})
    assert not reply["ok"]
    assert reply["error"]["reason"] == "invalid_predictor_file"


def test_filename_without_an_entry_has_failed_precondition(simulator) -> None:
    _, _, sock = simulator
    reply = call(sock, "predictor.info", {"filename": "complete.json"})
    assert not reply["ok"]
    assert reply["error"]["code"] == "precondition_failed"
    assert reply["error"]["reason"] == "no_entry"


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
    assert reply["error"]["reason"] == "invalid_predictor_file"
    assert path.read_text(encoding="utf-8") == content


def test_mcp_discovery_errors_and_file_sources_follow_the_opened_entry(
    simulator, tmp_path: Path, request
) -> None:
    controller, adapter, _ = simulator
    directory = prepare_files(controller)
    original = (directory / "complete.json").read_text(encoding="utf-8")
    bridge, invoke = mcp_client(adapter.port, tmp_path, request=request)
    try:
        invoke("connect", {"port": adapter.port, "mode": "readonly"})
        files = invoke("rpc_call", {"method": "predictor.list_files"})
        assert files == {"files": ["complete.json", "partial.json", "transmon.json"]}
        info = invoke(
            "rpc_call",
            {"method": "predictor.info", "params": {"filename": "partial.json"}},
        )
        assert info["missing"] == MISSING
        assert info["flux"] == FLUX
        with pytest.raises(GuiRpcError) as missing:
            invoke(
                "rpc_call",
                {"method": "predictor.info", "params": {"filename": "missing.json"}},
            )
        assert missing.value.reason == "predictor_file_unavailable"
        controller.setup_control.new_entry(EntryRequest("another"))
        assert invoke("rpc_call", {"method": "predictor.list_files"}) == {"files": []}
        entry = controller.context_control.get_entry_info()
        assert entry is not None
        other = entry.result_path / "predictors"
        other.mkdir()
        document = json.loads(original)
        document["flux"]["offset"] = 0.04
        (other / "complete.json").write_text(json.dumps(document), encoding="utf-8")
        assert (
            invoke(
                "rpc_call",
                {"method": "predictor.info", "params": {"filename": "complete.json"}},
            )["flux"]["offset"]
            == 0.04
        )
        assert (directory / "complete.json").read_text(encoding="utf-8") == original
    finally:
        bridge.disconnect()


def test_transmon_info_does_not_mix_flux_into_missing_energy_fields(simulator) -> None:
    _, _, sock = simulator
    direct = {"model": "transmon", "EJmax": 20.0, "d": 0.0}
    reply = call(sock, "predictor.info", {"params": direct})
    assert reply["ok"], reply
    assert reply["result"] == {
        "model": "transmon",
        "params": {"EJmax": 20.0, "EC": None, "d": 0.0},
        "missing": ["EC"],
        "flux": None,
    }
    invalid = call(sock, "predictor.info", {"params": {**direct, "EL": 1.0}})
    assert invalid["error"]["reason"] == "invalid_predictor_params"
