"""Predictor tool contract through the real GUI socket and shared State."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.resources.qubit_params import FluxDepFit, ParamsProject, QubitParams

from ._helpers import Fixture, call, mcp_client, open_client

pytestmark = pytest.mark.uses_wall_clock

_MODEL = {
    "EJ": 4.0,
    "EC": 1.0,
    "EL": 1.0,
    "flux_half": 0.0,
    "flux_period": 1.0,
}


def test_predictor_install_and_multiple_transitions_share_gui_state(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = Fixture()
    fx.start()
    # The fixture's fake SoC is not a ZCU216; status is outside this seam.
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        assert invoke("rpc_call", {"method": "predictor.info"}) == {"loaded": False}
        installed = invoke(
            "rpc_call", {"method": "predictor.set_model_params", "params": _MODEL}
        )
        gui_info = call(sock, "predictor.info")["result"]
        assert installed == {
            "loaded": True,
            "path": None,
            "EJ": gui_info["EJ"],
            "EC": gui_info["EC"],
            "EL": gui_info["EL"],
            "flux_half": gui_info["flux_half"],
            "flux_period": gui_info["flux_period"],
            "flux_bias": gui_info["flux_bias"],
        }
        assert invoke("rpc_call", {"method": "predictor.info"}) == installed
        for frm, to in ((0, 1), (0, 2)):
            row = invoke(
                "rpc_call",
                {
                    "method": "predictor.predict",
                    "params": {"device_value": 0.5, "from_level": frm, "to_level": to},
                },
            )
            gui = call(
                sock,
                "predictor.predict",
                {"device_value": 0.5, "from_level": frm, "to_level": to},
            )["result"]
            freq = row["freq_mhz"]
            assert isinstance(freq, (int, float))
            assert freq == pytest.approx(gui["freq_mhz"])
            assert freq > 0

        measured = call(
            sock,
            "predictor.predict",
            {"device_value": 0.18, "from_level": 0, "to_level": 1},
        )["result"]["freq_mhz"]
        bad_transition = call(
            sock,
            "predictor.calibrate",
            {
                "device_value": 0.15,
                "frequency_mhz": measured,
                "from_level": 1,
                "to_level": 1,
            },
        )
        assert bad_transition["ok"] is False
        assert bad_transition["error"]["code"] == "invalid_params"
        assert bad_transition["error"]["reason"] == "invalid_transition"
        assert call(sock, "predictor.info")["result"]["flux_bias"] == 0.0
        calibrated = invoke(
            "rpc_call",
            {
                "method": "predictor.calibrate",
                "params": {"device_value": 0.15, "frequency_mhz": measured},
            },
        )
        assert calibrated["flux_bias_before"] == pytest.approx(0.0)
        assert calibrated["flux_bias_after"] == pytest.approx(
            call(sock, "predictor.info")["result"]["flux_bias"]
        )
        assert calibrated["flux_bias_after"] != pytest.approx(0.0)
        at_point = invoke(
            "rpc_call",
            {"method": "predictor.predict", "params": {"device_value": 0.15}},
        )
        assert at_point["freq_mhz"] == pytest.approx(measured, rel=1e-5)

        params_path = tmp_path / "params.json"
        params = QubitParams(params_path)
        params.ensure_project(ParamsProject("chip", "qubit"))
        params.set_fluxdep_fit(
            FluxDepFit(
                EJ=4.5,
                EC=1.2,
                EL=0.9,
                flux_half=0.31,
                flux_int=0.71,
                flux_period=0.8,
            )
        )
        from_file = invoke(
            "rpc_call",
            {
                "method": "predictor.load",
                "params": {"path": str(params_path), "flux_bias": 0.13},
            },
        )
        gui_file = call(sock, "predictor.info")["result"]
        assert from_file["path"] == str(params_path)
        assert from_file["EJ"] == gui_file["EJ"] == pytest.approx(4.5)
        assert from_file["flux_bias"] == gui_file["flux_bias"] == pytest.approx(0.13)
        assert invoke("rpc_call", {"method": "predictor.info"}) == from_file
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()


def test_predictor_missing_model_and_failed_file_load_preserve_gui_state(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = Fixture()
    fx.start()
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        missing = call(
            sock,
            "predictor.calibrate",
            {"device_value": 0.15, "frequency_mhz": 4567.0},
        )
        assert missing["ok"] is False
        assert missing["error"]["code"] == "precondition_failed"
        assert missing["error"]["reason"] == "predictor_not_loaded"
        with pytest.raises(GuiRpcError, match="Failed to load predictor"):
            invoke(
                "rpc_call",
                {
                    "method": "predictor.load",
                    "params": {"path": str(tmp_path / "missing.json")},
                },
            )
        assert call(sock, "predictor.info")["result"] == {"loaded": False}
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()
