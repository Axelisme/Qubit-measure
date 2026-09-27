"""Predictor tool contract through the real GUI socket and shared State."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.session import GuiRpcError

from ._helpers import Fixture, call, mcp_client, open_client

pytestmark = pytest.mark.uses_wall_clock

_MODEL = {
    "EJ": 4.0,
    "EC": 1.0,
    "EL": 1.0,
    "flux_half": 0.0,
    "flux_period": 1.0,
}


def _prediction_rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        pytest.fail("predict must return a list")
    rows: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict) or not all(isinstance(key, str) for key in row):
            pytest.fail("each prediction must have named fields")
        rows.append(row)
    return rows


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
        assert invoke("predictor_info", {}) == {"loaded": False}
        installed = invoke("predictor_load", {"model": _MODEL})
        gui_info = call(sock, "predictor.info")["result"]
        assert installed == {
            "loaded": True,
            "source": "model",
            "EJ": gui_info["EJ"],
            "EC": gui_info["EC"],
            "EL": gui_info["EL"],
            "flux_half": gui_info["flux_half"],
            "flux_period": gui_info["flux_period"],
            "flux_bias": gui_info["flux_bias"],
        }
        assert invoke("predictor_info", {}) == installed
        predicted = _prediction_rows(
            invoke("predict", {"value": 0.5, "transitions": [[0, 1], [0, 2]]})
        )
        assert [row["transition"] for row in predicted] == [[0, 1], [0, 2]]
        for row in predicted:
            transition = row["transition"]
            assert isinstance(transition, list)
            frm, to = transition
            assert isinstance(frm, int) and isinstance(to, int)
            gui = call(
                sock,
                "predictor.predict",
                {"device_value": 0.5, "from_level": frm, "to_level": to},
            )["result"]
            freq = row["freq_mhz"]
            assert isinstance(freq, (int, float))
            assert freq == pytest.approx(gui["freq_mhz"])
            assert freq > 0
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()


def test_predictor_rejects_ambiguous_load_without_replacing_gui_model(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx = Fixture()
    fx.start()
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        with pytest.raises((GuiRpcError, ValueError), match="exactly one"):
            invoke("predictor_load", {"path": "unused.json", "model": _MODEL})
        assert call(sock, "predictor.info")["result"] == {"loaded": False}
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()
