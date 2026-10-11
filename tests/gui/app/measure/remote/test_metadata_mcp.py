"""Native general.ext values and complete/summary/path wire readback."""

from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.mcp.measure.session import GuiRpcError

from ._helpers import Fixture, call, mcp_client, open_client


@pytest.mark.uses_wall_clock
def test_md_rpc_shares_gui_values_and_rejects_reserved_keys(
    qapp, tmp_path: Path
) -> None:
    fx = Fixture(project_root=str(tmp_path), empty_project=True)
    fx.start()
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    sock = open_client(fx.service.port)
    try:
        invoke("connect", {"port": fx.service.port})
        invoke("rpc_call", {"method": "entry.new", "params": {"name": "q"}})
        invoke("rpc_call", {"method": "point.new", "params": {"label": "base"}})
        assert (
            invoke(
                "rpc_call",
                {"method": "context.snapshot", "params": {"summaries": True}},
            )["point"]["general"]["ext"]
            == {}
        )
        matrix: list[YamlValue] = [[1, 2], [3, 4]]
        context = fx.ctrl.context_control
        for key, value in (("freq", 5.0), ("matrix", matrix), ("bias", 0.25)):
            context.set_parameter(
                context.read_context().target, f"general.ext.{key}", value
            )
        assert invoke(
            "rpc_call", {"method": "context.snapshot", "params": {"summaries": True}}
        )["point"]["general"]["ext"] == {
            "freq": 5.0,
            "matrix": "2 × 2 matrix",
            "bias": 0.25,
        }
        assert (
            invoke(
                "rpc_call",
                {
                    "method": "context.snapshot",
                    "params": {"path": "point.general.ext.matrix"},
                },
            )["value"]
            == matrix
        )
        invoke(
            "rpc_call",
            {"method": "point.new", "params": {"label": "clone", "clone_from": "base"}},
        )
        invoke("rpc_call", {"method": "point.use", "params": {"label": "base"}})
        assert (
            invoke(
                "rpc_call",
                {
                    "method": "context.snapshot",
                    "params": {"path": "point.general.ext.matrix"},
                },
            )["value"]
            == matrix
        )
        assert (
            call(sock, "context.snapshot", {"path": "point.general.ext.freq"})[
                "result"
            ]["value"]
            == 5.0
        )
        assert (
            call(sock, "context.snapshot", {"path": "point.general.ext.bias"})[
                "result"
            ]["value"]
            == 0.25
        )
        with pytest.raises(GuiRpcError, match="missing") as missing:
            invoke(
                "rpc_call",
                {
                    "method": "context.snapshot",
                    "params": {"path": "point.general.ext.missing"},
                },
            )
        assert missing.value.code == "invalid_params"
        captured = context.read_context()
        point = captured.document()
        general = point.general.model_copy(
            update={"ext": {**point.general.ext, "complex": 1 + 2j}}
        )
        parameters = ParameterContext(
            point=point.model_copy(update={"general": general}),
            library=captured.library,
            target=captured.target,
            roles=captured.roles,
            source_faults=captured.source_faults,
            storage=captured.storage,
        )
        fx.state.set_context(replace(fx.state.session_env, parameters=parameters))
        assert call(sock, "context.snapshot", {})["result"]["point"]["general"][
            "ext"
        ] == {
            "freq": 5.0,
            "matrix": matrix,
            "bias": 0.25,
            "complex": {"__complex__": [1.0, 2.0]},
        }
    finally:
        bridge.disconnect()
        sock.close()
        fx.stop()
