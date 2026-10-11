"""Read-only concrete-qubit predictor callers through the public controller/socket.

Installed-model lifecycle cases are retired by D348; the full independent params
wire alternative belongs to the subsequent MCP contract slice.
"""

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
)
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import component_registry

import zcu_lab.components as components

from ._helpers import call, open_client


def test_qubit_queries_and_calibration_preserve_current_point(
    qapp, tmp_path: Path
) -> None:
    before_kinds = component_registry.kinds()
    added = "qubit/fluxonium" not in before_kinds
    if added:
        component_registry.register("qubit/fluxonium", components.QubitSchema)
    installed = component_registry.kinds()
    try:
        controller = Controller(
            State(SessionEnv(parameters=None, soc=None, soccfg=None)),
            Registry(),
            None,
            BaseEventBus(),
            project_root=str(tmp_path),
            params_for=components.params_for,
        )
        try:
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
            controller.setup_control.new_point(
                PointCreateRequest("first", "", CommandOrigin("gui"))
            )
            before = controller.context_control.read_context().document()
            adapter = RemoteControlAdapter(
                controller, ControlOptions(port=0), controller.owner_scheduler
            )
            adapter.start()
            sock = open_client(adapter.port)
            try:
                info = call(sock, "predictor.info", {"qubit": "Q2"})["result"]
                assert info["params"]["EJ"] == 4.1
                assert info["missing"] == [
                    "EC",
                    "EL",
                    "flux_half",
                    "flux_period",
                    "flux_bias",
                ]
                freq = call(
                    sock, "predictor.predict", {"qubit": "Q1", "device_value": 0.1}
                )["result"]["freq_mhz"]
                assert freq == pytest.approx(1707.3748654723192, abs=1e-7)
                result = call(
                    sock,
                    "predictor.calibrate",
                    {"qubit": "Q1", "device_value": 0.101, "frequency_mhz": freq},
                )["result"]
                assert result["flux_bias_after"] == pytest.approx(-0.001, abs=1e-4)
                assert (
                    call(sock, "predictor.info", {"qubit": "Q1"})["result"]["params"][
                        "flux_bias"
                    ]
                    == 0.0
                )
                assert controller.context_control.read_context().document() == before
            finally:
                sock.close()
                adapter.stop()
        finally:
            assert vars(controller)["_background_svc"].quiesce()
        assert component_registry.kinds() == installed
    finally:
        if added:
            component_registry.unregister("qubit/fluxonium")
        assert component_registry.kinds() == before_kinds
