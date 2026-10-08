"""Real parameter-container application shared by point and snapshot contracts."""

from collections.abc import Mapping
from dataclasses import dataclass
from socket import socket

from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.remote import RemoteControlAdapter
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.session.context_types import EntryInfo, EntryRequest
from zcu_tools.resources.entry import RoleSpec

from zcu_lab.v2.twotone.time_domain.t1.gui import T1Adapter

from ._helpers import recv_response, send


class PointAdapter(T1Adapter):
    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        return {"driver": RoleSpec("fake/drive/*")}

    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
        return CfgSchema(
            CfgSectionSpec(fields={"rate": ScalarSpec("Rate", float)}),
            CfgSectionValue(fields={"rate": EvalValue("driver.rate")}),
        )


class PointController(Controller):
    """Give the fixture owned background teardown without exposing app APIs."""

    def quiesce(self) -> None:
        self._background_svc.quiesce()


@dataclass
class PointApplication:
    state: State
    controller: PointController
    remote: RemoteControlAdapter
    client: socket

    def rpc(self, method: str, params: Mapping[str, object]) -> dict[str, object]:
        send(
            self.client,
            {"id": "point-contract", "method": method, "params": dict(params)},
        )
        return recv_response(self.client, "point-contract")

    def result(self, method: str, params: Mapping[str, object]) -> dict[str, object]:
        reply = self.rpc(method, params)
        assert reply["ok"] is True, reply
        result = reply["result"]
        assert isinstance(result, dict)
        return result

    def open_entry(self) -> EntryInfo:
        self.controller.setup_control.new_entry(EntryRequest("entry"))
        info = self.controller.context_control.get_entry_info()
        assert info is not None
        return info
