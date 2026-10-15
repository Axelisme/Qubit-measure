"""Real parameter-container application shared by point and snapshot contracts."""

from collections.abc import Mapping
from dataclasses import dataclass
from socket import socket

from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.remote import RemoteControlAdapter
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.session.context_types import EntryInfo, EntryRequest

from tests.gui.app.measure._synthetic import SampleAdapter

from ._helpers import recv_response, send


class PointAdapter(SampleAdapter):
    pass


@dataclass
class PointApplication:
    state: State
    controller: Controller
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
