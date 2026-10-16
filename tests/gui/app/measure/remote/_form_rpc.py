"""In-process Form RPC collaborator and registered module adapter."""

import json
from collections.abc import Callable
from typing import Any

from zcu_tools.cfg_model import PulseCfg, Range
from zcu_tools.gui.app.measure.remote import RemoteControlAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request

from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg


class ImmediateScheduler:
    """Dispatch on the test's current owner thread."""

    def is_owner_thread(self) -> bool:
        return True

    def post(self, callback: Callable[[], None]) -> None:
        callback()

    def call[T](self, callback: Callable[[], T]) -> T:
        return callback()


class FormRpc:
    """Record real dispatch replies without a socket or a rendered window."""

    def __init__(self, remote: RemoteControlAdapter) -> None:
        self.remote = remote
        self.link = ClientLink("form-test", token_required=False)
        remote.on_client_open(self.link)
        self.call("control.connect", {})

    def call(self, method: str, params: dict[str, object]) -> dict[str, Any]:
        self.remote.route(self.link, Request("form-request", method, params))
        return json.loads(self.link.outbound.get(timeout=2.0))


class ModuleRpcCfg(SampleCfg):
    pulse: PulseCfg | None = None


class ModuleRpcAdapter(SampleAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[ModuleRpcCfg]:
        del ctx
        return Form(ModuleRpcCfg, {"sweep": Range.between(1.0, 2.0, expts=3)})
