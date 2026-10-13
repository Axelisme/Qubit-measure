"""Form RPC observations and atomic writes on an in-process app connection."""

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request

from ._helpers import Fixture


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
        return json.loads(self.link.outbound.get_nowait())


@pytest.fixture
def form_rpc(qapp) -> Iterator[tuple[Fixture, str, FormRpc]]:
    fixture = Fixture()
    tab_id = fixture.ctrl.new_tab("fake")
    remote = RemoteControlAdapter(
        fixture.ctrl, ControlOptions(port=0), ImmediateScheduler()
    )
    rpc = FormRpc(remote)
    try:
        yield fixture, tab_id, rpc
    finally:
        fixture.ctrl.quiesce()
        fixture.ctrl.tab_control.close_tab(tab_id)
        remote.on_client_close(rpc.link, on_owner_thread=True)


def test_get_cfg_returns_authored_input_and_ordinary_leaf_values(form_rpc) -> None:
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    owner.write({("gain",): {"expr": "2 / 5"}})
    reply = rpc.call("tab.get_cfg", {"tab_id": tab})
    assert reply["ok"], reply
    observed = reply["result"]
    assert observed["input"]["gain"] == {"expr": "2 / 5"}
    assert {tuple(item["path"]): item["value"] for item in observed["values"]}[
        ("gain",)
    ] == 0.4
    assert observed["schema"]["properties"]["gain"]
    assert observed["errors"] == []


def test_edit_cfg_commits_leaf_batch_and_returns_invalid_input_errors(form_rpc) -> None:
    fixture, tab, rpc = form_rpc
    reply = rpc.call(
        "tab.edit_cfg",
        {
            "tab_id": tab,
            "edits": [
                {"path": ["gain"], "value": {"expr": "2 / 5"}},
                {"path": ["sweep", "start"], "value": 8.0},
            ],
        },
    )
    assert reply["ok"], reply
    assert fixture.ctrl.cfg_resources.lookup(tab).input.data["gain"] == {
        "expr": "2 / 5"
    }
    invalid = rpc.call(
        "tab.edit_cfg",
        {
            "tab_id": tab,
            "edits": [
                {"path": ["gain"], "value": {"expr": "1 / 0"}},
            ],
        },
    )
    assert invalid["ok"], invalid
    assert invalid["result"]["input"]["gain"] == {"expr": "1 / 0"}
    assert invalid["result"]["errors"][0]["path"] == ["gain"]
    assert rpc.call("tab.get_cfg", {"tab_id": tab})["result"] == invalid["result"]


def test_reset_returns_fresh_form_defaults_and_keeps_results(form_rpc) -> None:
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    initial = owner.input.data
    owner.write({("gain",): 0.75})
    reply = rpc.call("tab.reset_cfg", {"tab_id": tab})
    assert reply["ok"], reply
    assert reply["result"]["input"] == initial
    assert reply["result"]["errors"] == []


def test_edit_structural_error_rejects_the_entire_batch(form_rpc) -> None:
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    initial = owner.input.data
    reply = rpc.call(
        "tab.edit_cfg",
        {
            "tab_id": tab,
            "edits": [
                {"path": ["gain"], "value": 0.75},
                {"path": ["missing"], "value": 2},
            ],
        },
    )
    assert reply["ok"] is False
    assert reply["error"]["code"] == "invalid_params", reply
    assert owner.input.data == initial
