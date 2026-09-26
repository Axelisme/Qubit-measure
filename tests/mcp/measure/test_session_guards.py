"""Observed RPCs determine the versions attached to subsequent mutations."""

from pathlib import Path
from typing import Any

import pytest

from ._support import MeasureClient, make_client


@pytest.fixture
def client(tmp_path: Path) -> MeasureClient:
    return make_client(tmp_path)


def set_versions(client: MeasureClient, versions: dict[str, int]) -> None:
    client.transport.replies["resources.versions"] = {
        "ok": True,
        "result": {"versions": versions},
    }


def send(client: MeasureClient, method: str, params: dict[str, Any]) -> dict[str, Any]:
    client.transport.replies[method] = {"ok": True, "result": {}}
    client.context.send_gui_rpc(method, params)
    return next(
        params for name, params in reversed(client.transport.sent) if name == method
    )


def context_version_for_load(client: MeasureClient) -> int:
    return send(client, "tab.load_data", {"tab_id": "t", "data_path": "result.h5"})[
        "expected_versions"
    ]["context"]


@pytest.mark.parametrize(
    ("method", "params", "expected"),
    [
        (
            "tab.run_start",
            {"tab_id": "t"},
            {
                "tab:t:cfg": 3,
                "tab:t": 1,
                "soc": 2,
                "device:yoko": 5,
                "device:sgs": 8,
                "devices:__set__": 6,
            },
        ),
        ("tab.save_data", {"tab_id": "t"}, {"tab:t:result": 7, "tab:t:path:data": 9}),
        (
            "tab.load_data",
            {"tab_id": "t", "data_path": "result.h5"},
            {
                "tab:t": 1,
                "tab:t:result": 7,
                "tab:t:analyze": 10,
                "context": 4,
            },
        ),
    ],
)
def test_mutations_attach_only_their_observed_dependencies(
    client: MeasureClient,
    method: str,
    params: dict[str, Any],
    expected: dict[str, int],
) -> None:
    client.observe_versions(
        {
            "tab:t:cfg": 3,
            "tab:t": 1,
            "soc": 2,
            "context": 4,
            "device:yoko": 5,
            "device:sgs": 8,
            "devices:__set__": 6,
            "tab:t:result": 7,
            "tab:t:path:data": 9,
            "tab:t:analyze": 10,
        }
    )
    assert send(client, method, params)["expected_versions"] == expected


def test_new_tab_receipt_establishes_only_the_created_tab_existence(
    client: MeasureClient,
) -> None:
    client.transport.replies["tab.new"] = {
        "ok": True,
        "result": {
            "tab_id": "new-tab",
            "__agent_write_versions": {
                "tab:new-tab": [0, 1],
                "context": [0, 5],
            },
        },
    }
    assert client.call(
        "rpc_call", {"method": "tab.new", "params": {"adapter_name": "fake"}}
    ) == {"tab_id": "new-tab"}
    expected = send(client, "tab.run_start", {"tab_id": "new-tab"})["expected_versions"]
    assert expected["tab:new-tab"] == 1
    assert context_version_for_load(client) == 0


def test_only_explicit_full_tab_and_soc_reads_reveal_their_guard_versions(
    client: MeasureClient,
) -> None:
    set_versions(client, {"tab:t": 3, "soc": 4, "context": 8})
    client.transport.replies["tab.snapshot"] = {
        "ok": True,
        "result": {"tabs": [{"tab_id": "t"}]},
    }
    client.transport.replies["soc.info"] = {
        "ok": True,
        "result": {"cfg": {"gens": []}, "is_mock": True},
    }
    client.call("rpc_call", {"method": "tab.snapshot"})
    client.call("rpc_call", {"method": "soc.info"})
    before = send(client, "tab.run_start", {"tab_id": "t"})["expected_versions"]
    assert before["tab:t"] == 0
    assert before["soc"] == 0
    assert context_version_for_load(client) == 0

    client.call("rpc_call", {"method": "tab.snapshot", "params": {"tab_id": "t"}})
    client.call("rpc_call", {"method": "soc.info", "params": {"include_cfg": True}})
    after = send(client, "tab.run_start", {"tab_id": "t"})["expected_versions"]
    assert after["tab:t"] == 3
    assert after["soc"] == 4
    assert context_version_for_load(client) == 0


def test_unseen_device_membership_is_guarded_at_zero(client: MeasureClient) -> None:
    client.observe_versions({"tab:t:cfg": 1, "tab:t": 1, "soc": 1, "context": 1})
    assert (
        send(client, "tab.run_start", {"tab_id": "t"})["expected_versions"][
            "devices:__set__"
        ]
        == 0
    )


@pytest.mark.parametrize("method", ["tab.writeback_set", "tab.writeback_apply"])
@pytest.mark.parametrize(
    ("pane", "resource", "version"),
    [
        ("analysis", "analyze", 4),
        ("post_analysis", "post_analyze", 6),
    ],
)
def test_writeback_guards_only_the_selected_pane(
    client: MeasureClient,
    method: str,
    pane: str,
    resource: str,
    version: int,
) -> None:
    client.observe_versions(
        {
            "tab:t:result": 7,
            "tab:t:analyze": 4,
            "tab:t:post_analyze": 6,
            "context": 9,
            "tab:t:save_path": 2,
        }
    )
    params = send(client, method, {"tab_id": "t", "subtab_id": pane})
    assert params["expected_versions"] == {
        "tab:t:result": 7,
        f"tab:t:{resource}": version,
        "context": 9,
    }


@pytest.mark.parametrize(
    ("method", "params", "resource", "description"),
    [
        (
            "tab.writeback_apply",
            {"tab_id": "t", "subtab_id": "post_analysis"},
            "tab:t:post_analyze",
            "this tab's post-analysis",
        ),
        ("tab.run_start", {"tab_id": "t"}, "tab:t:cfg", "PRECONDITION_FAILED"),
    ],
)
def test_stale_error_requires_a_new_read_before_refreshing_observed_versions(
    client: MeasureClient,
    method: str,
    params: dict[str, Any],
    resource: str,
    description: str,
) -> None:
    client.observe_versions({resource: 6, "context": 9})
    client.transport.replies[method] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_version",
            "message": "stale",
            "data": {"stale": [resource]},
        },
    }
    set_versions(client, {resource: 7, "context": 10})
    with pytest.raises(RuntimeError, match=description) as error:
        client.context.send_gui_rpc(method, params)
    assert resource not in str(error.value)
    first = next(p for name, p in client.transport.sent if name == method)
    assert first["expected_versions"][resource] == 6
    with pytest.raises(RuntimeError):
        client.context.send_gui_rpc(method, params)
    retry = [p for name, p in client.transport.sent if name == method][-1]
    assert retry["expected_versions"][resource] == 6
    assert context_version_for_load(client) == 9


def test_cfg_snapshot_after_stale_refreshes_only_cfg(client: MeasureClient) -> None:
    client.observe_versions({"tab:t:cfg": 6, "context": 9})
    client.transport.replies["tab.run_start"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_version",
            "message": "stale",
            "data": {"stale": ["tab:t:cfg"]},
        },
    }
    set_versions(client, {"tab:t:cfg": 7, "context": 10})
    with pytest.raises(RuntimeError):
        client.context.send_gui_rpc("tab.run_start", {"tab_id": "t"})
    send(client, "tab.get_cfg", {"tab_id": "t"})
    retried = send(client, "tab.run_start", {"tab_id": "t"})
    assert retried["expected_versions"]["tab:t:cfg"] == 7
    assert context_version_for_load(client) == 9


def test_stale_error_identifies_changed_resources_through_the_rpc_boundary(
    client: MeasureClient,
) -> None:
    client.transport.replies["tab.run_start"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "stale_version",
            "message": "stale",
            "data": {
                "stale": [
                    "context",
                    "soc",
                    "tab:abc123:cfg",
                    "device:flux",
                    "devices:__set__",
                    "arb_waveforms",
                ]
            },
        },
    }
    with pytest.raises(RuntimeError) as error:
        client.context.send_gui_rpc("tab.run_start", {"tab_id": "abc123"})
    message = str(error.value)
    for resource in [
        "the active context (md/ml)",
        "the SoC connection",
        "this tab's cfg",
        "device 'flux'",
        "the set of devices (one added/removed)",
        "the arbitrary waveform asset store",
    ]:
        assert resource in message
    assert "abc123" not in message


@pytest.mark.parametrize("read_method", ["arb_waveform.list", "arb_waveform.preview"])
def test_asset_reads_reveal_the_guard_baseline_for_asset_writes(
    client: MeasureClient,
    read_method: str,
) -> None:
    client.observe_versions({"arb_waveforms": 2, "context": 1})
    set_versions(client, {"arb_waveforms": 3, "context": 1})
    send(client, read_method, {"name": "pulse"})
    set_versions(client, {"arb_waveforms": 4, "context": 2})
    params = send(client, "arb_waveform.set", {"name": "pulse", "recipe": {}})
    assert params["expected_versions"] == {"arb_waveforms": 3}


def test_status_and_unrelated_catalog_read_do_not_accept_unread_gui_cfg_edit(
    client: MeasureClient,
) -> None:
    def respond(method: str, params: dict[str, Any]) -> dict[str, Any]:
        replies = {
            "state.has_project": {"value": False},
            "state.has_active_context": {"value": False},
            "state.has_soc": {"value": False},
            "context.active": {"label": None},
            "device.list": {"devices": []},
            "predictor.info": {"loaded": False},
            "tab.snapshot": {"tabs": []},
            "operation.active": {"operations": []},
            "soc.info": {"is_mock": True},
            "tab.get_cfg": {"cfg": {}},
        }
        return replies[method]

    client.transport.responder = respond
    set_versions(client, {"tab:t:cfg": 3})
    client.call("rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": "t"}})
    set_versions(client, {"tab:t:cfg": 4})  # GUI user changed the cfg after the read.
    client.call("status", {})
    client.call("rpc_call", {"method": "soc.info"})

    def guarded_run(params: dict[str, Any]) -> dict[str, Any]:
        if params["expected_versions"]["tab:t:cfg"] != 4:
            return {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "stale_version",
                    "message": "cfg changed",
                    "data": {"stale": ["tab:t:cfg"]},
                },
            }
        return {"ok": True, "result": {"operation_id": 1}}

    client.transport.replies["tab.run_start"] = guarded_run
    with pytest.raises(RuntimeError) as error:
        client.context.send_gui_rpc("tab.run_start", {"tab_id": "t"})
    assert getattr(error.value, "reason", None) == "stale_version"
    assert (
        next(
            params
            for method, params in reversed(client.transport.sent)
            if method == "tab.run_start"
        )["expected_versions"]["tab:t:cfg"]
        == 3
    )


@pytest.mark.parametrize(
    ("read_method", "read_params", "guard_method", "resource"),
    [
        ("tab.get_cfg", {"tab_id": "t"}, "tab.run_start", "tab:t:cfg"),
        ("editor.get", {"editor_id": "e"}, "editor.commit", "editor:e"),
    ],
)
@pytest.mark.parametrize("prefix", ["modules.readout", "not.there", ""])
def test_partial_or_unmatched_cfg_read_does_not_accept_an_unread_gui_edit(
    client: MeasureClient,
    read_method: str,
    read_params: dict[str, Any],
    guard_method: str,
    resource: str,
    prefix: str,
) -> None:
    def respond(method: str, params: dict[str, Any]) -> dict[str, Any]:
        replies = {
            "state.has_project": {"value": False},
            "state.has_active_context": {"value": False},
            "state.has_soc": {"value": False},
            "context.active": {"label": None},
            "device.list": {"devices": []},
            "predictor.info": {"loaded": False},
            "tab.snapshot": {"tabs": []},
            "operation.active": {"operations": []},
        }
        return replies.get(
            method,
            {"tree": {} if params.get("prefix") == "not.there" else {"readout": 1}},
        )

    client.transport.responder = respond
    set_versions(client, {resource: 3})
    client.call("rpc_call", {"method": read_method, "params": read_params})
    set_versions(client, {resource: 4})  # GUI edits a field outside the partial tree.
    client.call(
        "rpc_call", {"method": read_method, "params": {**read_params, "prefix": prefix}}
    )
    client.call("status", {})

    def guarded(params: dict[str, Any]) -> dict[str, Any]:
        if params["expected_versions"][resource] != 4:
            return {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "stale_version",
                    "message": "cfg changed",
                    "data": {"stale": [resource]},
                },
            }
        return {"ok": True, "result": {"operation_id": 1}}

    client.transport.replies[guard_method] = guarded
    guard_params = (
        {"tab_id": "t"}
        if guard_method == "tab.run_start"
        else {"editor_id": "e", "name": "copy"}
    )
    with pytest.raises(RuntimeError) as error:
        client.context.send_gui_rpc(guard_method, guard_params)
    assert getattr(error.value, "reason", None) == "stale_version"
    assert (
        next(
            params["expected_versions"][resource]
            for method, params in reversed(client.transport.sent)
            if method == guard_method
        )
        == 3
    )


@pytest.mark.parametrize(
    ("read_method", "read_params", "guard_method", "resource"),
    [
        ("tab.get_cfg", {"tab_id": "t"}, "tab.run_start", "tab:t:cfg"),
        ("editor.get", {"editor_id": "e"}, "editor.commit", "editor:e"),
    ],
)
def test_read_reply_cannot_reveal_a_later_unseen_version(
    client: MeasureClient,
    read_method: str,
    read_params: dict[str, Any],
    guard_method: str,
    resource: str,
) -> None:
    set_versions(client, {resource: 3})

    def read_then_gui_edits(params: dict[str, Any]) -> dict[str, Any]:
        set_versions(client, {resource: 4})
        return {"ok": True, "result": {"tree": {"readout": 3}}}

    client.transport.replies[read_method] = read_then_gui_edits
    client.call("rpc_call", {"method": read_method, "params": read_params})

    def guarded(params: dict[str, Any]) -> dict[str, Any]:
        if params["expected_versions"][resource] == 3:
            return {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "stale_version",
                    "message": "cfg changed",
                    "data": {"stale": [resource]},
                },
            }
        return {"ok": True, "result": {"operation_id": 1}}

    client.transport.replies[guard_method] = guarded
    params = (
        {"tab_id": "t"}
        if guard_method == "tab.run_start"
        else {"editor_id": "e", "name": "copy"}
    )
    with pytest.raises(RuntimeError) as error:
        client.context.send_gui_rpc(guard_method, params)
    assert getattr(error.value, "reason", None) == "stale_version"
    sent = [method for method, _ in client.transport.sent]
    assert sent.index("resources.versions") < sent.index(read_method)


@pytest.mark.parametrize(
    ("read_method", "read_params", "guard_method", "guard_params", "resource"),
    [
        ("tab.get_cfg", {"tab_id": "t"}, "tab.run_start", {"tab_id": "t"}, "tab:t:cfg"),
        (
            "editor.get",
            {"editor_id": "e"},
            "editor.commit",
            {"editor_id": "e", "name": "copy"},
            "editor:e",
        ),
    ],
)
def test_failed_cfg_read_and_bare_versions_preserve_previous_observation(
    client: MeasureClient,
    read_method: str,
    read_params: dict[str, Any],
    guard_method: str,
    guard_params: dict[str, Any],
    resource: str,
) -> None:
    set_versions(client, {resource: 3})
    client.transport.replies[read_method] = {
        "ok": True,
        "result": {"tree": {"kind": "section", "children": {}}},
    }
    client.call("rpc_call", {"method": read_method, "params": read_params})
    set_versions(client, {resource: 4})
    client.transport.replies[read_method] = {
        "ok": False,
        "error": {"code": "controller_error", "message": "cfg projection failed"},
    }
    with pytest.raises(RuntimeError, match="cfg projection failed"):
        client.call("rpc_call", {"method": read_method, "params": read_params})
    assert client.context.session.read_version_table() == {resource: 4}
    assert send(client, guard_method, guard_params)["expected_versions"][resource] == 3


def test_unguarded_read_does_not_attach_versions(client: MeasureClient) -> None:
    assert send(client, "tab.snapshot", {"tab_id": "t"}) == {"tab_id": "t"}


@pytest.mark.parametrize("concurrent_change", [False, True])
def test_cfg_read_baseline_does_not_advance_without_another_read(
    client: MeasureClient,
    concurrent_change: bool,
) -> None:
    set_versions(client, {"tab:t:cfg": 3})
    send(client, "tab.get_cfg", {"tab_id": "t"})
    set_versions(client, {"tab:t:cfg": 4 if concurrent_change else 3})
    assert (
        send(client, "tab.run_start", {"tab_id": "t"})["expected_versions"]["tab:t:cfg"]
        == 3
    )


def test_reading_another_cfg_does_not_invalidate_the_target_cfg(
    client: MeasureClient,
) -> None:
    client.observe_versions({"tab:t:cfg": 3, "tab:other:cfg": 2})
    set_versions(client, {"tab:t:cfg": 3, "tab:other:cfg": 4})
    send(client, "tab.get_cfg", {"tab_id": "other"})
    expected = send(client, "tab.run_start", {"tab_id": "t"})["expected_versions"]
    assert expected["tab:t:cfg"] == 3
    assert "tab:other:cfg" not in expected


def test_unrelated_read_reveals_its_own_resource_without_absorbing_cfg_change(
    client: MeasureClient,
) -> None:
    client.observe_versions({"tab:t:cfg": 3, "device:yoko": 5})
    set_versions(client, {"tab:t:cfg": 4, "device:yoko": 6})
    send(client, "device.snapshot", {"name": "yoko"})
    expected = send(client, "tab.run_start", {"tab_id": "t"})["expected_versions"]
    assert expected["tab:t:cfg"] == 3
    assert expected["device:yoko"] == 6


def test_device_list_refreshes_membership_without_masking_device_edit(
    client: MeasureClient,
) -> None:
    client.observe_versions({"device:yoko": 5, "devices:__set__": 2})
    set_versions(client, {"device:yoko": 6, "devices:__set__": 3})
    send(client, "device.list", {})
    expected = send(client, "tab.run_start", {"tab_id": "t"})["expected_versions"]
    assert expected["device:yoko"] == 5
    assert expected["devices:__set__"] == 3


def test_discarding_another_editor_does_not_accept_an_unread_cfg_edit(
    client: MeasureClient,
) -> None:
    def respond(method: str, params: dict[str, Any]) -> dict[str, Any]:
        replies = {
            "state.has_project": {"value": False},
            "state.has_active_context": {"value": False},
            "state.has_soc": {"value": False},
            "context.active": {"label": None},
            "device.list": {"devices": []},
            "predictor.info": {"loaded": False},
            "tab.snapshot": {"tabs": []},
            "operation.active": {"operations": []},
            "tab.get_cfg": {"tree": {"value": 3}},
            "editor.discard": {},
        }
        return replies[method]

    client.transport.responder = respond
    set_versions(client, {"tab:t:cfg": 3})
    client.call("rpc_call", {"method": "tab.get_cfg", "params": {"tab_id": "t"}})
    set_versions(client, {"tab:t:cfg": 4})  # GUI changes only tab t.
    client.call("rpc_call", {"method": "editor.discard", "params": {"editor_id": "e"}})
    client.call("status", {})

    def guarded_run(params: dict[str, Any]) -> dict[str, Any]:
        if params["expected_versions"]["tab:t:cfg"] != 4:
            return {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "stale_version",
                    "message": "cfg changed",
                    "data": {"stale": ["tab:t:cfg"]},
                },
            }
        return {"ok": True, "result": {"operation_id": 1}}

    client.transport.replies["tab.run_start"] = guarded_run
    with pytest.raises(RuntimeError) as error:
        client.context.send_gui_rpc("tab.run_start", {"tab_id": "t"})
    assert getattr(error.value, "reason", None) == "stale_version"
    run_params = [
        params for method, params in client.transport.sent if method == "tab.run_start"
    ][-1]
    assert run_params["expected_versions"]["tab:t:cfg"] == 3


def test_successful_write_refreshes_only_changed_previously_seen_resources(
    client: MeasureClient,
) -> None:
    client.observe_versions({"context": 7, "tab:t:cfg": 1})
    set_versions(client, {"context": 8, "tab:t:cfg": 9, "soc": 4})
    client.transport.replies["editor.commit"] = {
        "ok": True,
        "result": {"__agent_write_versions": {"context": [7, 8]}},
    }
    assert (
        client.context.send_gui_rpc("editor.commit", {"editor_id": "e", "name": "m"})
        == {}
    )
    expected = send(client, "tab.run_start", {"tab_id": "t"})["expected_versions"]
    assert context_version_for_load(client) == 8
    assert expected["tab:t:cfg"] == 1
    assert expected["soc"] == 0


@pytest.mark.parametrize("receipt", [None, {"context": [7, "8"]}, {"context": [7, 7]}])
def test_invalid_write_receipt_does_not_advance_observations(
    client: MeasureClient, receipt: object
) -> None:
    client.observe_versions({"context": 7})
    client.transport.replies["editor.commit"] = {
        "ok": True,
        "result": {"__agent_write_versions": receipt},
    }
    with pytest.raises(RuntimeError) as error:
        client.context.send_gui_rpc("editor.commit", {"editor_id": "e", "name": "m"})
    assert getattr(error.value, "reason", None) == "incompatible_wire"
    assert context_version_for_load(client) == 7


def test_write_does_not_approve_a_prior_unread_edit_to_the_same_cfg(
    client: MeasureClient,
) -> None:
    client.observe_versions({"tab:t:cfg": 3})
    set_versions(client, {"tab:t:cfg": 4})  # GUI edits before this unguarded write.
    client.transport.replies["tab.set_cfg"] = {
        "ok": True,
        "result": {"__agent_write_versions": {"tab:t:cfg": [4, 5]}},
    }
    client.call(
        "rpc_call",
        {
            "method": "tab.set_cfg",
            "params": {
                "tab_id": "t",
                "edits": [{"path": "modules.readout.gain", "value": 0.25}],
            },
        },
    )
    set_versions(client, {"tab:t:cfg": 5})
    expected = send(client, "tab.run_start", {"tab_id": "t"})["expected_versions"]
    assert expected["tab:t:cfg"] == 3


def test_write_reply_does_not_approve_a_later_unseen_gui_edit(
    client: MeasureClient,
) -> None:
    client.observe_versions({"context": 7})

    def commit_then_gui_edits(params: dict[str, Any]) -> dict[str, Any]:
        # The receipt is fixed by the owner-thread handler; the GUI edits later.
        set_versions(client, {"context": 9})
        return {
            "ok": True,
            "result": {"__agent_write_versions": {"context": [7, 8]}},
        }

    client.transport.replies["editor.commit"] = commit_then_gui_edits
    client.call(
        "rpc_call",
        {"method": "editor.commit", "params": {"editor_id": "e", "name": "m"}},
    )
    assert context_version_for_load(client) == 8


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("soc.info", {}),
        ("context.md_get", {}),
        ("context.md_get_attr", {"key": "x", "attr": "y"}),
        ("context.ml_get", {"name": "x"}),
        ("context.ml_list_roles", {}),
        ("value.list", {}),
        ("value.read", {"key": "x"}),
    ],
)
def test_unmapped_read_keeps_unrelated_baseline(
    client: MeasureClient, method: str, params: dict[str, Any]
) -> None:
    client.observe_versions({"context": 7, "tab:t:cfg": 1, "device:removed": 3})
    set_versions(client, {"context": 8, "tab:t:cfg": 9, "soc": 4})
    send(client, method, params)
    expected = send(client, "tab.run_start", {"tab_id": "t"})["expected_versions"]
    assert context_version_for_load(client) == 7
    assert expected["tab:t:cfg"] == 1
    assert expected["soc"] == 0
    assert expected["device:removed"] == 3


def test_full_context_read_uses_pre_read_version_and_partial_reads_cannot_advance_it(
    client: MeasureClient,
) -> None:
    set_versions(client, {"context": 7})

    def full_read_then_gui_edit(params: dict[str, Any]) -> dict[str, Any]:
        set_versions(client, {"context": 8})
        return {
            "ok": True,
            "result": {
                "label": "ctx",
                "md": {"r_f": 6000.0},
                "ml": {"modules": {}, "waveforms": {}},
            },
        }

    client.transport.replies["context.snapshot"] = full_read_then_gui_edit
    client.transport.replies["context.md_get"] = {
        "ok": True,
        "result": {"keys": ["r_f", "unread"]},
    }
    assert client.call("rpc_call", {"method": "context.snapshot"})["md"] == {
        "r_f": 6000.0
    }
    client.call("rpc_call", {"method": "context.md_get"})
    guarded = send(client, "tab.load_data", {"tab_id": "t", "data_path": "old.h5"})
    assert guarded["expected_versions"]["context"] == 7

    client.transport.replies["context.snapshot"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "unserializable_context",
            "message": "cannot fully snapshot the active context",
        },
    }
    with pytest.raises(RuntimeError):
        client.call("rpc_call", {"method": "context.snapshot"})
    retry = send(client, "editor.commit", {"editor_id": "e", "name": "copy"})
    assert retry["expected_versions"]["context"] == 7

    client.transport.replies["context.snapshot"] = {
        "ok": True,
        "result": {"label": "ctx", "md": {"r_f": 6100.0}, "ml": {}},
    }
    client.call("rpc_call", {"method": "context.snapshot"})
    refreshed = send(client, "editor.commit", {"editor_id": "e", "name": "copy"})
    assert refreshed["expected_versions"]["context"] == 8


def test_context_read_keeps_its_baseline_when_context_changes_later(
    client: MeasureClient,
) -> None:
    client.observe_versions({"context": 7})
    send(client, "context.md_get", {})
    set_versions(client, {"context": 8})
    assert (
        send(client, "tab.writeback_apply", {"tab_id": "t", "subtab_id": "analysis"})[
            "expected_versions"
        ]["context"]
        == 7
    )
