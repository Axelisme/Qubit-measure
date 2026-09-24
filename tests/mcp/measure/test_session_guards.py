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
                "context": 4,
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
def test_stale_error_translates_and_refreshes_the_next_request(
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
    retry = send(client, method, params)
    assert retry["expected_versions"][resource] == 7
    assert retry["expected_versions"]["context"] == 10


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


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("editor.commit", {"editor_id": "e", "name": "m"}),
        ("soc.info", {}),
        ("state.has_soc", {}),
        ("context.md_get", {}),
        ("context.md_get_attr", {"key": "x", "attr": "y"}),
        ("context.ml_get", {"name": "x"}),
        ("context.ml_list_roles", {}),
        ("value.list", {}),
        ("value.read", {"key": "x"}),
    ],
)
def test_write_or_unmapped_read_refreshes_the_whole_baseline(
    client: MeasureClient,
    method: str,
    params: dict[str, Any],
) -> None:
    client.observe_versions({"context": 7, "tab:t:cfg": 1, "device:removed": 3})
    set_versions(client, {"context": 8, "tab:t:cfg": 9, "soc": 4})
    send(client, method, params)
    expected = send(client, "tab.run_start", {"tab_id": "t"})["expected_versions"]
    assert expected["context"] == 8
    assert expected["tab:t:cfg"] == 9
    assert expected["soc"] == 4
    assert "device:removed" not in expected


def test_context_read_keeps_its_baseline_when_context_changes_later(
    client: MeasureClient,
) -> None:
    set_versions(client, {"context": 7})
    send(client, "context.md_get", {})
    set_versions(client, {"context": 8})
    assert (
        send(client, "tab.writeback_apply", {"tab_id": "t", "subtab_id": "analysis"})[
            "expected_versions"
        ]["context"]
        == 7
    )
