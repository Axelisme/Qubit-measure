"""Project observation and setup through the shipped RPC route."""

from __future__ import annotations

import os

import pytest
from zcu_tools.gui.app.fluxdep.event_bus import ProjectChangedPayload
from zcu_tools.gui.event_bus import EventMeta
from zcu_tools.gui.project import ProjectInfo
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply


def _assert_project_stale(reply: RouteReply) -> None:
    assert not reply["ok"]
    assert "error" in reply
    error = reply["error"]
    assert error["code"] == "precondition_failed"
    assert error.get("reason") == "stale_version"
    assert error.get("data") == {"stale": ["project"]}


def test_zero_version_requires_a_full_project_read_on_each_client(
    route_harness: RouteHarness,
) -> None:
    first, second = route_harness.client(), route_harness.client()
    initial = route_harness.ctrl.state.project
    _assert_project_stale(
        route_harness.request(first, "project.setup", chip_name="chip", qub_name="Q1")
    )
    assert route_harness.ctrl.state.project is initial
    assert route_harness.ctrl.state.version.get("project") == 0

    read = route_harness.request(first, "project.info")
    assert read["ok"] is True
    assert read["result"]["chip_name"] == initial.chip_name
    assert read["result"]["qub_name"] == initial.qub_name
    assert read["result"]["result_dir"] == initial.result_dir
    assert read["result"]["database_path"] == initial.database_path
    _assert_project_stale(
        route_harness.request(second, "project.setup", chip_name="other", qub_name="Q2")
    )
    assert route_harness.ctrl.state.project is initial
    assert route_harness.ctrl.state.version.get("project") == 0

    assert route_harness.request(
        first, "project.setup", chip_name="chip", qub_name="Q1"
    )["ok"]
    _assert_project_stale(
        route_harness.request(second, "project.setup", chip_name="other", qub_name="Q2")
    )
    assert route_harness.ctrl.state.project.chip_name == "chip"
    assert route_harness.ctrl.state.version.get("project") == 1


@pytest.mark.parametrize(
    "read_method",
    [
        "resources.versions",
        "state.check",
        "selection.pointcloud",
        "fit.result",
        "spectrum.list",
    ],
)
def test_other_resource_and_derived_reads_do_not_unlock_project(
    route_harness: RouteHarness,
    read_method: str,
) -> None:
    client = route_harness.client()
    initial = route_harness.ctrl.state.project
    assert route_harness.request(client, read_method)["ok"]
    _assert_project_stale(
        route_harness.request(client, "project.setup", chip_name="chip", qub_name="Q1")
    )
    assert route_harness.ctrl.state.project is initial
    assert route_harness.ctrl.state.version.get("project") == 0


def test_other_client_and_gui_publications_require_an_explicit_reread(
    route_harness: RouteHarness,
) -> None:
    first, second = route_harness.client(), route_harness.client()
    assert route_harness.request(first, "project.info")["ok"]
    assert route_harness.request(second, "project.info")["ok"]
    assert route_harness.request(
        first, "project.setup", chip_name="first", qub_name="Q1"
    )["ok"]

    # Reading versions does not repair an already stale full-read baseline.
    assert route_harness.request(second, "resources.versions")["ok"]
    _assert_project_stale(
        route_harness.request(
            second, "project.setup", chip_name="second", qub_name="Q2"
        )
    )
    assert route_harness.ctrl.state.project.chip_name == "first"
    assert route_harness.ctrl.state.version.get("project") == 1

    assert route_harness.request(second, "project.info")["ok"]
    assert route_harness.request(
        second, "project.setup", chip_name="second", qub_name="Q2"
    )["ok"]
    _assert_project_stale(
        route_harness.request(first, "project.setup", chip_name="old", qub_name="Q1")
    )
    assert route_harness.ctrl.state.project.chip_name == "second"

    assert route_harness.request(first, "project.info")["ok"]
    route_harness.ctrl.setup_project(ProjectInfo(chip_name="GUI", qub_name="Q3"))
    for client in (first, second):
        _assert_project_stale(
            route_harness.request(
                client, "project.setup", chip_name="old", qub_name="Q1"
            )
        )
    assert route_harness.ctrl.state.project.chip_name == "GUI"
    assert route_harness.ctrl.state.version.get("project") == 3


def test_setup_emits_native_agent_fact_only_after_guard_admission(
    route_harness: RouteHarness,
) -> None:
    first, second = route_harness.client(), route_harness.client()
    facts: list[EventMeta] = []

    def record(_payload: ProjectChangedPayload, meta: EventMeta) -> None:
        facts.append(meta)

    subscription = route_harness.ctrl.bus.subscribe_with_meta(
        ProjectChangedPayload, record
    )
    try:
        _assert_project_stale(
            route_harness.request(
                first, "project.setup", chip_name="chip", qub_name="Q1"
            )
        )
        assert facts == []
        assert route_harness.request(first, "project.info")["ok"]
        assert facts == []
        assert route_harness.request(
            first, "project.setup", chip_name="chip", qub_name="Q1"
        )["ok"]
        assert len(facts) == 1
        assert facts[0].origin.kind == "agent"
        assert facts[0].origin.client_id is not None
        _assert_project_stale(
            route_harness.request(
                second, "project.setup", chip_name="other", qub_name="Q2"
            )
        )
        assert len(facts) == 1
        assert route_harness.ctrl.state.version.get("project") == 1
    finally:
        subscription.unsubscribe()


def test_successful_self_write_preserves_the_existing_observation(
    route_harness: RouteHarness,
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "project.info")["ok"]
    for name in ("first", "second"):
        reply = route_harness.request(
            client,
            "project.setup",
            chip_name=name,
            qub_name="Q1",
            result_dir="relative-output",
            database_path="raw-data-root",
        )
        assert reply["ok"] is True
        assert reply["result"] == {
            "project": {
                "chip_name": name,
                "qub_name": "Q1",
                "result_dir": "relative-output",
                "database_path": "raw-data-root",
            }
        }
        assert route_harness.ctrl.state.project.chip_name == name
    assert route_harness.ctrl.state.version.get("project") == 2


@pytest.mark.parametrize(
    "paths",
    [
        {},
        {"result_dir": None, "database_path": None},
        {"result_dir": "", "database_path": ""},
    ],
)
def test_setup_uses_native_defaults_anchored_at_the_controller_root(
    route_harness: RouteHarness,
    paths: dict[str, object],
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "project.info")["ok"]
    reply = route_harness.request(
        client,
        "project.setup",
        chip_name="晶片",
        qub_name="Q:1",
        **paths,
    )
    assert reply["ok"] is True
    root = route_harness.ctrl.get_project_root()
    assert reply["result"] == {
        "project": {
            "chip_name": "晶片",
            "qub_name": "Q:1",
            "result_dir": os.path.join(root, "result", "晶片", "Q:1"),
            "database_path": os.path.join(root, "Database", "晶片", "Q:1"),
        }
    }
    assert route_harness.ctrl.state.project.root_dir == root


def test_reconnected_client_cannot_reuse_the_previous_connections_read(
    route_harness: RouteHarness,
) -> None:
    old_client = route_harness.client()
    assert route_harness.request(old_client, "project.info")["ok"]
    route_harness.disconnect(old_client)
    fresh_client = route_harness.client()
    _assert_project_stale(
        route_harness.request(
            fresh_client, "project.setup", chip_name="chip", qub_name="Q1"
        )
    )
    assert route_harness.ctrl.state.version.get("project") == 0
    assert route_harness.request(fresh_client, "project.info")["ok"]
    assert route_harness.request(
        fresh_client, "project.setup", chip_name="chip", qub_name="Q1"
    )["ok"]


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"chip_name": "chip"},
        {"chip_name": "", "qub_name": "Q1"},
        {"chip_name": None, "qub_name": "Q1"},
        {"chip_name": 1, "qub_name": "Q1"},
        {"chip_name": "chip", "qub_name": False},
        {"chip_name": "chip", "qub_name": "Q1", "result_dir": []},
        {"chip_name": "chip", "qub_name": "Q1", "database_path": 10},
    ],
)
def test_malformed_setup_is_rejected_before_publication(
    route_harness: RouteHarness,
    params: dict[str, object],
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "project.info")["ok"]
    initial = route_harness.ctrl.state.project
    with pytest.raises(RemoteError) as rejected:
        route_harness.request(client, "project.setup", **params)
    assert rejected.value.code is ErrorCode.INVALID_PARAMS
    assert route_harness.ctrl.state.project is initial
    assert route_harness.ctrl.state.version.get("project") == 0
    # A failed invocation does not consume the valid observation.
    assert route_harness.request(
        client, "project.setup", chip_name="chip", qub_name="Q1"
    )["ok"]
