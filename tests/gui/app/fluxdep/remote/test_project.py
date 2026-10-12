"""Project identity inputs and GUI-computed paths through the shipped RPC route."""

import os

import pytest
from zcu_tools.gui.app.fluxdep.event_bus import ProjectChangedPayload
from zcu_tools.gui.event_bus import EventMeta
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


def test_setup_emits_native_agent_fact_only_after_guard_admission(
    route_harness: RouteHarness,
) -> None:
    client = route_harness.client()
    facts: list[EventMeta] = []

    def record(_payload: ProjectChangedPayload, meta: EventMeta) -> None:
        facts.append(meta)

    subscription = route_harness.ctrl.bus.subscribe_with_meta(ProjectChangedPayload, record)
    try:
        assert route_harness.request(client, "project.info")["ok"]
        assert facts == []
        assert route_harness.request(
            client, "project.setup", chip_name="chip", qub_name="Q1"
        )["ok"]
        assert len(facts) == 1
        assert facts[0].origin.kind == "agent"
        assert facts[0].origin.client_id is not None
    finally:
        subscription.unsubscribe()


def test_setup_computes_paths_from_identity_and_controller_root(
    route_harness: RouteHarness,
) -> None:
    client = route_harness.client()
    root = route_harness.ctrl.get_project_root()
    for name in ("first", "晶片"):
        reply = route_harness.request(client, "project.setup", chip_name=name, qub_name="Q:1")
        assert reply["ok"] is True
        project = {
            "chip_name": name,
            "qub_name": "Q:1",
            "result_dir": os.path.join(root, "result", name, "Q:1"),
            "data_root": os.path.join(root, "Database", name, "Q:1"),
        }
        assert reply["result"] == {"project": project}
        observed = route_harness.request(client, "project.info")
        assert observed["ok"] is True
        assert observed["result"] == project
    assert route_harness.ctrl.state.version.get("project") == 2


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"chip_name": "chip"},
        {"chip_name": "", "qub_name": "Q1"},
        {"chip_name": None, "qub_name": "Q1"},
        {"chip_name": 1, "qub_name": "Q1"},
        {"chip_name": "chip", "qub_name": False},
        {"chip_name": "chip", "qub_name": "Q1", "result_dir": "/override"},
        {"chip_name": "chip", "qub_name": "Q1", "data_root": "/override"},
    ],
)
def test_malformed_setup_is_rejected_before_publication(
    route_harness: RouteHarness,
    params: dict[str, object],
) -> None:
    client = route_harness.client()
    initial = route_harness.ctrl.state.project
    with pytest.raises(RemoteError) as rejected:
        route_harness.request(client, "project.setup", **params)
    assert rejected.value.code is ErrorCode.INVALID_PARAMS
    assert route_harness.ctrl.state.project is initial
    assert route_harness.ctrl.state.version.get("project") == 0
