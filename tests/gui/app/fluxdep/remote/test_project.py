"""Entry target admission and failed opening through the shipped RPC route."""

import pytest
from zcu_tools.gui.app.fluxdep.event_bus import ProjectChangedPayload
from zcu_tools.gui.event_bus import EventMeta
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


def test_setup_emits_native_agent_fact_only_after_guard_admission(
    route_harness: RouteHarness,
    fluxonium_entry,
) -> None:
    client = route_harness.client()
    facts: list[EventMeta] = []

    def record(_payload: ProjectChangedPayload, meta: EventMeta) -> None:
        facts.append(meta)

    subscription = route_harness.ctrl.bus.subscribe_with_meta(
        ProjectChangedPayload, record
    )
    try:
        assert route_harness.request(client, "project.info")["ok"]
        assert facts == []
        assert route_harness.request(
            client,
            "project.setup",
            project_root=str(fluxonium_entry.result_path.parent.parent),
            entry="synthetic",
            component="Q1",
        )["ok"]
        assert len(facts) == 1
        assert facts[0].origin.kind == "agent"
        assert facts[0].origin.client_id is not None
        info = route_harness.request(client, "project.info")
        assert info["ok"]
        assert "result" in info
        assert info["result"]["component"] == "Q1"
    finally:
        subscription.unsubscribe()


@pytest.mark.parametrize("component", ["missing", ""])
def test_unknown_component_preserves_previous_target(
    route_harness, fluxonium_entry, component
):
    client = route_harness.client()
    root = str(fluxonium_entry.result_path.parent.parent)
    assert route_harness.request(
        client, "project.setup", project_root=root, entry="synthetic"
    )["ok"]
    before = route_harness.request(client, "project.info")
    try:
        rejected = route_harness.request(
            client,
            "project.setup",
            project_root=root,
            entry="synthetic",
            component=component,
        )
    except RemoteError as error:
        assert error.code is ErrorCode.INVALID_PARAMS
    else:
        assert not rejected["ok"]
    assert route_harness.request(client, "project.info") == {
        **before,
        "id": str(route_harness.request_id),
    }


@pytest.mark.parametrize(
    "params",
    [{}, {"project_root": "root"}, {"project_root": False, "entry": "synthetic"}],
)
def test_malformed_setup_is_rejected_before_publication(route_harness, params):
    client = route_harness.client()
    with pytest.raises(RemoteError) as rejected:
        route_harness.request(client, "project.setup", **params)
    assert rejected.value.code is ErrorCode.INVALID_PARAMS
    assert route_harness.ctrl.state.project is None
