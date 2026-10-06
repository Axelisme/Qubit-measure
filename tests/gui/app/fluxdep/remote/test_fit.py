"""Fit replacement and observation through the native RPC route."""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload
from zcu_tools.gui.event_bus import EventMeta
from zcu_tools.gui.project import ProjectInfo
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply


def _inputs(**overrides: object) -> dict[str, object]:
    inputs: dict[str, object] = {
        "database_path": "search-database.h5",
        "EJb": [3, 9.5],
        "ECb": [0.2, 2],
        "ELb": [0.1, 1.5],
        "transitions": {"transitions1": [[0, 1], [1, 2]], "mirror2": []},
    }
    inputs.update(overrides)
    return inputs


def _assert_stale(reply: RouteReply, key: str = "fit") -> None:
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == "precondition_failed"
    assert reply["error"].get("reason") == "stale_version"
    assert reply["error"].get("data") == {"stale": [key]}


def test_fit_replacement_clears_result_emits_agent_fact_and_does_not_start_search(
    route_harness: RouteHarness,
) -> None:
    client = route_harness.client()
    state = route_harness.ctrl.state
    state.set_fit_result((5.0, 1.0, 0.5))
    facts: list[tuple[FitChangedPayload, EventMeta]] = []

    def record(payload: FitChangedPayload, meta: EventMeta) -> None:
        facts.append((payload, meta))

    subscription = route_harness.ctrl.bus.subscribe_with_meta(FitChangedPayload, record)
    try:
        read = route_harness.request(client, "fit.result")
        assert read["ok"]
        assert "result" in read
        assert read["result"]["params"] == {"EJ": 5.0, "EC": 1.0, "EL": 0.5}
        inputs = _inputs(
            transitions={"custom category": [[-1, 3]], "r_f": 6, "sample_f": 7.5},
            r_f=8,
            sample_f=9,
        )
        reply = route_harness.request(client, "fit.set_params", **inputs)
        assert reply["ok"]
        assert "result" in reply
        assert reply["result"] == {
            "fit": {
                "has_result": False,
                "params": None,
                **inputs,
            }
        }
        assert state.fit.EJb == (3.0, 9.5)
        assert state.fit.transitions == {
            "custom category": [(-1, 3)],
            "r_f": 6.0,
            "sample_f": 7.5,
        }
        assert state.fit.r_f == 8.0
        assert state.fit.sample_f == 9.0
        assert state.version.get("fit") == 2
        assert len(facts) == 1
        assert not facts[0][0].has_result
        assert facts[0][1].origin.kind == "agent"
        assert facts[0][1].origin.client_id is not None

        # Replacing again needs no hidden read. Null/omission clears frequencies
        # and the transition mapping rather than merging the old configuration.
        reply = route_harness.request(
            client, "fit.set_params", **_inputs(transitions={}, r_f=None)
        )
        assert reply["ok"]
        assert "result" in reply
        assert reply["result"]["fit"] == {
            "has_result": False,
            "params": None,
            **_inputs(transitions={}),
            "r_f": None,
            "sample_f": None,
        }
        assert state.fit.transitions == {}
        assert state.fit.r_f is None and state.fit.sample_f is None
        assert state.version.get("fit") == 3
        assert len(facts) == 2
        # The harness has no search runtime. A hidden start would fail the
        # successful request, and the owner must still have no operation.
        assert route_harness.ctrl.search.current is None
    finally:
        subscription.unsubscribe()


@pytest.mark.parametrize(
    "read_method",
    [
        "resources.versions",
        "state.check",
        "selection.pointcloud",
        "selection.snapshot",
        "project.info",
        "spectrum.list",
    ],
)
def test_non_fit_reads_do_not_unlock_fit(
    route_harness: RouteHarness,
    read_method: str,
) -> None:
    client = route_harness.client()
    initial = route_harness.ctrl.state.fit
    assert route_harness.request(client, read_method)["ok"]
    _assert_stale(route_harness.request(client, "fit.set_params", **_inputs()))
    assert route_harness.ctrl.state.fit is initial
    assert route_harness.ctrl.state.version.get("fit") == 0


def test_fit_guard_is_per_client_and_native_publication_requires_reread(
    route_harness: RouteHarness,
) -> None:
    first, second = route_harness.client(), route_harness.client()
    assert route_harness.request(first, "fit.result")["ok"]
    _assert_stale(route_harness.request(second, "fit.set_params", **_inputs()))
    assert route_harness.request(second, "fit.result")["ok"]
    assert route_harness.request(first, "fit.set_params", **_inputs())["ok"]
    assert route_harness.request(second, "resources.versions")["ok"]
    _assert_stale(route_harness.request(second, "fit.set_params", **_inputs()))
    assert route_harness.request(second, "fit.result")["ok"]
    assert route_harness.request(second, "fit.set_params", **_inputs())["ok"]
    _assert_stale(route_harness.request(first, "fit.set_params", **_inputs()))
    assert route_harness.request(first, "fit.result")["ok"]

    route_harness.ctrl.set_fit_params("GUI-db", (1, 2), (2, 3), (3, 4), {}, None, None)
    _assert_stale(route_harness.request(first, "fit.set_params", **_inputs()))
    assert route_harness.ctrl.state.fit.database_path == "GUI-db"
    assert route_harness.request(first, "fit.result")["ok"]
    assert route_harness.request(first, "fit.set_params", **_inputs())["ok"]
    route_harness.disconnect(first)
    fresh = route_harness.client()
    _assert_stale(route_harness.request(fresh, "fit.set_params", **_inputs()))


@pytest.mark.parametrize("baseline", ["matching", "stale", "unread"])
def test_fit_self_write_refreshes_only_matching_observed_side_effects(
    route_harness: RouteHarness,
    baseline: str,
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "fit.result")["ok"]
    if baseline != "unread":
        assert route_harness.request(client, "project.info")["ok"]
    if baseline == "stale":
        route_harness.ctrl.setup_project(
            ProjectInfo(chip_name="external", qub_name="Q1")
        )

    # A native synchronous fact consumer publishes another resource in the
    # same owner turn. Shared tracking, not this adapter, decides its baseline.
    def publish_project(_payload: FitChangedPayload) -> None:
        route_harness.ctrl.setup_project(
            ProjectInfo(chip_name="side-effect", qub_name="Q1")
        )

    subscription = route_harness.ctrl.bus.subscribe(FitChangedPayload, publish_project)
    try:
        assert route_harness.request(client, "fit.set_params", **_inputs())["ok"]
    finally:
        subscription.unsubscribe()
    reply = route_harness.request(
        client, "project.setup", chip_name="next", qub_name="Q2"
    )
    if baseline == "matching":
        assert reply["ok"]
        assert route_harness.ctrl.state.project.chip_name == "next"
    else:
        _assert_stale(reply, "project")
        assert route_harness.ctrl.state.project.chip_name == "side-effect"


@pytest.mark.parametrize(
    "overrides",
    [
        {"EJb": []},
        {"ECb": [1]},
        {"ELb": [1, 2, 3]},
        {"EJb": [True, 2]},
        {"ECb": [1, "2"]},
        {"ELb": [float("inf"), 2]},
        {"EJb": [1, float("nan")]},
        {"ECb": [10**400, 2]},
        {"transitions": {"r_f": True}},
        {"transitions": {"sample_f": float("-inf")}},
        {"transitions": {"r_f": None}},
        {"transitions": {"transitions1": "not pairs"}},
        {"transitions": {"transitions1": [[0]]}},
        {"transitions": {"transitions1": [[0, 1, 2]]}},
        {"transitions": {"transitions1": [[0.0, 1]]}},
        {"transitions": {"transitions1": [[False, 1]]}},
        {"transitions": {"transitions1": [[0, True]]}},
        {"transitions": {"transitions1": [(0, 1)]}},
        {"transitions": {1: [[0, 1]]}},
        {"r_f": float("nan")},
        {"sample_f": float("inf")},
    ],
)
def test_nested_fit_admission_fails_before_publication(
    route_harness: RouteHarness,
    overrides: dict[str, object],
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "fit.result")["ok"]
    initial = route_harness.ctrl.state.fit
    reply = route_harness.request(client, "fit.set_params", **_inputs(**overrides))
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == "invalid_params"
    assert route_harness.ctrl.state.fit is initial
    assert route_harness.ctrl.state.version.get("fit") == 0
    assert route_harness.request(client, "fit.set_params", **_inputs())["ok"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"database_path": ""},
        {"database_path": None},
        {"EJb": (1, 2)},
        {"transitions": []},
        {"r_f": True},
        {"sample_f": "2"},
    ],
)
def test_top_level_fit_admission_uses_shared_param_validation(
    route_harness: RouteHarness,
    overrides: dict[str, object],
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "fit.result")["ok"]
    initial = route_harness.ctrl.state.fit
    with pytest.raises(RemoteError) as caught:
        route_harness.request(client, "fit.set_params", **_inputs(**overrides))
    assert caught.value.code is ErrorCode.INVALID_PARAMS
    assert route_harness.ctrl.state.fit is initial
    assert route_harness.ctrl.state.version.get("fit") == 0


def test_fit_domain_semantics_are_not_reimplemented_at_wire_admission(
    route_harness: RouteHarness,
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "fit.result")["ok"]
    reply = route_harness.request(
        client,
        "fit.set_params",
        **_inputs(EJb=[9, -1], transitions={"unknown category": [[-2, -3]]}),
    )
    assert reply["ok"]
    assert route_harness.ctrl.state.fit.EJb == (9.0, -1.0)
    assert route_harness.ctrl.state.fit.transitions == {"unknown category": [(-2, -3)]}
