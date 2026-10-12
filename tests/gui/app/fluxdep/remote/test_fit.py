"""Fit replacement and observation through the native RPC route."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload
from zcu_tools.gui.event_bus import EventMeta
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


@pytest.fixture
def search_db(tmp_path: Path) -> str:
    path = tmp_path / "search.h5"
    with h5py.File(path, "w") as database:
        database.create_dataset("fluxs", data=[0.0, 0.5, 1.0])
        database.create_dataset("params", data=[[3.0, 0.2, 0.1], [9.5, 2.0, 1.5]])
        database.create_dataset("energies", data=np.zeros((2, 3, 4)))
    return str(path)


def test_database_info_reports_native_ranges_without_changing_fit(
    route_harness: RouteHarness, search_db: str,
) -> None:
    initial = route_harness.ctrl.state.fit
    reply = route_harness.request(route_harness.client(), "fit.database_info", path=search_db)
    assert reply["ok"]
    assert "result" in reply
    assert reply["result"] == {
        "path": search_db, "entries": 2, "levels": 4,
        "EJ": [3.0, 9.5], "EC": [0.2, 2.0], "EL": [0.1, 1.5], "flux": [0.0, 1.0],
    }
    assert route_harness.ctrl.state.fit is initial


def _inputs(search_db: str, **overrides: object) -> dict[str, object]:
    inputs: dict[str, object] = {
        "search_db": search_db,
        "EJb": [3, 9.5],
        "ECb": [0.2, 2],
        "ELb": [0.1, 1.5],
        "transitions": {"transitions": [[0, 1], [1, 2]], "mirror2": []},
    }
    inputs.update(overrides)
    return inputs


def test_fit_replacement_clears_result_emits_agent_fact_and_does_not_start_search(
    route_harness: RouteHarness, search_db: str,
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
        inputs = _inputs(search_db,
            transitions={"mirror": [[0, 3]], "r_f": 6, "sample_f": 7.5},
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
            "mirror": [(0, 3)],
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
            client, "fit.set_params", **_inputs(search_db, transitions={}, r_f=None)
        )
        assert reply["ok"]
        assert "result" in reply
        assert reply["result"]["fit"] == {
            "has_result": False,
            "params": None,
            **_inputs(search_db, transitions={}),
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
    route_harness: RouteHarness, search_db: str,
    overrides: dict[str, object],
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "fit.result")["ok"]
    initial = route_harness.ctrl.state.fit
    reply = route_harness.request(client, "fit.set_params", **_inputs(search_db, **overrides))
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == "invalid_params"
    assert route_harness.ctrl.state.fit is initial
    assert route_harness.ctrl.state.version.get("fit") == 0
    assert route_harness.request(client, "fit.set_params", **_inputs(search_db))["ok"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"search_db": ""},
        {"search_db": None},
        {"EJb": (1, 2)},
        {"transitions": []},
        {"r_f": True},
        {"sample_f": "2"},
    ],
)
def test_top_level_fit_admission_uses_shared_param_validation(
    route_harness: RouteHarness, search_db: str,
    overrides: dict[str, object],
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "fit.result")["ok"]
    initial = route_harness.ctrl.state.fit
    with pytest.raises(RemoteError) as caught:
        route_harness.request(client, "fit.set_params", **_inputs(search_db, **overrides))
    assert caught.value.code is ErrorCode.INVALID_PARAMS
    assert route_harness.ctrl.state.fit is initial
    assert route_harness.ctrl.state.version.get("fit") == 0


@pytest.mark.parametrize("transitions", [
    {"transitons": []}, {"transitions1": []}, {"mirror01": []},
    {"transitions": [[-1, 1]]}, {"transitions": [[0, 4]]},
    {"mirror4": []}, {"transitions4": [[0, 1]]},
])
def test_transition_rejection_does_not_publish(
    route_harness: RouteHarness, search_db: str, transitions: dict[str, object],
) -> None:
    client = route_harness.client()
    initial = route_harness.ctrl.state.fit
    reply = route_harness.request(client, "fit.set_params", database_path=search_db, **_inputs(search_db, transitions=transitions))
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == "invalid_params"
    assert route_harness.ctrl.state.fit is initial
    assert route_harness.ctrl.state.version.get("fit") == 0


@pytest.mark.parametrize("transitions,code", [
    ({}, None), ({"transitions": [], "mirror": []}, None),
    ({"transitons": []}, "invalid_params"),
    ({"transitions2": []}, "internal_error"),
    ({"transitions": [[0, 1]]}, "internal_error"),
])
def test_database_is_read_only_when_validation_needs_levels(
    route_harness: RouteHarness, tmp_path: Path, transitions: dict[str, object], code: str | None,
) -> None:
    initial = route_harness.ctrl.state.fit
    reply = route_harness.request(
        route_harness.client(), "fit.set_params",
        database_path=str(tmp_path / "missing.h5"),
        **_inputs(str(tmp_path / "missing.h5"), transitions=transitions),
    )
    if code is None:
        assert reply["ok"]
    else:
        assert not reply["ok"]
        assert "error" in reply
        assert reply["error"]["code"] == code
        assert route_harness.ctrl.state.fit is initial


def test_bounds_semantics_remain_search_owned(
    route_harness: RouteHarness, search_db: str,
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "fit.result")["ok"]
    reply = route_harness.request(
        client,
        "fit.set_params",
        **_inputs(search_db, EJb=[9, -1]),
    )
    assert reply["ok"]
    assert route_harness.ctrl.state.fit.EJb == (9.0, -1.0)

