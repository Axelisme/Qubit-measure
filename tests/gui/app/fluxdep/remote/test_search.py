"""Search admission and operation observation through the shipped RPC route."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import (
    DatabaseSearchResult,
    ParamBounds,
    SearchCancelled,
    SearchExecution,
)
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload, SearchChangedPayload
from zcu_tools.gui.app.fluxdep.search import FluxDepSearchOwner
from zcu_tools.gui.app.fluxdep.services import fit
from zcu_tools.gui.event_bus import EventMeta
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.session.operation_handles import AwaitResult, OperationOutcome

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply
from tests.gui.app.fluxdep.remote._search_harness import PumpedRouteHarness


@dataclass
class SearchCase:
    """Controllable numeric kernel only; route, owners, executor and waits are real.

    release gates worker completion; entered confirms the detached kernel ran.
    failure injects a worker failure, while ignore_cancel models a success after
    the last cancellation checkpoint. captured records detached kernel inputs.
    """

    harness: PumpedRouteHarness
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    failure: Exception | None = None
    ignore_cancel: bool = False
    captured: list[
        tuple[
            NDArray[np.float64], NDArray[np.float64], str, TransitionDict, ParamBounds
        ]
    ] = field(default_factory=list)

    def compute(
        self,
        fluxs: NDArray[np.float64],
        freqs: NDArray[np.float64],
        datapath: str,
        transitions: TransitionDict,
        bounds: ParamBounds,
        *,
        execution: SearchExecution | None = None,
    ) -> DatabaseSearchResult:
        """Block off-owner, then return a numeric result or requested failure."""
        assert not self.harness.owner.is_owner_thread()
        self.captured.append(
            (fluxs.copy(), freqs.copy(), datapath, transitions, bounds)
        )
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("test kernel was not released")
        if self.failure is not None:
            raise self.failure
        if (
            not self.ignore_cancel
            and execution is not None
            and execution.cancel_requested is not None
            and execution.cancel_requested()
        ):
            raise SearchCancelled("requested")
        return DatabaseSearchResult(
            params=(5.0, 1.0, 0.5),
            best_distance=0.1,
            best_scale=1.0,
            best_index=0,
            entry_results=np.array([[0.1, 1.0]]),
            entry_params=np.array([[5.0, 1.0, 0.5]]),
            fluxs=fluxs,
            freqs=freqs,
            predicted_freqs=freqs.copy(),
            bounds=bounds,
        )


@pytest.fixture
def search_case(
    cross_controller: Controller, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[SearchCase]:
    harness = PumpedRouteHarness.create(str(tmp_path), cross_controller.state)
    harness.ctrl.set_fit_params(
        "search-db.h5",
        (2, 15),
        (0.2, 2),
        (0.1, 2),
        {"transitions": [(0, 1)]},
        6.0,
        None,
    )
    case = SearchCase(harness)
    monkeypatch.setattr(fit, "search_database", case.compute)
    try:
        yield case
    finally:
        case.release.set()
        harness.close()


_READS = {
    "project": ("project.info", {}),
    "fit": ("fit.result", {}),
    "selection": ("selection.snapshot", {}),
    "spectrums:__set__": ("spectrum.list", {}),
    "spectrum:a": ("spectrum.snapshot", {"name": "a"}),
    "spectrum:empty": ("spectrum.snapshot", {"name": "empty"}),
    "spectrum:b": ("spectrum.snapshot", {"name": "b"}),
}


def _observe(
    harness: PumpedRouteHarness, client: ClientLink, *, skip: str | None = None
) -> None:
    for key, (method, params) in _READS.items():
        if key != skip:
            assert harness.request(client, method, **params)["ok"]


def _result(reply: RouteReply) -> dict[str, object]:
    assert reply["ok"]
    assert "result" in reply
    return reply["result"]


def _error(reply: RouteReply, code: str, reason: str) -> None:
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == code
    assert reply["error"].get("reason") == reason


def _start(case: SearchCase, client: ClientLink) -> int:
    _observe(case.harness, client)
    reply = _result(case.harness.request(client, "fit.search"))
    token = reply["token"]
    assert isinstance(token, int) and token > 0
    assert case.entered.wait(2)
    return token


@pytest.mark.parametrize("missing", list(_READS))
def test_search_requires_every_full_observation_including_empty_sources(
    search_case: SearchCase, missing: str
) -> None:
    h = search_case.harness
    client = h.client()
    _observe(h, client, skip=missing)
    _error(h.request(client, "fit.search"), "precondition_failed", "stale_version")
    assert h.ctrl.search.current is None
    assert not search_case.entered.is_set()
    method, params = _READS[missing]
    assert h.request(client, method, **params)["ok"]
    assert h.request(client, "fit.search")["ok"]


def test_search_observation_is_per_client_and_operation_reads_do_not_unlock(
    search_case: SearchCase,
) -> None:
    h = search_case.harness
    first, second = h.client(), h.client()
    token = _start(search_case, first)
    assert _result(h.request(second, "operation.status")) == {
        "activity": {"token": token, "status": "pending", "error": None}
    }
    assert _result(h.request(second, "operation.await", token=token, timeout=0)) == {
        "token": token,
        "reason": "timeout",
        "outcome": None,
        "feedback": None,
    }
    assert _result(h.request(second, "operation.cancel", token=token)) == {
        "token": token,
        "cancel_requested": True,
    }
    _error(h.request(second, "fit.search"), "precondition_failed", "stale_version")
    _error(h.request(first, "fit.search"), "precondition_failed", "search_busy")
    assert h.ctrl.search.active_token == token
    assert len(search_case.captured) == 1


def test_search_commits_native_result_before_await_and_does_not_refresh_fit(
    search_case: SearchCase,
) -> None:
    h = search_case.harness
    client = h.client()
    facts: list[tuple[SearchChangedPayload | FitChangedPayload, EventMeta]] = []

    def record(
        payload: SearchChangedPayload | FitChangedPayload, meta: EventMeta
    ) -> None:
        facts.append((payload, meta))

    subscriptions = [
        h.ctrl.bus.subscribe_with_meta(SearchChangedPayload, record),
        h.ctrl.bus.subscribe_with_meta(FitChangedPayload, record),
    ]
    try:
        token = _start(search_case, client)
        assert h.ctrl.state.fit.params is None
        assert _result(h.request(client, "operation.status", token=token)) == {
            "activity": {"token": token, "status": "pending", "error": None}
        }
        awaiting = h.request_async(client, "operation.await", token=token)
        search_case.release.set()
        assert _result(h.wait(awaiting)) == {
            "token": token,
            "reason": "completed",
            "outcome": {"status": "finished", "error": None},
            "feedback": None,
        }
        assert h.ctrl.state.fit.params == (5.0, 1.0, 0.5)
        assert h.ctrl.search.outcome(token) == OperationOutcome("finished")
        assert h.ctrl.search.result is not None
        assert _result(h.request(client, "operation.status", token=None)) == {
            "activity": {"token": token, "status": "finished", "error": None}
        }
        assert [type(payload) for payload, _ in facts] == [
            SearchChangedPayload,
            FitChangedPayload,
            SearchChangedPayload,
        ]
        assert (
            isinstance(facts[0][0], SearchChangedPayload)
            and facts[0][0].status == "pending"
        )
        assert isinstance(facts[1][0], FitChangedPayload) and facts[1][0].has_result
        assert (
            isinstance(facts[2][0], SearchChangedPayload)
            and facts[2][0].status == "finished"
        )
        assert all(meta.origin.kind == "agent" for _, meta in facts)
        assert len({meta.origin.client_id for _, meta in facts}) == 1
        np.testing.assert_array_equal(search_case.captured[0][0], [0, 0.5, 1, 0.5])
        assert search_case.captured[0][2] == "search-db.h5"
        assert search_case.captured[0][3] == {"transitions": [(0, 1)], "r_f": 6.0}
        assert search_case.captured[0][4] == ParamBounds(
            EJ=(2, 15), EC=(0.2, 2), EL=(0.1, 2)
        )
        _error(h.request(client, "fit.search"), "precondition_failed", "stale_version")
        _error(
            h.request(
                client,
                "fit.set_params",
                database_path="new-db",
                EJb=[2, 15],
                ECb=[0.2, 2],
                ELb=[0.1, 2],
                transitions={},
            ),
            "precondition_failed",
            "stale_version",
        )
        read = _result(h.request(client, "fit.result"))
        assert read["has_result"] is True
        assert read["params"] == {"EJ": 5.0, "EC": 1.0, "EL": 0.5}
        assert h.request(client, "fit.search")["ok"]
    finally:
        for subscription in subscriptions:
            subscription.unsubscribe()


@pytest.mark.parametrize("source", ["project", "fit", "selection", "spectrum:empty"])
def test_gui_changes_require_new_search_observations(
    search_case: SearchCase, source: str
) -> None:
    h = search_case.harness
    client = h.client()
    _observe(h, client)
    if source == "project":
        h.ctrl.setup_project(h.ctrl.state.project)
    elif source == "fit":
        h.ctrl.set_fit_params("new-db", (2, 15), (0.2, 2), (0.1, 2), {}, None, None)
    elif source == "selection":
        h.ctrl.set_selection(np.ones(4, dtype=np.bool_))
    else:
        h.ctrl.set_points("empty", np.array([]), np.array([]))
    reply = h.request(client, "fit.search")
    _error(reply, "precondition_failed", "stale_version")
    assert h.ctrl.search.current is None
    _observe(h, client)
    assert h.request(client, "fit.search")["ok"]


@pytest.mark.parametrize("terminal", ["finished", "failed", "cancelled"])
def test_cancel_receipt_is_not_outcome_and_terminal_cancel_is_noop(
    search_case: SearchCase, terminal: str
) -> None:
    h = search_case.harness
    client, canceller = h.client(), h.client()
    token = _start(search_case, client)
    if terminal == "finished":
        search_case.ignore_cancel = True
    elif terminal == "failed":
        search_case.failure = OSError("database read failed")
    waiting = h.request_async(client, "operation.await", token=token, timeout=30)
    assert _result(h.request(canceller, "operation.cancel", token=token)) == {
        "token": token,
        "cancel_requested": True,
    }
    assert h.ctrl.search.outcome(token) is None
    assert _result(h.request(canceller, "operation.status", token=token))[
        "activity"
    ] == {"token": token, "status": "pending", "error": None}
    assert not waiting.done()
    search_case.release.set()
    error = "database read failed" if terminal == "failed" else None
    expected = {
        "token": token,
        "reason": "completed",
        "outcome": {"status": terminal, "error": error},
        "feedback": None,
    }
    assert _result(h.wait(waiting)) == expected
    assert _result(h.request(client, "operation.cancel", token=token)) == {
        "token": token,
        "cancel_requested": True,
    }
    assert (
        _result(h.request(client, "operation.await", token=token, timeout=0))
        == expected
    )
    assert _result(h.request(client, "operation.status", token=token)) == {
        "activity": {"token": token, "status": terminal, "error": error}
    }
    assert h.ctrl.state.fit.has_result is (terminal == "finished")


def test_wait_timeout_does_not_cancel_and_stale_delivery_is_failed_outcome(
    search_case: SearchCase,
) -> None:
    h = search_case.harness
    client = h.client()
    token = _start(search_case, client)
    assert _result(h.request(client, "operation.await", token=token, timeout=0.01)) == {
        "token": token,
        "reason": "timeout",
        "outcome": None,
        "feedback": None,
    }
    h.ctrl.set_points("empty", np.array([]), np.array([]))
    search_case.release.set()
    assert _result(h.request(client, "operation.await", token=token)) == {
        "token": token,
        "reason": "completed",
        "outcome": {"status": "failed", "error": "search inputs changed"},
        "feedback": None,
    }
    assert h.ctrl.state.fit.params is None


def test_reconnect_status_finds_gui_latest_and_explicit_old_handle(
    search_case: SearchCase,
) -> None:
    h = search_case.harness
    first = h.client()
    assert _result(h.request(first, "operation.status")) == {"activity": None}
    token = h.ctrl.search.start()
    h.disconnect(first)
    client = h.client()
    assert _result(h.request(client, "operation.status")) == {
        "activity": {"token": token, "status": "pending", "error": None}
    }
    search_case.release.set()
    assert (
        _result(h.request(client, "operation.await", token=token))["reason"]
        == "completed"
    )
    search_case.release.clear()
    search_case.entered.clear()
    new_token = h.ctrl.search.start()
    assert search_case.entered.wait(2)
    assert new_token != token
    assert _result(h.request(client, "operation.status"))["activity"] == {
        "token": new_token,
        "status": "pending",
        "error": None,
    }
    assert _result(h.request(client, "operation.status", token=token))["activity"] == {
        "token": token,
        "status": "finished",
        "error": None,
    }
    _error(h.request(client, "fit.search"), "precondition_failed", "stale_version")


@pytest.mark.parametrize(
    "method", ["operation.status", "operation.cancel", "operation.await"]
)
@pytest.mark.parametrize("token", [-1, 0, 9999])
def test_unknown_explicit_tokens_are_nominal_errors(
    search_case: SearchCase, method: str, token: int
) -> None:
    h = search_case.harness
    reply = h.request(h.client(), method, token=token)
    _error(reply, "invalid_params", "unknown_operation")
    assert h.ctrl.search.current is None


@pytest.mark.parametrize(
    "method", ["operation.status", "operation.cancel", "operation.await"]
)
@pytest.mark.parametrize("token", [True, "1", 1.0])
def test_operation_token_json_admission(
    search_case: SearchCase, method: str, token: object
) -> None:
    h = search_case.harness
    with pytest.raises(RemoteError) as caught:
        h.request(h.client(), method, token=token)
    assert caught.value.code is ErrorCode.INVALID_PARAMS
    assert h.ctrl.search.current is None


@pytest.mark.parametrize("timeout", [-0.01, 30.01, float("nan"), float("inf")])
def test_wait_range_admission_preserves_pending_search(
    search_case: SearchCase, timeout: float
) -> None:
    h = search_case.harness
    client = h.client()
    token = _start(search_case, client)
    reply = h.request(client, "operation.await", token=token, timeout=timeout)
    assert not reply["ok"] and "error" in reply
    assert reply["error"]["code"] == "invalid_params"
    assert h.ctrl.search.active_token == token
    assert h.ctrl.search.outcome(token) is None


@pytest.mark.parametrize("timeout", [True, "1", 10**400])
def test_wait_json_admission_is_shared(
    search_case: SearchCase, timeout: object
) -> None:
    h = search_case.harness
    with pytest.raises(RemoteError) as caught:
        h.request(h.client(), "operation.await", token=1, timeout=timeout)
    assert caught.value.code is ErrorCode.INVALID_PARAMS


@pytest.mark.parametrize("timeout", [None, 0, 30])
def test_wait_projects_native_user_feedback_without_state_observations(
    search_case: SearchCase, monkeypatch: pytest.MonkeyPatch, timeout: float | None
) -> None:
    # Search currently has no GUI feedback producer. Replace only the public
    # wait collaborator to cover this shared AwaitResult projection, not lifecycle.
    h = search_case.harness
    client = h.client()
    token = _start(search_case, client)
    fresh = h.client()
    calls: list[tuple[int, float]] = []

    def feedback(
        self: FluxDepSearchOwner, supplied: int, seconds: float
    ) -> AwaitResult:
        assert not h.owner.is_owner_thread()
        assert self.outcome(supplied) is None
        calls.append((supplied, seconds))
        return AwaitResult("user_feedback", feedback="請先檢查輸入")

    monkeypatch.setattr(FluxDepSearchOwner, "await_outcome", feedback)
    assert _result(
        h.request(fresh, "operation.await", token=token, timeout=timeout)
    ) == {
        "token": token,
        "reason": "user_feedback",
        "outcome": None,
        "feedback": "請先檢查輸入",
    }
    assert calls == [(token, 10.0 if timeout is None else timeout)]
    _error(h.request(fresh, "fit.search"), "precondition_failed", "stale_version")
    assert h.ctrl.search.active_token == token


@pytest.mark.parametrize(
    "reason",
    ["no_database_path", "no_selected_points", "selection_stale", "search_closing"],
)
def test_native_search_admission_errors_do_not_open_handles(
    search_case: SearchCase, reason: str
) -> None:
    h = search_case.harness
    if reason == "no_database_path":
        h.ctrl.set_fit_params("", (2, 15), (0.2, 2), (0.1, 2), {}, None, None)
    elif reason == "no_selected_points":
        h.ctrl.set_selection(np.zeros(4, dtype=np.bool_))
    elif reason == "selection_stale":
        h.ctrl.state.set_selection(np.ones(1, dtype=np.bool_))
    else:
        h.ctrl.search.begin_close()
    client = h.client()
    _observe(h, client)
    _error(h.request(client, "fit.search"), "precondition_failed", reason)
    assert h.ctrl.search.current is None
    assert not search_case.entered.is_set()


def test_evicted_token_is_rejected_instead_of_fabricating_completion(
    search_case: SearchCase,
) -> None:
    h = search_case.harness
    client = h.client()
    search_case.release.set()
    first: int | None = None
    # Exercise retention through public searches rather than changing the
    # handle registry or setting its private retention policy in the test.
    for _ in range(40):
        token = h.ctrl.search.start()
        if first is None:
            first = token
        assert _result(h.request(client, "operation.await", token=token))[
            "outcome"
        ] == {"status": "finished", "error": None}
    for method in ["operation.status", "operation.cancel", "operation.await"]:
        _error(
            h.request(client, method, token=first),
            "invalid_params",
            "unknown_operation",
        )
    latest = h.ctrl.search.current
    assert latest is not None
    assert _result(h.request(client, "operation.status"))["activity"] == {
        "token": latest.token,
        "status": "finished",
        "error": None,
    }


def test_missing_search_runtime_stays_unexpected(route_harness: RouteHarness) -> None:
    client = route_harness.client()
    for method in ["project.info", "fit.result", "selection.snapshot", "spectrum.list"]:
        assert route_harness.request(client, method)["ok"]
    reply = route_harness.request(client, "fit.search")
    assert not reply["ok"] and "error" in reply
    assert reply["error"]["code"] == "controller_error"
    assert route_harness.ctrl.search.current is None
