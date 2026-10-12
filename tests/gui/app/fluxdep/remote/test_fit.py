"""Fit replacement and observation through the native RPC route."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image
from zcu_tools.analysis.fluxdep.models import PointsData
from zcu_tools.analysis.fluxdep.search import (
    DatabaseSearchResult,
    ParamBounds,
    SearchQuality,
    TransitionResidual,
)
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload
from zcu_tools.gui.app.fluxdep.state import AlignmentPublication, SpectrumEntry
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


def _png_data(figure: object) -> bytes:
    assert isinstance(figure, dict)
    encoded = figure["png_b64"]
    assert isinstance(encoded, str)
    data = base64.b64decode(encoded, validate=True)
    assert figure["bytes"] == len(data)
    return data


def _source(name: str) -> SpectrumEntry:
    axis = np.array([0.0, 0.5, 1.0])
    return SpectrumEntry(
        name=name,
        spec_type="OneTone",
        raw=SpectrumData(
            dev_values=axis.copy(),
            fluxs=axis.copy(),
            freqs=axis.copy(),
            signals=np.ones((3, 3), dtype=np.complex128),
        ),
        points=PointsData(
            dev_values=np.empty(0),
            fluxs=np.empty(0),
            freqs=np.empty(0),
        ),
        aligned=True,
        points_completed=True,
    )


@pytest.mark.parametrize(
    "change, reason",
    [
        ("align", "empty re-aligned"),
        ("points", "points of empty changed"),
        ("filter", "filter changed"),
        ("add", "spectrum collection changed"),
        ("remove", "spectrum collection changed"),
        ("replace", "empty re-aligned"),
    ],
)
def test_upstream_publication_retains_fit_marks_stale_and_refuses_export(
    route_harness: RouteHarness,
    tmp_path: Path,
    change: str,
    reason: str,
) -> None:
    state = route_harness.ctrl.state
    state.put_spectrum(_source("empty"))
    state.set_fit_result((3.0, 0.8, 0.5))
    match change:
        case "align":
            state.set_alignment(
                "empty",
                AlignmentPublication(0.1, 0.6, 1.0),
                np.array([0.1, 0.6, 1.1]),
                np.empty(0),
            )
        case "points":
            state.set_points(
                "empty",
                PointsData(
                    dev_values=np.empty(0), fluxs=np.empty(0), freqs=np.empty(0)
                ),
            )
        case "filter":
            state.set_selection(np.empty(0, dtype=np.bool_))
        case "add":
            state.put_spectrum(_source("new"))
        case "remove":
            state.remove_spectrum("empty")
        case "replace":
            state.put_spectrum(_source("empty"))
    client = route_harness.client()
    reply = route_harness.request(client, "fit.result")
    assert reply["ok"] and "result" in reply
    assert reply["result"].get("stale") is True
    assert reason in reply["result"].get("stale_reasons", [])
    assert reply["result"]["params"] == {"EJ": 3.0, "EC": 0.8, "EL": 0.5}
    path = tmp_path / "stale.json"
    export = route_harness.request(client, "fit.export_params", savepath=str(path))
    assert not export["ok"] and "error" in export
    assert export["error"]["reason"] == "fit_stale"
    assert reason in export["error"]["message"]
    assert not path.exists()


def test_active_is_display_only_and_replacement_clears_stale_result(
    route_harness: RouteHarness,
) -> None:
    state = route_harness.ctrl.state
    state.put_spectrum(_source("empty"))
    state.set_fit_result((3.0, 0.8, 0.5))
    state.set_active("empty")
    client = route_harness.client()
    reply = route_harness.request(client, "fit.result")
    assert reply["ok"] and "result" in reply
    assert reply["result"].get("stale", "missing") is False
    assert reply["result"].get("stale_reasons") == []
    state.set_selection(np.empty(0, dtype=np.bool_))
    state.set_fit_params("", (2, 15), (0.2, 2), (0.1, 2), {}, None, None)
    cleared = route_harness.request(client, "fit.result")
    assert cleared["ok"] and "result" in cleared
    assert cleared["result"]["params"] is None
    assert cleared["result"].get("stale", "missing") is False
    assert cleared["result"].get("stale_reasons") == []
    assert cleared["result"].get("quality", "missing") is None


def test_fit_result_projects_recorded_quality_without_rounding_or_merging(
    route_harness: RouteHarness,
) -> None:
    ctrl = route_harness.ctrl
    client = route_harness.client()
    initial = route_harness.request(client, "fit.result")
    assert initial["ok"] and "result" in initial
    assert initial["result"].get("quality", "missing") is None
    quality = SearchQuality(
        mean=0.123456789,
        outliers=1,
        outlier_threshold=0.370370367,
        per_transition=(
            TransitionResidual("transitions 0-1", 2, 0.123456789, 0.4),
            TransitionResidual("transitions 0-1", 1, 0.025678912, 0.025678912),
            TransitionResidual("mirror 0-2", 0, None, None),
        ),
    )
    ctrl.record_search_result(
        DatabaseSearchResult(
            params=(3.0, 0.8, 0.5),
            best_distance=quality.mean,
            best_scale=1.0,
            best_index=0,
            entry_results=np.array([[quality.mean, 1.0]]),
            entry_params=np.array([[3.0, 0.8, 0.5]]),
            fluxs=np.array([0.1, 0.2, 0.3]),
            freqs=np.array([4.0, 4.1, 4.2]),
            predicted_freqs=np.array([4.0, 4.1, 4.2]),
            bounds=ParamBounds(EJ=(2, 15), EC=(0.2, 2), EL=(0.1, 2)),
            transition_indices=np.array([0, 0, 1], dtype=np.int64),
            residuals=np.array([0.1, 0.4, 0.025678912]),
            outlier_mask=np.array([False, True, False]),
            quality=quality,
        )
    )
    reply = route_harness.request(client, "fit.result")
    assert reply["ok"] and "result" in reply
    assert reply["result"].get("quality") == {
        "mean": 0.123456789,
        "outliers": 1,
        "outlier_threshold": 0.370370367,
        "per_transition": [
            {
                "transition": "transitions 0-1",
                "points": 2,
                "mean": 0.123456789,
                "max": 0.4,
            },
            {
                "transition": "transitions 0-1",
                "points": 1,
                "mean": 0.025678912,
                "max": 0.025678912,
            },
            {"transition": "mirror 0-2", "points": 0, "mean": None, "max": None},
        ],
    }


def test_fit_render_returns_png_at_shared_geometry(
    route_harness: RouteHarness,
) -> None:
    route_harness.ctrl.state.set_fit_params(
        "", (2, 15), (0.2, 2), (0.1, 2), {"transitions": [(0, 1)]}, None, None
    )
    route_harness.ctrl.state.set_fit_result((3.0, 0.8, 0.5))
    reply = route_harness.request(route_harness.client(), "fit.render")
    assert reply["ok"]
    assert "result" in reply
    data = _png_data(reply["result"]["figure"])
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "PNG"
        assert image.size == (640, 480)


def test_fit_render_without_result_is_a_precondition_failure(
    route_harness: RouteHarness,
) -> None:
    reply = route_harness.request(route_harness.client(), "fit.render")
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == "precondition_failed"


def test_database_info_reports_native_ranges_without_changing_fit(
    route_harness: RouteHarness,
    search_db: str,
) -> None:
    initial = route_harness.ctrl.state.fit
    reply = route_harness.request(
        route_harness.client(), "fit.database_info", path=search_db
    )
    assert reply["ok"]
    assert "result" in reply
    assert reply["result"] == {
        "path": search_db,
        "entries": 2,
        "levels": 4,
        "EJ": [3.0, 9.5],
        "EC": [0.2, 2.0],
        "EL": [0.1, 1.5],
        "flux": [0.0, 1.0],
    }
    assert route_harness.ctrl.state.fit is initial


def _inputs(db_path: str, **overrides: object) -> dict[str, object]:
    inputs: dict[str, object] = {
        "search_db": db_path,
        "EJb": [3, 9.5],
        "ECb": [0.2, 2],
        "ELb": [0.1, 1.5],
        "transitions": {"transitions": [[0, 1], [1, 2]], "mirror2": []},
    }
    inputs.update(overrides)
    return inputs


def test_fit_replacement_clears_result_emits_agent_fact_and_does_not_start_search(
    route_harness: RouteHarness,
    search_db: str,
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
            search_db,
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
    route_harness: RouteHarness,
    search_db: str,
    overrides: dict[str, object],
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "fit.result")["ok"]
    initial = route_harness.ctrl.state.fit
    reply = route_harness.request(
        client, "fit.set_params", **_inputs(search_db, **overrides)
    )
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
    route_harness: RouteHarness,
    search_db: str,
    overrides: dict[str, object],
) -> None:
    client = route_harness.client()
    assert route_harness.request(client, "fit.result")["ok"]
    initial = route_harness.ctrl.state.fit
    with pytest.raises(RemoteError) as caught:
        route_harness.request(
            client, "fit.set_params", **_inputs(search_db, **overrides)
        )
    assert caught.value.code is ErrorCode.INVALID_PARAMS
    assert route_harness.ctrl.state.fit is initial
    assert route_harness.ctrl.state.version.get("fit") == 0


@pytest.mark.parametrize(
    "transitions",
    [
        {"transitons": []},
        {"transitions1": []},
        {"mirror01": []},
        {"transitions": [[-1, 1]]},
        {"transitions": [[0, 4]]},
        {"mirror4": []},
        {"transitions4": [[0, 1]]},
    ],
)
def test_transition_rejection_does_not_publish(
    route_harness: RouteHarness,
    search_db: str,
    transitions: dict[str, object],
) -> None:
    client = route_harness.client()
    initial = route_harness.ctrl.state.fit
    reply = route_harness.request(
        client,
        "fit.set_params",
        database_path=search_db,
        **_inputs(search_db, transitions=transitions),
    )
    assert not reply["ok"]
    assert "error" in reply
    assert reply["error"]["code"] == "invalid_params"
    assert route_harness.ctrl.state.fit is initial
    assert route_harness.ctrl.state.version.get("fit") == 0


@pytest.mark.parametrize(
    "transitions,code",
    [
        ({}, None),
        ({"transitions": [], "mirror": []}, None),
        ({"transitons": []}, "invalid_params"),
        ({"2": []}, "invalid_params"),
        ({"transitions2": []}, "controller_error"),
        ({"transitions": [[0, 1]]}, "controller_error"),
    ],
)
def test_database_is_read_only_when_validation_needs_levels(
    route_harness: RouteHarness,
    tmp_path: Path,
    transitions: dict[str, object],
    code: str | None,
) -> None:
    initial = route_harness.ctrl.state.fit
    reply = route_harness.request(
        route_harness.client(),
        "fit.set_params",
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
    route_harness: RouteHarness,
    search_db: str,
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
