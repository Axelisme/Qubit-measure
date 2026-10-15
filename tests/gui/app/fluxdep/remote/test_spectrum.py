"""Literal spectrum snapshots, guarded native commands and resource retirement."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.gui.app.fluxdep.state import SpectrumEntry
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.rpc_endpoint import ClientLink

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply


def _spectrum(name: str) -> SpectrumEntry:
    # Descending axes and annotations distinguish extents from endpoint guesses
    # and published point order from adapter-side sorting.
    return SpectrumEntry(
        name=name,
        spec_type="TwoTone",
        raw={
            "dev_values": np.array([2.0, 1.0, 0.0]),
            "fluxs": np.array([0.8, 0.3, -0.2]),
            "freqs": np.array([5.0, 4.0]),
            "signals": np.ones((3, 2), dtype=np.complex128),
        },
        points={
            "dev_values": np.array([1.6, 0.4]),
            "fluxs": np.array([0.6, 0.0]),
            "freqs": np.array([4.8, 4.2]),
        },
        flux_half=1.4,
        flux_int=2.4,
        flux_period=2.0,
        aligned=True,
        alignment_seeded=True,
        points_completed=True,
    )


@pytest.mark.parametrize(
    ("aligned", "completed", "empty", "draft", "stage", "pending"),
    [
        (False, False, True, None, "loaded", None),
        (False, True, False, "line", "loaded", "align"),
        (True, False, True, "sample", "aligned", "sample"),
        (True, True, True, None, "sampled", None),
        (True, True, False, "line", "sampled", "align"),
        (True, True, False, "sample", "sampled", "sample"),
    ],
)
def test_list_reports_published_stage_independently_of_drafts(
    route_harness: RouteHarness,
    aligned: bool,
    completed: bool,
    empty: bool,
    draft: str | None,
    stage: str,
    pending: str | None,
) -> None:
    ctrl = route_harness.ctrl
    axis = np.linspace(0.0, 3.0, 5)
    entry = replace(
        _spectrum("one"),
        aligned=aligned,
        points_completed=completed,
        raw={
            "dev_values": axis.copy(),
            "fluxs": axis.copy(),
            "freqs": np.linspace(4.0, 5.0, 5),
            "signals": np.ones((5, 5), dtype=np.complex128),
        },
    )
    if empty:
        entry.points = {
            "dev_values": entry.points["dev_values"][:0],
            "fluxs": entry.points["fluxs"][:0],
            "freqs": entry.points["freqs"][:0],
        }
    ctrl.state.put_spectrum(entry)
    owner = route_harness.client()
    if draft is not None:
        assert route_harness.request(
            owner, "spectrum.interactive.open", name="one", kind=draft
        )["ok"]
    observer = route_harness.client(mode="readonly")
    before = ctrl.state.version.snapshot()
    active = ctrl.state.active_spectrum
    context = ctrl.interactive.inspect("one")
    for _ in range(2):
        reply = route_harness.request(observer, "spectrum.list")
        assert reply["ok"] is True, reply
        assert reply["result"]["spectrums"] == [
            {
                "name": "one",
                "spec_type": "TwoTone",
                "aligned": aligned,
                "points_completed": completed,
                "point_count": 0 if empty else 2,
                "stage": stage,
                "filter_excluded_count": 0,
                "pending": pending,
            }
        ]
        assert ctrl.interactive.inspect("one") is context
        assert ctrl.state.active_spectrum == active
        assert ctrl.state.version.snapshot() == before


@pytest.mark.parametrize("published", [False, True])
def test_list_counts_only_published_per_spectrum_filter_masks(
    joint_route: RouteHarness, published: bool
) -> None:
    ctrl = joint_route.ctrl
    ctrl.state.put_spectrum(_spectrum("c"))
    ctrl.set_points("a", np.array([0.0, 0.5, 0.01]), np.array([4.0, 4.5, 4.01]))
    if published:
        ctrl.set_selection(np.array([True, False, True, False, False, False]), 0.1)
    # Filter opens all-selected, not from the published per-spectrum masks.
    owner = joint_route.client()
    opened = joint_route.request(owner, "selection.interactive.open")
    assert opened["ok"] is True, opened
    context = opened["result"]["context"]
    assert isinstance(context, dict)
    edited = joint_route.request(
        owner,
        "selection.interactive.command",
        context_id=context["context_id"],
        command="stroke",
        params={"vertices": [[0.0, 4.0]], "width": 0.03, "mode": "erase"},
    )
    assert edited["ok"], edited
    observer = joint_route.client(mode="readonly")
    before = ctrl.state.version.snapshot()
    sources = copy.deepcopy(ctrl.state.spectrums)
    draft = ctrl.interactive.inspect_selection()
    selection = joint_route.request(observer, "selection.snapshot")
    assert selection["ok"] is True, selection
    assert selection["result"]["pending"] is True
    for _ in range(2):
        reply = joint_route.request(observer, "spectrum.list")
        assert reply["ok"] is True, reply
        rows = reply["result"]["spectrums"]
        assert isinstance(rows, list)
        assert [
            (row["name"], row["point_count"], row.get("filter_excluded_count"))
            for row in rows
        ] == [
            ("a", 3, 1 if published else 0),
            ("empty", 0, 0),
            ("b", 1, 1 if published else 0),
            ("c", 2, 2 if published else 0),
        ]
        after_selection = joint_route.request(observer, "selection.snapshot")
        assert after_selection["ok"] is True, after_selection
        assert after_selection["result"] == selection["result"]
        assert ctrl.interactive.inspect_selection() is draft
        assert all(ctrl.interactive.inspect(name) is None for name in sources)
        assert ctrl.state.active_spectrum == "a"
        assert ctrl.state.version.snapshot() == before
        for name, source in sources.items():
            current = ctrl.state.get_spectrum(name)
            for key in ("dev_values", "fluxs", "freqs"):
                np.testing.assert_array_equal(current.points[key], source.points[key])
            np.testing.assert_array_equal(current.filter_mask, source.filter_mask)


def _assert_unknown_spectrum(reply: RouteReply) -> None:
    assert reply["ok"] is False
    assert reply["error"]["code"] == "invalid_params"
    assert reply["error"].get("reason") == "unknown_spectrum"


def _read_spectrum(
    harness: RouteHarness, client: ClientLink, name: str
) -> dict[str, object]:
    reply = harness.request(client, "spectrum.snapshot", name=name)
    assert reply["ok"] is True
    return reply["result"]


@pytest.mark.parametrize("name", ["測量:Q1", "name*", "測量:Q1:*"])
def test_snapshot_preserves_complete_native_state_for_literal_names(
    route_harness: RouteHarness,
    name: str,
) -> None:
    state = route_harness.ctrl.state
    state.put_spectrum(_spectrum(name))
    before = state.version.snapshot()
    result = _read_spectrum(route_harness, route_harness.client(), name)
    assert result == {
        "name": name,
        "exists": True,
        "spec_type": "TwoTone",
        "aligned": True,
        "points_completed": True,
        "alignment_seeded": True,
        "pending": None,
        "alignment": {"inherited_from": None},
        "flux_half": 1.4,
        "flux_int": 2.4,
        "flux_period": 2.0,
        "raw_axes": {
            "dev_values": {
                "count": 3,
                "minimum": 0.0,
                "maximum": 2.0,
                "unit": "native",
            },
            "fluxs": {"count": 3, "minimum": -0.2, "maximum": 0.8, "unit": "Phi_0"},
            "freqs": {"count": 2, "minimum": 4.0, "maximum": 5.0, "unit": "GHz"},
            "signals_shape": [3, 2],
        },
        "points": {"dev_values": [1.6, 0.4], "fluxs": [0.6, 0.0], "freqs": [4.8, 4.2]},
    }
    assert state.version.snapshot() == before


def test_empty_completed_spectrum_has_empty_axes_and_published_points(
    route_harness: RouteHarness,
) -> None:
    entry = _spectrum("empty")
    empty = np.empty(0, dtype=np.float64)
    entry = replace(
        entry,
        raw={
            "dev_values": empty.copy(),
            "fluxs": empty.copy(),
            "freqs": empty.copy(),
            "signals": np.empty((0, 0), dtype=np.complex128),
        },
        points={
            "dev_values": empty.copy(),
            "fluxs": empty.copy(),
            "freqs": empty.copy(),
        },
    )
    route_harness.ctrl.state.put_spectrum(entry)
    result = _read_spectrum(route_harness, route_harness.client(), "empty")
    assert result["points_completed"] is True
    assert result["points"] == {"dev_values": [], "fluxs": [], "freqs": []}
    assert result["raw_axes"] == {
        "dev_values": {"count": 0, "minimum": None, "maximum": None, "unit": "native"},
        "fluxs": {"count": 0, "minimum": None, "maximum": None, "unit": "Phi_0"},
        "freqs": {"count": 0, "minimum": None, "maximum": None, "unit": "GHz"},
        "signals_shape": [0, 0],
    }


def test_literal_star_guard_does_not_expand_to_a_same_prefix_neighbor(
    route_harness: RouteHarness,
) -> None:
    name = "測量:one*"
    neighbor = name + ":neighbor"
    state = route_harness.ctrl.state
    state.put_spectrum(_spectrum(name))
    state.put_spectrum(_spectrum(neighbor))
    client = route_harness.client()
    _read_spectrum(route_harness, client, name)
    # The neighbor remains unread and changes after this exact full read.
    route_harness.ctrl.reset_alignment(neighbor)
    for _ in range(2):
        reply = route_harness.request(client, "spectrum.reset_alignment", name=name)
        assert reply["ok"] is True
        assert reply["result"] == {"name": name}
    entry = state.get_spectrum(name)
    assert not entry.aligned
    assert entry.points_completed
    assert entry.alignment_seeded
    assert entry.flux_half == 1.4
    assert entry.flux_int == 2.4
    assert entry.flux_period == 2.0
    np.testing.assert_array_equal(entry.points["dev_values"], [1.6, 0.4])
    np.testing.assert_array_equal(entry.points["fluxs"], [0.6, 0.0])
    assert state.version.get(f"spectrum:{name}") == 3
    assert state.version.get(f"spectrum:{neighbor}") == 2


def test_reset_points_keeps_alignment_and_unaligned_error_does_not_publish(
    route_harness: RouteHarness,
) -> None:
    state = route_harness.ctrl.state
    state.put_spectrum(_spectrum("one"))
    client = route_harness.client()
    _read_spectrum(route_harness, client, "one")
    reply = route_harness.request(client, "spectrum.reset_points", name="one")
    assert reply["ok"] is True
    assert reply["result"] == {"name": "one"}
    result = _read_spectrum(route_harness, client, "one")
    assert result["aligned"] is True
    assert result["points_completed"] is False
    assert result["points"] == {"dev_values": [], "fluxs": [], "freqs": []}
    assert result["flux_half"] == 1.4
    assert route_harness.request(client, "spectrum.reset_alignment", name="one")["ok"]
    before = state.version.snapshot()
    failed = route_harness.request(client, "spectrum.reset_points", name="one")
    assert failed["ok"] is False
    assert failed["error"]["code"] == "precondition_failed"
    assert failed["error"].get("reason") == "spectrum_not_aligned"
    assert state.version.snapshot() == before


@pytest.mark.parametrize(
    "method",
    ["spectrum.reset_alignment", "spectrum.reset_points", "spectrum.remove"],
)
def test_full_absence_read_admits_owner_lookup_but_is_not_existence_proof(
    route_harness: RouteHarness,
    method: str,
) -> None:
    client = route_harness.client()
    name = "不存在:*"
    if method == "spectrum.remove":
        assert route_harness.request(client, "spectrum.list")["ok"]
    assert _read_spectrum(route_harness, client, name) == {
        "name": name,
        "exists": False,
    }
    before = route_harness.ctrl.state.version.snapshot()
    _assert_unknown_spectrum(route_harness.request(client, method, name=name))
    assert route_harness.ctrl.state.version.snapshot() == before


def test_remove_requires_the_collection_and_exact_leaf_reads(
    route_harness: RouteHarness,
) -> None:
    state = route_harness.ctrl.state
    state.put_spectrum(_spectrum("one"))
    client = route_harness.client()
    _read_spectrum(route_harness, client, "one")
    assert route_harness.request(client, "spectrum.list")["ok"]
    state.put_spectrum(_spectrum("another"))
    assert state.get_spectrum("one").aligned
    assert route_harness.request(client, "spectrum.list")["ok"]
    assert route_harness.request(client, "spectrum.remove", name="one")["ok"]
    _assert_unknown_spectrum(
        route_harness.request(client, "spectrum.remove", name="one")
    )


def test_remove_and_same_name_recreate_retire_only_the_exact_leaf(
    route_harness: RouteHarness,
) -> None:
    name, neighbor = "測量:*", "測量:*:neighbor"
    state = route_harness.ctrl.state
    original = _spectrum(name)
    state.put_spectrum(original)
    state.put_spectrum(_spectrum(neighbor))
    route_harness.ctrl.set_active_spectrum(name)
    client = route_harness.client()
    assert route_harness.request(client, "spectrum.list")["ok"]
    _read_spectrum(route_harness, client, name)
    _read_spectrum(route_harness, client, neighbor)
    removed = route_harness.request(client, "spectrum.remove", name=name)
    assert removed["ok"] is True
    assert removed["result"] == {"name": name, "removed": True}
    assert state.active_spectrum is None
    assert state.version.get(f"spectrum:{name}") == 0
    assert state.version.get(f"spectrum:{neighbor}") == 1
    assert _read_spectrum(route_harness, client, name) == {
        "name": name,
        "exists": False,
    }

    state.put_spectrum(original)
    assert state.version.get(f"spectrum:{name}") == 2
    assert state.get_spectrum(name).aligned
    assert route_harness.request(client, "spectrum.reset_alignment", name=neighbor)[
        "ok"
    ]
    _read_spectrum(route_harness, client, name)
    assert route_harness.request(client, "spectrum.reset_alignment", name=name)["ok"]


@pytest.mark.parametrize("params", [{}, {"name": None}])
def test_set_active_null_and_omission_clear_display_without_source_version_changes(
    route_harness: RouteHarness,
    params: dict[str, object],
) -> None:
    state = route_harness.ctrl.state
    state.put_spectrum(_spectrum("one"))
    client = route_harness.client()
    assert route_harness.request(client, "spectrum.list")["ok"]
    before = state.version.snapshot()
    chosen = route_harness.request(client, "spectrum.set_active", name="one")
    assert chosen["ok"] is True
    assert chosen["result"] == {"active_spectrum": "one"}
    assert state.active_spectrum == "one"
    _assert_unknown_spectrum(
        route_harness.request(client, "spectrum.set_active", name="missing")
    )
    assert state.active_spectrum == "one"
    cleared = route_harness.request(client, "spectrum.set_active", **params)
    assert cleared["ok"] is True
    assert cleared["result"] == {"active_spectrum": None}
    assert state.active_spectrum is None
    assert state.version.snapshot() == before


@pytest.mark.parametrize("name", ["", None, False, 1, []])
def test_malformed_snapshot_name_cannot_unlock_a_leaf(
    route_harness: RouteHarness,
    name: object,
) -> None:
    state = route_harness.ctrl.state
    state.put_spectrum(_spectrum("one"))
    client = route_harness.client()
    before = state.version.snapshot()
    with pytest.raises(RemoteError) as failure:
        route_harness.request(client, "spectrum.snapshot", name=name)
    assert failure.value.code is ErrorCode.INVALID_PARAMS
    assert state.version.snapshot() == before
