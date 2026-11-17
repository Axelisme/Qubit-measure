"""Frozen scene consumers decode tool declarations without replaying builders."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    FrozenIndexEntry,
    FrozenRecord,
    physical_input,
    write_frozen_index,
    write_scene_record,
)

from tests.tools.family_pareto_support import make_scene
from tools import family_pareto, family_pareto_inputs


@pytest.mark.parametrize("missing_default", [False, True])
def test_default_selection_checks_required_names_before_reading_only_pareto_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_default: bool
) -> None:
    scene = make_scene("idle_coherence")
    with closing(scene.source):
        physical = physical_input(scene.source, scene.batch)
    cache = tmp_path / "scenes"
    paths = (tmp_path / "first.index.json", tmp_path / "second.index.json")
    names = sorted(family_pareto.PARETO_DEFAULT_NAMES)
    records = [
        FrozenRecord(name, physical, {"reason": "fixture", "expected_families": ["t4"]})
        for name in names
    ]
    extra = FrozenIndexEntry.from_record(replace(records[0], name="other-workload"))
    if missing_default:
        # No payload exists: missing required names must fail at index selection.
        entries = [FrozenIndexEntry.from_record(record) for record in records[1:]]
    else:
        entries = [write_scene_record(cache, record) for record in records]
    write_frozen_index(paths[0], entries)
    write_frozen_index(paths[1], (extra,))
    # Default path discovery has no path override; substitute only that I/O boundary.
    monkeypatch.setattr(family_pareto, "scene_index_paths", lambda index: paths)

    if missing_default:
        with pytest.raises(ValueError, match=f"Missing Pareto default.*{names[0]}"):
            family_pareto.select_inputs(scene_cache=cache)
    else:
        selected = family_pareto.select_inputs(scene_cache=cache)
        assert [item.input.name for item in selected] == names
        assert [
            FrozenIndexEntry.from_record(item.record) for item in selected
        ] == entries


@pytest.mark.parametrize("explicit_index", [False, True])
def test_select_inputs_reads_one_frozen_scenario_without_producer_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_index: bool,
) -> None:
    scene = make_scene("idle_coherence")
    with closing(scene.source):
        physical = physical_input(scene.source, scene.batch)
    records = (
        FrozenRecord(
            "first", physical, {"reason": "first", "expected_families": ["t4"]}
        ),
        FrozenRecord(
            "second", physical, {"reason": "second", "expected_families": ["t4"]}
        ),
    )
    path = tmp_path / "scenes.json.gz"
    cache = tmp_path / "scenes"
    # The unselected first scene deliberately has no numeric cache file.
    write_frozen_index(
        path,
        (
            FrozenIndexEntry.from_record(records[0]),
            write_scene_record(cache, records[1]),
        ),
    )

    if not explicit_index:
        # Default discovery has no path argument; keep real index/record selection.
        monkeypatch.setattr(family_pareto, "scene_index_paths", lambda index: (path,))
    (selected,) = family_pareto.select_inputs(
        path if explicit_index else None, "second", scene_cache=cache
    )

    assert selected.input.name == "second"
    assert selected.record.name == "second"
    scene = selected.input.build()
    with closing(scene.source):
        np.testing.assert_array_equal(scene.batch.points, [0, 1])
        assert scene.families == ("t4",)


def test_explicit_index_selects_all_rows_outside_the_pareto_default(
    tmp_path: Path,
) -> None:
    scene = make_scene("idle_coherence")
    with closing(scene.source):
        physical = physical_input(scene.source, scene.batch)
    cache, path = tmp_path / "scenes", tmp_path / "inputs.index.json"
    records = [
        FrozenRecord(name, physical, {"reason": "fixture", "expected_families": ["t4"]})
        for name in ("first-workload", "second-workload")
    ]
    entries = [write_scene_record(cache, record) for record in records]
    write_frozen_index(path, entries)
    selected = family_pareto.select_inputs(path, scene_cache=cache)
    assert [FrozenIndexEntry.from_record(item.record) for item in selected] == entries


def test_read_inputs_combines_manifests_and_builds_fresh_owned_scenes(
    tmp_path: Path,
) -> None:
    scene = make_scene("idle_coherence")
    with closing(scene.source):
        first = FrozenRecord(
            "first",
            physical_input(scene.source, scene.batch),
            {"reason": "idle", "expected_families": ["t4"]},
        )
        second = FrozenRecord(
            "second",
            physical_input(scene.source, replace(scene.batch, points=np.array([7, 3]))),
            {
                "reason": "ordered fixture",
                "expected_families": ["dop853"],
                "category": "real",
                "companions": ["t4"],
                "companions_by_target": {"dop853": []},
                "expected_routes": {"dop853": {"t4": [2, 3.0]}},
                "metadata": {"axis": "ordered"},
                "manifest": {"fixture": "small"},
            },
        )
    paths = [tmp_path / "one.json.gz", tmp_path / "two.json.gz"]
    cache = tmp_path / "scenes"
    write_frozen_index(paths[0], (write_scene_record(cache, first),))
    write_frozen_index(paths[1], (write_scene_record(cache, second),))

    items = family_pareto_inputs.read_inputs(paths, cache)

    assert [item.name for item in items] == ["first", "second"]
    assert items[0].reason == "idle"
    assert items[0].category == "synthetic"
    assert items[1].category == "real"
    assert items[1].metadata == {"axis": "ordered"}
    assert items[1].manifest == {"fixture": "small"}
    assert items[1].expected_routes == {"dop853": {"t4": (2, 3.0)}}
    assert items[1].companions_for("other") == ("t4",)
    assert items[1].companions_for("dop853") == ()
    original, fresh, other = (items[index].build() for index in (0, 0, 1))
    with closing(original.source), closing(fresh.source), closing(other.source):
        assert original.name == "first"
        assert original.families == ("t4",)
        assert other.name == "second"
        assert other.families == ("dop853",)
        np.testing.assert_array_equal(original.batch.points, [0, 1])
        np.testing.assert_array_equal(other.batch.points, [7, 3])
        assert original.source is not fresh.source
        assert original.source.history is not fresh.source.history
