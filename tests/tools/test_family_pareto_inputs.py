"""Frozen scene consumers decode tool declarations without replaying builders."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
from zcu_tools.simulate.evolution import (
    FrozenIndexEntry,
    FrozenRecord,
    physical_input,
    write_frozen_index,
    write_scene_record,
)

from tests.tools.family_pareto_support import make_scene
from tools import family_pareto, family_pareto_inputs


def test_select_inputs_reads_one_frozen_scenario_without_producer_building(
    tmp_path: Path,
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

    (selected,) = family_pareto.select_inputs(path, "second", scene_cache=cache)

    assert selected.input.name == "second"
    assert selected.record.name == "second"
    scene = selected.input.build()
    with closing(scene.source):
        np.testing.assert_array_equal(scene.batch.points, [0, 1])
        assert scene.families == ("t4",)


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
