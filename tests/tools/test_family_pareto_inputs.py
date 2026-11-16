"""Frozen scene consumers decode tool declarations without replaying builders."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
from zcu_tools.simulate.evolution import (
    FrozenRecord,
    physical_input,
    write_frozen_manifest,
)

from tests.tools.family_pareto_support import make_scene
from tools import family_pareto_inputs


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
    write_frozen_manifest(paths[0], (first,))
    write_frozen_manifest(paths[1], (second,))

    items = family_pareto_inputs.read_inputs(paths)

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
