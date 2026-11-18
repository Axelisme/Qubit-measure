"""Public capture metadata reader validates JSON shape without model construction."""

import json
from collections.abc import Sequence

import pytest
from pydantic import ValidationError
from zcu_tools.simulate.evolution import (
    CaptureMetadata,
    FrozenRecord,
    canonical_json,
    read_capture_metadata,
)


@pytest.fixture
def metadata() -> CaptureMetadata:
    return {
        "capture_id": "evolve-a",
        "batch_index": 0,
        "batch_count": 2,
        "budget": 1e-6,
        "families": ["direct"],
        "source_identity": "synthetic",
        "segment_numeric_version": "v-test",
        "frozen_physical_version": 1,
        "complete": True,
        "routes": [
            {
                "point": 4,
                "segment": 1,
                "start_ns": 0.0,
                "duration_ns": 20.0,
                "round": "caller",
                "prefix_run": 0,
                "family": "direct",
                "candidates": [{"family": "direct", "guard": "accepted"}],
            }
        ],
        "sampled_rwa_admissions": [],
        "prefix_runs": [
            {
                "representative": {"point": 4, "segment": 2},
                "adopters": [{"batch": 1, "point": 5, "segment": 2}],
            }
        ],
    }


def test_reader_returns_complete_typed_metadata_without_reconstructing_physics(
    metadata: CaptureMetadata,
) -> None:
    record = FrozenRecord("capture", {}, metadata)
    assert read_capture_metadata(record) == metadata


@pytest.mark.parametrize("value", [None, [], "metadata", 3])
def test_reader_rejects_non_mapping_metadata(value: object) -> None:
    with pytest.raises(ValidationError) as error:
        read_capture_metadata(FrozenRecord("capture", {}, value))
    assert error.value.errors()[0]["loc"] == ()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("complete",), 1),
        (("families",), "direct"),
        (("budget",), "0.1"),
        (("routes", 0, "round"), "later"),
        (("routes", 0, "candidates", 0, "guard"), "untested"),
        (("routes", 0, "candidates", 0, "family"), 5),
        (("prefix_runs", 0, "representative", "point"), "4"),
        (("prefix_runs", 0, "adopters", 0, "batch"), False),
        (("extra",), 1),
        (("routes", 0, "extra"), 1),
        (("routes", 0, "candidates", 0, "extra"), 1),
        (("prefix_runs", 0, "extra"), 1),
        (("prefix_runs", 0, "representative", "extra"), 1),
        (("prefix_runs", 0, "adopters", 0, "extra"), 1),
    ],
)
def test_reader_rejects_wrong_types_literals_and_extra_keys_at_their_field_path(
    metadata: CaptureMetadata,
    path: Sequence[str | int],
    value: object,
) -> None:
    payload = json.loads(canonical_json(metadata))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError) as error:
        read_capture_metadata(FrozenRecord("capture", {}, payload))
    assert error.value.errors()[0]["loc"] == tuple(path)


@pytest.mark.parametrize("path", [("complete",), ("routes", 0, "family")])
def test_reader_rejects_missing_required_keys(
    metadata: CaptureMetadata,
    path: Sequence[str | int],
) -> None:
    payload = json.loads(canonical_json(metadata))
    target = payload
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    with pytest.raises(ValidationError) as error:
        read_capture_metadata(FrozenRecord("capture", {}, payload))
    assert error.value.errors()[0]["loc"] == tuple(path)
