"""Numeric input persistence at the public evolution boundary."""

import gzip
import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from zcu_tools.simulate import evolution

from tests.simulate.evolution.contract.accuracy_cases import accuracy_case


def test_manifest_round_trip_is_deterministic_and_rejects_invalid_identity(
    tmp_path: Path,
) -> None:
    assert callable(getattr(evolution, "write_frozen_manifest", None)), (
        "missing public frozen manifest codec"
    )
    case = accuracy_case("idle_coherence")
    with closing(case.source):
        physical = evolution.physical_input(case.source, case.batch)
        oracle_hash = evolution.digest(
            evolution.physical_input(case.source, case.batch, include_identity=False)
        )
    metadata = {"opaque": [None, True, 3, "caller-owned"], "category": "arbitrary"}
    records = (
        evolution.FrozenRecord("second", physical, metadata),
        evolution.FrozenRecord("first", physical),
    )
    path, other = tmp_path / "inputs.gz", tmp_path / "different-name.gz"
    evolution.write_frozen_manifest(path, records)
    evolution.write_frozen_manifest(other, records)
    assert path.read_bytes() == other.read_bytes()
    restored = evolution.read_frozen_manifest(path)
    assert tuple(record.name for record in restored) == ("second", "first")
    assert restored[0].metadata == metadata
    assert restored[1].metadata is None
    assert evolution.canonical_json(restored[0].physical) == evolution.canonical_json(
        physical
    )
    assert restored[0].physical_sha256 == evolution.digest(physical)
    assert restored[0].oracle_sha256 == oracle_hash
    with pytest.raises(ValueError, match="duplicate"):
        evolution.write_frozen_manifest(other, (records[0], records[0]))
    original = gzip.decompress(path.read_bytes())
    for field in ("physical_sha256", "oracle_sha256"):
        payload = json.loads(original)
        payload["inputs"][0][field] = "0" * 64
        other.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
        with pytest.raises(ValueError, match="hash"):
            evolution.read_frozen_manifest(other)
    payload = json.loads(original)
    payload["inputs"].append(payload["inputs"][0])
    other.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
    with pytest.raises(ValueError, match="duplicate"):
        evolution.read_frozen_manifest(other)


def test_index_cache_round_trip_preserves_order_and_metadata(tmp_path: Path) -> None:
    write_record = getattr(evolution, "write_scene_record", None)
    assert callable(write_record), "missing public single-record scene cache writer"
    case = accuracy_case("idle_coherence")
    with closing(case.source):
        physical = evolution.physical_input(case.source, case.batch)
    records = (
        evolution.FrozenRecord("second", physical, {"opaque": 2}),
        evolution.FrozenRecord("first", physical, None),
    )
    cache, index = tmp_path / "cache", tmp_path / "index.json"
    entries = tuple(evolution.write_scene_record(cache, record) for record in records)
    evolution.write_frozen_index(index, entries)
    restored = evolution.read_scene_records(cache, evolution.read_frozen_index(index))
    assert [record.name for record in restored] == ["second", "first"]
    assert [record.metadata for record in restored] == [{"opaque": 2}, None]
    assert [record.physical_sha256 for record in restored] == [
        record.physical_sha256 for record in records
    ]
    assert entries[0].metadata_sha256 == evolution.digest({"opaque": 2})


def test_scene_index_rejects_version_missing_and_mismatched_records(
    tmp_path: Path,
) -> None:
    case = accuracy_case("idle_coherence")
    with closing(case.source):
        record = evolution.FrozenRecord(
            "pinned",
            evolution.physical_input(case.source, case.batch),
            {"opaque": True},
        )
    entry = evolution.FrozenIndexEntry.from_record(record)
    cache, index = tmp_path / "cache", tmp_path / "index.json"
    evolution.write_frozen_index(index, (entry,))
    payload = json.loads(index.read_text())
    payload["version"] = 99
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="version"):
        evolution.read_frozen_index(index)
    with pytest.raises(FileNotFoundError, match=f"pinned.*{entry.physical_sha256}"):
        evolution.read_scene_records(cache, (entry,))
    evolution.write_scene_record(cache, record)
    wrong = replace(entry, oracle_sha256="0" * 64)
    with pytest.raises(ValueError, match="hash/name mismatch.*pinned"):
        evolution.read_scene_records(cache, (wrong,))
    # The filename identity cannot hide a modified opaque metadata payload.
    evolution.write_frozen_manifest(
        entry.cache_path(cache), (replace(record, metadata={"opaque": False}),)
    )
    with pytest.raises(ValueError, match="hash/name mismatch.*pinned"):
        evolution.read_scene_records(cache, (entry,))
    evolution.write_frozen_manifest(
        entry.cache_path(cache), (record, replace(record, name="extra"))
    )
    with pytest.raises(ValueError, match="hash/name mismatch.*pinned"):
        evolution.read_scene_records(cache, (entry,))


def test_atomic_scene_publication_preserves_pins_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    case = accuracy_case("idle_coherence")
    with closing(case.source):
        record = evolution.FrozenRecord(
            "pinned", evolution.physical_input(case.source, case.batch)
        )
    entry = evolution.write_scene_record(tmp_path / "cache", record)
    index = tmp_path / "index.json"
    evolution.write_frozen_index(index, (entry,))
    changed = replace(entry, metadata_sha256="0" * 64)

    def fail_replace(source: object, destination: object) -> None:
        # Atomic replace is the filesystem boundary, not codec internals.
        assert evolution.read_frozen_index(index) == (entry,)
        raise OSError("injected publication failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        evolution.write_frozen_index(index, (changed,))
    assert evolution.read_frozen_index(index) == (entry,)
    with pytest.raises(OSError, match="publication failure"):
        evolution.write_scene_record(
            tmp_path / "cache", replace(record, metadata="new")
        )
    assert (
        evolution.read_scene_records(tmp_path / "cache", (entry,))[0].name == "pinned"
    )
    with pytest.raises(FileNotFoundError):
        evolution.read_scene_records(
            tmp_path / "cache",
            (evolution.FrozenIndexEntry.from_record(replace(record, metadata="new")),),
        )


@pytest.mark.parametrize("loops", [1, 2])
def test_zero_point_round_trip_preserves_empty_array_dtypes_and_loop_shape(
    tmp_path: Path, loops: int
) -> None:
    case = accuracy_case("idle_coherence")
    batch = evolution.TimelineBatch(
        np.empty(0, dtype=np.int64),
        (),
        evolution.FluxSchedule(
            np.array([0], dtype=np.int64),
            np.empty(0),
            np.empty((0, loops), dtype=np.int64),
        ),
        np.empty(0),
    )
    with closing(case.source):
        path = tmp_path / "empty.json"
        evolution.write_frozen_input(path, case.source, batch)
        restored = evolution.read_frozen_input(path)
        with closing(restored.source):
            assert_array_equal(restored.batch.points, batch.points)
            assert restored.batch.points.dtype == np.int64
            assert_array_equal(restored.batch.flux.offsets, batch.flux.offsets)
            assert restored.batch.flux.offsets.dtype == np.int64
            assert_array_equal(restored.batch.flux.start_ns, batch.flux.start_ns)
            assert restored.batch.flux.start_ns.dtype == np.float64
            assert restored.batch.flux.level.shape == (0, loops)
            assert restored.batch.flux.level.dtype == np.int64
            assert_array_equal(restored.batch.end_ns, batch.end_ns)
            assert restored.batch.end_ns.dtype == np.float64
            assert evolution.canonical_json(
                evolution.physical_input(restored.source, restored.batch)
            ) == evolution.canonical_json(evolution.physical_input(case.source, batch))


def test_numeric_round_trip_preserves_models_tracks_flux_and_point_order(
    tmp_path: Path,
) -> None:
    write = getattr(evolution, "write_frozen_input", None)
    read = getattr(evolution, "read_frozen_input", None)
    assert callable(write), "missing public numeric input writer"
    assert callable(read), "missing public numeric input reader"
    case = accuracy_case("continuous_flux")
    batch = replace(case.batch, points=np.array([17, 3]))
    with closing(case.source):
        path = tmp_path / "input.json"
        evolution.write_frozen_input(path, case.source, batch)
        restored = evolution.read_frozen_input(path)
        with closing(restored.source):
            assert_array_equal(restored.batch.points, [17, 3])
            assert restored.batch.points.dtype == np.int64
            assert_array_equal(
                restored.source.initial_density, case.source.initial_density
            )
            assert restored.source.initial_density.dtype == np.complex128
            assert_array_equal(restored.batch.flux.level, batch.flux.level)
            assert restored.batch.flux.level.dtype == np.int64
            assert_array_equal(restored.batch.end_ns, batch.end_ns)
            assert restored.batch.end_ns.dtype == np.float64
            assert_array_equal(restored.batch.tracks[0].active, batch.tracks[0].active)
            assert restored.batch.tracks[0].active.dtype == np.bool_
            assert_array_equal(
                restored.batch.tracks[0].envelopes[0].samples,
                batch.tracks[0].envelopes[0].samples,
            )
            assert_array_equal(
                restored.source.model((0,)).drives[0].in_phase_ghz,
                case.source.model((0,)).drives[0].in_phase_ghz,
            )
            assert_array_equal(
                restored.source.flux_expansion().static,
                case.source.flux_expansion().static,
            )
            assert (
                restored.source.slow_envelope_segments_per_period
                == case.source.slow_envelope_segments_per_period
            )
            assert restored.batch.flux_waveform is not None
            assert batch.flux_waveform is not None
            assert_array_equal(
                restored.batch.flux_waveform.flux, batch.flux_waveform.flux
            )
            assert evolution.canonical_json(
                evolution.physical_input(restored.source, restored.batch)
            ) == evolution.canonical_json(evolution.physical_input(case.source, batch))
