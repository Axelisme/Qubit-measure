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
