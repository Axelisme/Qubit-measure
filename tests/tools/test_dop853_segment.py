"""Public synchronous segment construction and reference identity contracts."""

from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    ReferenceResult,
    load_reference,
    reference_key,
    write_reference,
)

from tools.dop853_segment import Dop853Coordinates, dop853_segment_input


@pytest.mark.parametrize("envelope", ["gaussian", "sampled"])
def test_dop853_segment_repeats_one_synchronous_pair_on_each_timeline(
    envelope: str,
) -> None:
    coordinates = Dop853Coordinates.model_validate(
        {"envelope": envelope, "repeats": 2, "points": 3, "duration_ns": 12.0}
    )
    item = dop853_segment_input(coordinates)
    with closing((case := item.build()).source):
        assert len(case.batch.points) == 3
        np.testing.assert_array_equal(case.batch.end_ns, [24.0, 24.0, 24.0])
        assert len(case.batch.tracks) == 4
        for repeat in range(2):
            pump, stokes = case.batch.tracks[2 * repeat : 2 * repeat + 2]
            np.testing.assert_array_equal(pump.start_ns, [12.0 * repeat] * 3)
            np.testing.assert_array_equal(stokes.start_ns, pump.start_ns)
            np.testing.assert_array_equal(pump.duration_ns, [12.0] * 3)
            np.testing.assert_array_equal(stokes.duration_ns, pump.duration_ns)
            for track in (pump, stokes):
                shape = track.envelopes[0]
                assert len(shape.sample_ns) == coordinates.knots
                assert shape.sample_ns[0] == 0
                assert shape.sample_ns[-1] == 12.0
                assert np.ptp(shape.samples.real) > 0
        assert item.companions_for("dop853") == ()
        assert case.families == ("dop853",)


@pytest.mark.parametrize("levels", [3, 5])
def test_dop853_segment_rebuilds_model_and_initial_state_at_requested_dimension(
    levels: int,
) -> None:
    item = dop853_segment_input(Dop853Coordinates(levels=levels))
    with closing((case := item.build()).source):
        model = case.source.model((0,))
        assert model.energies_ghz.shape == (levels,)
        assert case.source.initial_density.shape == (levels, levels)
        assert np.trace(case.source.initial_density) == pytest.approx(1.0)
        for drive in model.drives:
            assert drive.in_phase_ghz.shape == (levels, levels)


def test_dop853_segment_rejects_dimension_above_native_cutoff() -> None:
    with pytest.raises(ValueError, match="native_cutoff"):
        dop853_segment_input(Dop853Coordinates(levels=100))


def test_segment_reference_key_predicts_cache_hit_and_separates_repeats_and_points(
    tmp_path: Path,
) -> None:
    keys = []
    for coordinates in (
        Dop853Coordinates(levels=3),
        Dop853Coordinates(levels=3, repeats=2),
        Dop853Coordinates(levels=3, points=2),
    ):
        item = dop853_segment_input(coordinates)
        with closing((case := item.build()).source):
            key = reference_key(case.source, case.batch)
            keys.append(key)
            with pytest.raises(FileNotFoundError, match=key):
                load_reference(tmp_path, case.source, case.batch)
            # Synthetic codec payload only; no reference integration is needed.
            expected = ReferenceResult(
                np.tile(case.source.initial_density, (len(case.batch.points), 1, 1)),
                1e-9,
            )
            assert write_reference(tmp_path, case.source, case.batch, expected) == key
        with closing((fresh := item.build()).source):
            assert reference_key(fresh.source, fresh.batch) == key
            actual = load_reference(tmp_path, fresh.source, fresh.batch)
            np.testing.assert_array_equal(actual.density, expected.density)
            assert actual.convergence_error == expected.convergence_error
    assert len(set(keys)) == 3
