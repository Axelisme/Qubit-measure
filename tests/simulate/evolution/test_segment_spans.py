"""Neutral segment queries and externally averaged shifts at the public seam."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate import evolution

from ._engine import batch, track


def test_segment_spans_include_neutral_boundaries_and_microwave_plateaus():
    query = evolution.segment_spans
    envelope = evolution.EnvelopeSpec(
        "sampled", np.array([0, 1, 1, 0], complex), np.array([0, 2, 14, 16])
    )
    timeline = replace(
        batch(2, duration=24),
        tracks=(track(2, envelope, start=2, duration=18),),
        segment_boundaries=((6.0, 22.0), (8.0,)),
    )
    spans = query(timeline)
    assert spans == (
        ((0, 2), (2, 4), (4, 6), (6, 16), (16, 20), (20, 22), (22, 24)),
        ((0, 2), (2, 4), (4, 8), (8, 16), (16, 20), (20, 24)),
    )
    assert query(replace(batch(1), end_ns=np.zeros(1))) == ((),)


@pytest.mark.parametrize("boundaries", [((3.0, 2.0),), ((2.0, 2.0),), ((-1.0,),), ((4.0,),), ((np.nan,),), ((), ())])
def test_segment_boundaries_fail_fast(boundaries):
    evolution.segment_spans
    with pytest.raises(ValueError, match="boundar"):
        replace(batch(1), segment_boundaries=boundaries)


def test_single_segment_constant_photons_matches_two_level_stark_solution():
    # Constant n=3, chi_e-chi_g=0.007 GHz gives 0.021 GHz Stark detuning.
    shift = evolution.SegmentShift(
        0.0, 8.0, np.array([0.0, 0.021]), np.array([[0.0, 0.003], [0.003, 0.0]])
    )
    model = evolution.SecularModel(
        (0,), "two-level", np.array([0.0, 0.13]), np.eye(2, dtype=complex),
        (), np.zeros((2, 2)), np.zeros((2, 2)),
    )
    initial = np.full((2, 2), 0.5, dtype=complex)
    source = evolution.ModelSource(initial, lambda level: replace(model, level=level))
    timeline = replace(batch(1, duration=8), segment_shifts=((shift,),))
    with closing(evolution.Engine(1, 1024**2)) as engine:
        result = engine.evolve(source, (timeline,), cancel=Event())
    expected = 0.5 * np.exp((2j * np.pi * 0.151 - 0.003) * 8)
    np.testing.assert_allclose(result.densities[0, 0, 1], expected, atol=1e-12)
    np.testing.assert_allclose(np.diag(result.densities[0]), [0.5, 0.5], atol=1e-12)
