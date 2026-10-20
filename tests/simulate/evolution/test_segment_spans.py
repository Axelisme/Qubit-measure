"""Neutral segment queries and externally averaged shifts at the public seam."""

from contextlib import closing
from dataclasses import replace
from threading import Event

from scipy.linalg import expm

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


def test_driven_shift_identities_do_not_alias_models_preparations_or_results():
    operator = np.array([[0, 0.004], [0.004, 0]], dtype=complex)
    model = evolution.SecularModel(
        (0,), "driven-two-level", np.array([0.0, 0.08]), np.eye(2, dtype=complex),
        (evolution.SecularDrive("drive", operator, np.zeros((2, 2), complex)),),
        np.zeros((2, 2)), np.zeros((2, 2)),
    )
    built_levels = []

    def build(level):
        built_levels.append(level)
        return replace(model, level=level)

    initial = np.diag([1.0, 0.0]).astype(complex)
    source = evolution.ModelSource(initial, build)
    tone = replace(track(1, evolution.EnvelopeSpec("constant"), duration=8), frequency_ghz=np.zeros(1))
    timeline = replace(batch(1, duration=8), tracks=(tone,))
    with closing(evolution.Engine(1, 1024**2)) as engine:
        for detuning in (0.0, 0.021, 0.037, 0.021):
            shifted = replace(timeline, segment_shifts=((evolution.SegmentShift(
                0, 8, np.array([0, detuning]), np.zeros((2, 2))
            ),),))
            result = engine.evolve(source, (shifted,), cancel=Event(), budget=1e-9)
            unitary = expm(-2j * np.pi * 8 * (np.diag([0, 0.08 + detuning]) + operator))
            np.testing.assert_allclose(result.densities[0], unitary @ initial @ unitary.conj().T, atol=1e-8)
    assert built_levels == [(0,)]


def test_shifts_follow_exact_query_spans_with_zero_gaps_and_repeat_cache_hits():
    query = evolution.segment_spans
    model = evolution.SecularModel(
        (0,), "gapped-two-level", np.array([0.0, 0.1]), np.eye(2, dtype=complex),
        (), np.zeros((2, 2)), np.zeros((2, 2)),
    )
    source = evolution.ModelSource(np.full((2, 2), 0.5, complex), lambda _: model)
    timeline = replace(batch(2, duration=12), segment_boundaries=((2.0, 6.0), (4.0, 8.0)))
    shifts = tuple((evolution.SegmentShift(a, b, np.array([0, delta]), np.zeros((2, 2))),)
                   for (a, b), delta in zip(((2, 6), (8, 12)), (0.02, 0.03), strict=True))
    shifted = replace(timeline, segment_shifts=shifts)
    assert query(shifted) == query(timeline)
    with closing(evolution.Engine(1, 1024**2)) as engine:
        first = engine.evolve(source, (shifted,), cancel=Event(), reuse_key=b"shifted", telemetry=True)
        again = engine.evolve(source, (shifted,), cancel=Event(), reuse_key=b"shifted", telemetry=True)
        assert engine.reuse(source, b"shifted")
    np.testing.assert_allclose(first.densities[:, 0, 1], 0.5 * np.exp(2j * np.pi * np.array([1.28, 1.32])), atol=1e-12)
    np.testing.assert_array_equal(again.densities, first.densities)
    assert sum(item.cache_hits for item in again.paths.routing.values()) == sum(first.paths.segments_by_kernel.values())
    assert sum(first.paths.segments_by_kernel.values()) == sum(map(len, query(timeline)))


@pytest.mark.parametrize("span", [(0, 12), (1, 2), (2, 7), (12, 14)])
def test_shift_span_must_match_final_query_span(span):
    shift = evolution.SegmentShift(*span, np.zeros(2), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="span"):
        replace(batch(1, duration=12), segment_boundaries=((2.0, 6.0),), segment_shifts=((shift,),))
