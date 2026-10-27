"""Point-aligned continuation preserves the public common-origin timeline."""

from contextlib import closing
from dataclasses import replace
from inspect import signature
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate import evolution

from ._engine import batch, track


def _require_continuation():
    assert "start_ns" in signature(evolution.TimelineBatch).parameters
    assert "initial_densities" in signature(evolution.TimelineBatch).parameters


def _source(driven=False):
    operator = np.array([[0, 0.004], [0.004, 0]], complex)
    model = evolution.SecularModel(
        (0,),
        "continuation",
        np.array([0.0, 0.08]),
        np.eye(2, dtype=complex),
        (evolution.SecularDrive("drive", operator, np.zeros((2, 2), complex)),)
        if driven
        else (),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    return evolution.ModelSource(
        np.full((2, 2), 0.5, complex), lambda level: replace(model, level=level)
    )


@pytest.mark.parametrize("shaped", [False, True], ids=["flat", "sampled-envelope"])
def test_continuation_matches_one_shot_without_replaying_phase_or_envelope(shaped):
    _require_continuation()
    source = _source(driven=True)
    envelope = (
        evolution.EnvelopeSpec(
            "sampled",
            np.array([0.0, 0.8j, 0.3, 0.0], complex),
            np.array([0.0, 2.0, 9.0, 12.0]),
        )
        if shaped
        else evolution.EnvelopeSpec("constant")
    )
    tone = replace(
        track(2, envelope, start=2.0, duration=12.0),
        frequency_ghz=np.full(2, 0.081),
        phase_rad=np.full(2, 0.37),
    )
    timeline = replace(
        batch(2, duration=18.0),
        tracks=(tone,),
        segment_boundaries=((5.0, 10.0), (7.0, 12.0)),
    )
    spans = evolution.segment_spans(timeline)
    with closing(evolution.Engine(1)) as engine:
        whole = engine.evolve(source, (timeline,), cancel=Event(), budget=1e-9)
        densities = np.tile(source.initial_density, (2, 1, 1))
        for step in range(max(map(len, spans))):
            selected = [point[min(step, len(point) - 1)] for point in spans]
            starts = np.array(
                [a if step < len(spans[i]) else b for i, (a, b) in enumerate(selected)]
            )
            ends = np.array([b for _, b in selected])
            continued = replace(
                timeline,
                start_ns=starts,
                end_ns=ends,
                initial_densities=densities,
                segment_boundaries=tuple(
                    tuple(t for t in boundaries if t <= end)
                    for boundaries, end in zip(
                        timeline.segment_boundaries, ends, strict=True
                    )
                ),
            )
            prefix = evolution.segment_spans(continued)
            assert all(
                tuple(span for span in full if span[1] <= end) == actual
                for full, end, actual in zip(spans, ends, prefix, strict=True)
            )
            result = engine.evolve(source, (continued,), cancel=Event(), budget=1e-9)
            assert sum(result.paths.segments_by_kernel.values()) == sum(starts < ends)
            densities = result.densities
    np.testing.assert_allclose(densities, whole.densities, atol=1e-8)


def test_continuation_omission_is_equivalent_and_memo_isolates_start_and_density():
    _require_continuation()
    source = _source()
    timeline = replace(
        batch(2, duration=12.0), segment_boundaries=((4.0, 8.0), (4.0, 8.0))
    )
    densities = np.tile(source.initial_density, (2, 1, 1))
    explicit = replace(timeline, start_ns=np.zeros(2), initial_densities=densities)
    with closing(evolution.Engine(1)) as engine:
        omitted = engine.evolve(source, (timeline,), cancel=Event())
        given = engine.evolve(source, (explicit,), cancel=Event())
        np.testing.assert_allclose(omitted.densities, given.densities, atol=1e-12)
        for starts, coherences in (
            ([4.0, 8.0], [0.5, 0.25j]),
            ([8.0, 4.0], [0.25j, 0.5]),
            ([4.0, 8.0], [0.5, 0.25j]),
        ):
            initial = np.array([[[0.5, c], [np.conj(c), 0.5]] for c in coherences])
            continued = replace(
                timeline, start_ns=np.array(starts), initial_densities=initial
            )
            result = engine.evolve(
                source,
                (continued,),
                cancel=Event(),
                reuse_key=b"memory",
                telemetry=True,
            )
            np.testing.assert_allclose(
                result.densities[:, 0, 1],
                np.array(coherences)
                * np.exp(2j * np.pi * 0.08 * (12 - np.array(starts))),
                atol=1e-12,
            )
            assert engine.reuse(source, b"memory")
            again = engine.evolve(source, (continued,), cancel=Event())
            np.testing.assert_array_equal(again.densities, result.densities)


@pytest.mark.parametrize("start", [-1.0, 3.0, 13.0, np.nan, np.inf])
def test_continuation_start_must_be_a_final_boundary_with_point_and_value(start):
    _require_continuation()
    timeline = replace(
        batch(1, duration=12.0), points=np.array([17]), segment_boundaries=((4.0, 8.0),)
    )
    with pytest.raises(ValueError, match=rf"point 17.*{str(start)}"):
        replace(
            timeline,
            start_ns=np.array([start]),
            initial_densities=np.array([np.eye(2) / 2], complex),
        )


@pytest.mark.parametrize("missing", ["start_ns", "initial_densities"])
def test_continuation_fields_are_paired(missing):
    _require_continuation()
    kwargs = (
        {"initial_densities": np.array([np.eye(2) / 2], complex)}
        if missing == "start_ns"
        else {"start_ns": np.zeros(1)}
    )
    with pytest.raises(ValueError, match="together|paired|both"):
        replace(batch(1), **kwargs)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((1, 2, 3)),
        np.array([np.eye(2)], complex),
        np.array([[[1.0, 1j], [1j, 0.0]]]),
        np.array([np.diag([1.1, -0.1])], complex),
        np.zeros((2, 2, 2)),
    ],
)
def test_continuation_initial_density_uses_existing_invariants(invalid):
    _require_continuation()
    with pytest.raises(ValueError, match="initial densit"):
        replace(batch(1), start_ns=np.zeros(1), initial_densities=invalid)


def test_continuation_end_boundary_returns_unchanged_owned_density():
    _require_continuation()
    initial = np.array([[[0.7, 0.1j], [-0.1j, 0.3]]])
    continued = replace(
        batch(1, duration=12.0), start_ns=np.array([12.0]), initial_densities=initial
    )
    initial[0, 0, 0] = 0
    assert continued.initial_densities is not None
    assert not continued.initial_densities.flags.writeable
    with closing(evolution.Engine(1)) as engine:
        result = engine.evolve(_source(), (continued,), cancel=Event())
    np.testing.assert_array_equal(result.densities, [[[0.7, 0.1j], [-0.1j, 0.3]]])
    assert not result.paths.segments_by_kernel
