"""Caller-owned capture observes actual Engine evolves, not a second query."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate import evolution

from ._engine import RecordingKernel, batch, source, track
from .contract.synthetic_models import synthetic_model


def test_capture_freezes_all_batches_and_appends_per_evolve():
    capture = evolution.CaptureBuffer()
    first = replace(batch(2, duration=3.0), points=np.array([9, 4]))
    empty = batch(0, duration=2.0)
    with closing(source()) as src, closing(evolution.Engine(1)) as engine:
        plain = engine.evolve(src, (first, empty), capture=None)
        result = engine.evolve(src, (first, empty), capture=capture, budget=2e-3)
        snapshot = capture.records
        engine.evolve(src, (empty,), capture=capture)
    np.testing.assert_array_equal(result.densities, plain.densities)
    assert len(snapshot) == 2
    assert len(capture.records) == 3
    assert len({record.name for record in capture.records}) == 3
    metadata = [record.metadata for record in capture.records]
    assert all(isinstance(item, dict) for item in metadata)
    first_metadata = metadata[0]
    assert isinstance(first_metadata, dict)
    assert first_metadata["batch_index"] == 0
    assert first_metadata["batch_count"] == 2
    assert first_metadata["budget"] == 2e-3
    assert first_metadata["families"] is None
    assert first_metadata["source_identity"] == "synthetic-3"
    assert (
        first_metadata["frozen_physical_version"] == evolution.FROZEN_PHYSICAL_VERSION
    )
    assert (
        first_metadata["segment_numeric_version"] == evolution.SEGMENT_NUMERIC_VERSION
    )
    assert first_metadata["prefix_runs"] == []
    for index, record in enumerate(snapshot):
        item = record.metadata
        assert isinstance(item, dict)
        assert item["capture_id"] == first_metadata["capture_id"]
        assert item["batch_index"] == index
        assert item["complete"] is True
        assert record.name == f"{item['capture_id']}-{index}"
        frozen = evolution.read_physical_input(record.physical)
        with closing(frozen.source):
            np.testing.assert_array_equal(
                frozen.batch.points, (first, empty)[index].points
            )
    last = capture.records[-1].metadata
    assert isinstance(last, dict)
    assert last["capture_id"] != first_metadata["capture_id"]
    assert last["complete"] is True
    assert last["routes"] == []


def test_capture_records_actual_guards_and_short_circuited_candidates():
    capture = evolution.CaptureBuffer()
    pulse = replace(
        track(2, evolution.EnvelopeSpec("constant")), amplitude=np.full(2, 1e-6)
    )
    timeline = replace(batch(2), points=np.array([9, 4]), tracks=(pulse,))
    with closing(source()) as src, closing(evolution.Engine(2)) as engine:
        result = engine.evolve(src, (timeline,), capture=capture, telemetry=True)
    metadata = capture.records[0].metadata
    assert isinstance(metadata, dict)
    routes = metadata["routes"]
    assert [event["point"] for event in routes] == [9, 4]
    assert {event["family"] for event in routes} == set(result.paths.routing)
    for event in routes:
        assert event["round"] == "caller"
        assert event["prefix_run"] is None
        assert event["segment"] == 0
        assert event["start_ns"] == 0.0
        assert event["duration_ns"] == 3.0
        assert event["family"] == "dyson"
        assert event["candidates"][0] == {"family": "dyson", "guard": "accepted"}
        assert event["candidates"][1:]
        assert all(
            candidate["guard"] == "untested_after_short_circuit"
            for candidate in event["candidates"][1:]
        )
        assert result.paths.routing["dyson"].rows == len(routes)


def test_capture_keeps_caller_deferral_and_worker_decision_in_input_order():
    capture = evolution.CaptureBuffer()
    model = synthetic_model()
    model = replace(
        model,
        rates_per_ns=model.rates_per_ns * 0.001,
        dephasing_per_ns=model.dephasing_per_ns * 0.001,
        drives=tuple(
            replace(
                d,
                in_phase_ghz=d.in_phase_ghz * 0.1,
                quadrature_ghz=d.quadrature_ghz * 0.1,
            )
            for d in model.drives
        ),
    )
    strong = replace(
        track(2, evolution.EnvelopeSpec("constant"), duration=1.1),
        frequency_ghz=np.array([3.7, 3.8]),
        amplitude=np.full(2, 0.1),
    )
    probe = replace(
        strong, frequency_ghz=np.array([0.11, 0.12]), amplitude=np.full(2, 0.002)
    )
    timeline = replace(
        batch(2, duration=1.1), points=np.array([9, 4]), tracks=(strong, probe)
    )
    with closing(
        evolution.ModelSource(np.eye(3, dtype=np.complex128) / 3, lambda level: model)
    ) as src:
        for workers in (1, 2):
            with closing(evolution.Engine(workers, families=("floquet",))) as engine:
                result = engine.evolve(
                    src, (timeline,), capture=capture, telemetry=True
                )
            metadata = capture.records[-1].metadata
            assert isinstance(metadata, dict)
            assert metadata["families"] == ["floquet"]
            assert [
                (e["round"], e["point"], e["family"]) for e in metadata["routes"]
            ] == [
                ("caller", 9, None),
                ("caller", 4, None),
                ("worker", 9, "floquet"),
                ("worker", 4, "floquet"),
            ]
            assert [e["candidates"] for e in metadata["routes"]] == [
                [{"family": "floquet", "guard": "deferred"}],
                [{"family": "floquet", "guard": "deferred"}],
                [{"family": "floquet", "guard": "accepted"}],
                [{"family": "floquet", "guard": "accepted"}],
            ]
            assert result.paths.routing["floquet"].rows == 2
    first, second = (record.metadata for record in capture.records)
    assert isinstance(first, dict) and isinstance(second, dict)
    assert first["routes"] == second["routes"]


def test_capture_records_prefix_representative_and_cross_batch_adopters():
    capture = evolution.CaptureBuffer()
    pre = replace(
        track(1, evolution.EnvelopeSpec("constant"), duration=2.0),
        amplitude=np.full(1, 0.15),
    )
    probe = replace(
        track(1, evolution.EnvelopeSpec("constant"), start=3.0, duration=3.0),
        frequency_ghz=np.full(1, 0.4),
        amplitude=np.full(1, 1e-6),
    )
    first = replace(batch(1, duration=6.0), points=np.array([9]), tracks=(pre, probe))
    second = replace(first, points=np.array([4]))
    with (
        closing(source()) as src,
        closing(
            evolution.Engine(2, families=("t4", "dyson", "floquet_period", "t6"))
        ) as engine,
    ):
        result = engine.evolve(src, (first, second), capture=capture, telemetry=True)
    representative, adopter = (record.metadata for record in capture.records)
    assert isinstance(representative, dict) and isinstance(adopter, dict)
    assert representative["prefix_runs"] == [
        {
            "representative": {"point": 9, "segment": 2},
            "adopters": [
                {"batch": 0, "point": 9, "segment": 2},
                {"batch": 1, "point": 4, "segment": 2},
            ],
        }
    ]
    assert adopter["prefix_runs"] == []
    prefix = [e for e in representative["routes"] if e["prefix_run"] == 0]
    assert [(e["point"], e["segment"]) for e in prefix] == [(9, 0), (9, 1)]
    assert all(e["prefix_run"] is None and e["segment"] == 2 for e in adopter["routes"])
    logical = {}
    for event in prefix:
        logical[event["family"]] = logical.get(event["family"], 0) + 2
    for metadata in (representative, adopter):
        for event in metadata["routes"]:
            if event["prefix_run"] is None and event["family"] is not None:
                logical[event["family"]] = logical.get(event["family"], 0) + 1
    assert logical == result.paths.segments_by_kernel


def test_capture_keeps_guard_rejections_after_numerical_failure():
    capture = evolution.CaptureBuffer()
    pulse = track(
        2, evolution.EnvelopeSpec("sampled", np.array([0.2, 1.0]), np.array([0.0, 3.0]))
    )
    timeline = replace(
        batch(2), tracks=(pulse, replace(pulse, frequency_ghz=np.full(2, 0.03)))
    )
    with (
        closing(source()) as src,
        closing(evolution.Engine(1, families=("multitone_rwa",))) as engine,
        pytest.raises(evolution.NoValidKernel),
    ):
        engine.evolve(src, (timeline,), capture=capture)
    metadata = capture.records[0].metadata
    assert isinstance(metadata, dict)
    assert metadata["complete"] is False
    assert [e["point"] for e in metadata["routes"]] == [0, 1]
    for event in metadata["routes"]:
        assert event["family"] is None
        assert event["candidates"] == [{"family": "multitone_rwa", "guard": "rejected"}]


def test_capture_keeps_partial_freeze_and_propagates_original_error():
    capture = evolution.CaptureBuffer()
    error = ValueError("synthetic missing level")
    model = synthetic_model()

    def build(level):
        if level == (1,):
            raise error
        return model

    first = batch(1)
    second = replace(
        first, flux=replace(first.flux, level=np.ones((1, 1), dtype=np.int64))
    )
    with (
        closing(
            evolution.ModelSource(np.eye(3, dtype=np.complex128) / 3, build)
        ) as src,
        closing(evolution.Engine(1)) as engine,
    ):
        with pytest.raises(ValueError, match="synthetic missing level") as raised:
            engine.evolve(src, (first, second), capture=capture)
        assert raised.value is error
        assert len(capture.records) == 1
        metadata = capture.records[0].metadata
        assert isinstance(metadata, dict)
        assert metadata["complete"] is False
        assert metadata["routes"] == []
        engine.evolve(src, (first,), capture=capture)
    last = capture.records[-1].metadata
    assert isinstance(last, dict)
    assert last["complete"] is True


def test_capture_retains_frozen_batches_on_cancellation():
    capture = evolution.CaptureBuffer()
    cancel = Event()
    cancel.set()
    with (
        closing(source()) as src,
        closing(evolution.Engine(1)) as engine,
        pytest.raises(evolution.Cancelled),
    ):
        engine.evolve(src, (batch(1), batch(0)), capture=capture, cancel=cancel)
    assert len(capture.records) == 2
    for record in capture.records:
        metadata = record.metadata
        assert isinstance(metadata, dict)
        assert metadata["complete"] is False


def test_capture_rejects_concurrent_entry_before_waiting_for_engine():
    capture = evolution.CaptureBuffer()
    entered, release = Event(), Event()
    model = synthetic_model()

    def build(level):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test did not release model construction")
        return model

    with (
        closing(
            evolution.ModelSource(np.eye(3, dtype=np.complex128) / 3, build)
        ) as src,
        closing(evolution.Engine(1)) as engine,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        first = pool.submit(engine.evolve, src, (batch(1),), capture=capture)
        try:
            assert entered.wait(10)
            with pytest.raises(ValueError, match="capture buffer"):
                engine.evolve(src, (batch(1),), capture=capture)
        finally:
            release.set()
        first.result(timeout=10)
        engine.evolve(src, (batch(1),), capture=capture)
    assert len(capture.records) == 2
    assert all(
        isinstance(record.metadata, dict) and record.metadata["complete"]
        for record in capture.records
    )


def test_capture_reconstructs_segment_corrections_with_existing_codec():
    capture = evolution.CaptureBuffer()
    shift = evolution.SegmentShift(
        0.0, 1.5, np.array([0.0, 0.001, 0.002]), np.zeros((3, 3))
    )
    timeline = replace(
        batch(1), segment_boundaries=((1.5,),), segment_shifts=((shift,),)
    )
    with closing(source()) as src, closing(evolution.Engine(1)) as engine:
        original = engine.evolve(src, (timeline,), capture=capture)
        frozen = evolution.read_physical_input(capture.records[0].physical)
        with closing(frozen.source):
            replayed = engine.evolve(frozen.source, (frozen.batch,))
    np.testing.assert_array_equal(original.densities, replayed.densities)


class CountedIdle(RecordingKernel):
    """Observe guard/cost work outside the capture behavior under test."""

    def __init__(self):
        super().__init__("t4")
        self.guards = 0
        self.costs = 0

    def guard(self, table, rows, context):
        self.guards += 1
        return super().guard(table, rows, context)

    def cost(self, table, rows):
        self.costs += 1
        return super().cost(table, rows)


def test_capture_does_not_add_guard_or_cost_work():
    counts = []
    for capture in (None, evolution.CaptureBuffer()):
        kernel = CountedIdle()
        with (
            closing(source()) as src,
            closing(evolution.Engine(1, kernels=(kernel,))) as engine,
        ):
            engine.evolve(src, (batch(2),), capture=capture)
        counts.append((kernel.guards, kernel.costs))
    assert counts[0] == counts[1]
    assert counts[0][0] > 0 and counts[0][1] > 0
