"""Sampled RWA prices OpenCL only when its independent endpoints could win."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec

from .accuracy_cases import AccuracyCase, accuracy_case, drive_track, timeline


def _stub_device(monkeypatch: pytest.MonkeyPatch, *, available: bool) -> list[str]:
    # Device discovery has no public injection point. Stub the external device,
    # not the backend state, wave prices, guard, or any dispatch decisions.
    from zcu_tools.simulate.evolution.kernels import _dop853_opencl

    events: list[str] = []

    class Device:
        def check_device(self) -> None:
            events.append("check")
            if not available:
                raise RuntimeError("No OpenCL device in this test")

        def cached_build_failure(self, dimension: int, *, dp54: bool) -> None:
            events.append("cached")

        def compile_variant(self, dimension: int, *, dp54: bool) -> str:
            events.append("compile")
            return "no point program in the test device"

        def close(self) -> None:
            pass

    monkeypatch.setattr(_dop853_opencl, "OpenCLDOP853", Device)
    return events


def _long_sampled_pair() -> AccuracyCase:
    count = 12
    case = accuracy_case("joint_gate_inventory")
    batch = timeline(count, 100.0)
    knots = np.linspace(0.0, 100.0, 200)
    envelope = EnvelopeSpec(
        "sampled", np.linspace(0.8, 1.0, len(knots)).astype(complex), knots
    )
    tracks = tuple(
        replace(
            drive_track(
                batch,
                envelope,
                frequency=float(track.frequency_ghz[0]),
                gain=float(track.amplitude[0]),
                port=track.port,
            ),
            phase_rad=np.full(count, track.phase_rad[0]),
        )
        for track in case.batch.tracks
    )
    return replace(case, batch=replace(batch, tracks=tracks))


@pytest.mark.parametrize(
    ("neighbor", "expected"),
    [("plateau", "dop853"), ("direct", "multitone_rwa"), ("pending", "multitone_rwa")],
)
def test_sampled_opencl_price_respects_adjacent_routes(
    monkeypatch: pytest.MonkeyPatch, neighbor: str, expected: str
) -> None:
    events = _stub_device(monkeypatch, available=True)
    case = _long_sampled_pair()
    count = len(case.batch.points)
    first = case.batch.tracks[0]
    if neighbor == "pending":
        # One distinct phase per endpoint prevents the later groups from sharing
        # their own price. They have not been routed when the first group prices.
        tail = tuple(
            replace(
                t, start_ns=np.full(count, 100.0), phase_rad=np.linspace(0, 1, count)
            )
            for t in case.batch.tracks
        )
        tracks = (*case.batch.tracks, *tail)
        end, sampled_segment = 200.0, 0
        neighbor_family = "dop853"
    else:
        frequency = (
            first.frequency_ghz if neighbor == "plateau" else np.full(count, 0.01)
        )
        leading = replace(
            first,
            duration_ns=np.full(count, 8.0),
            frequency_ghz=frequency,
            envelopes=(EnvelopeSpec("constant"),),
        )
        tracks = (
            leading,
            *(replace(t, start_ns=np.full(count, 8.0)) for t in case.batch.tracks),
        )
        end, sampled_segment = 108.0, 1
        neighbor_family = "single_tone_rwa" if neighbor == "plateau" else "dop853"
    batch = replace(case.batch, end_ns=np.full(count, end), tracks=tracks)
    families = ("single_tone_rwa", "multitone_rwa", "dop853")
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=families)) as cpu,
        closing(Engine(1, backend="auto", families=families)) as auto,
    ):
        for engine, family in ((cpu, "multitone_rwa"), (auto, expected)):
            query = engine.inspect_dispatch(case.source, (batch,))
            assert {row.family for row in query.rows[1 - sampled_segment :: 2]} == {
                neighbor_family
            }
            assert {row.family for row in query.rows[sampled_segment::2]} == {family}
    assert events == (["check", "cached"] if neighbor == "plateau" else [])


@pytest.mark.parametrize("available", [False, True], ids=["cpu-only", "gpu-available"])
def test_sampled_price_queries_reuse_only_a_decisive_device_check(
    monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    events = _stub_device(monkeypatch, available=available)
    case = _long_sampled_pair()
    batch = case.batch
    count = len(batch.points)
    envelope = batch.tracks[0].envelopes[0]
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu")) as cpu,
        closing(Engine(1, backend="auto")) as auto,
    ):
        assert {
            row.family for row in cpu.inspect_dispatch(case.source, (batch,)).rows
        } == {"multitone_rwa"}
        # One consumer cannot repay even its CPU probe. A device cannot alter
        # the family, so the query does not check hardware for this smaller group.
        small = timeline(1, 100.0)
        small = replace(
            small,
            tracks=tuple(
                drive_track(
                    small,
                    envelope,
                    port=t.port,
                    frequency=float(t.frequency_ghz[0]),
                    gain=float(t.amplitude[0]),
                )
                for t in batch.tracks
            ),
        )
        assert auto.inspect_dispatch(case.source, (small,)).rows[0].family == "dop853"
        assert not events
        expected = "dop853" if available else "multitone_rwa"
        for _ in range(2):
            query = auto.inspect_dispatch(case.source, (batch,))
            assert {row.family for row in query.rows} == {expected}
        assert events == (["check", "cached"] if available else ["check"])
        result = auto.evolve(case.source, (batch,))
        assert result.paths.segments_by_kernel == {expected: count}
        assert events.count("check") == 1
