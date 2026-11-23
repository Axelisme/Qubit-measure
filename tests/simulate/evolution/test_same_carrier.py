"""Same-carrier drive folding at the public Engine boundary."""

from collections.abc import Mapping
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    DriveTrack,
    Engine,
    EnvelopeSpec,
    ModelSource,
    NoValidKernel,
    RoutingHistory,
    SecularDrive,
    SecularModel,
    canonical_json,
    physical_input,
    read_capture_metadata,
)
from zcu_tools.simulate.evolution.kernel import GuardResult, KernelContext, SegmentTable
from zcu_tools.simulate.evolution.kernels.floquet import FloquetKernel
from zcu_tools.simulate.evolution.kernels.multitone_rwa import MultitoneRWAKernel

from ._engine import RecordingKernel, batch, run, source, track
from .contract.floquet_cases import floquet_case

pytestmark = pytest.mark.sim


def _drives():
    return (
        SecularDrive(
            "a",
            np.array([[0.2, 0.4j, 0], [-0.4j, -0.1, 0.3], [0, 0.3, 0.1]]),
            np.array([[0.1, -0.2, 0.3j], [-0.2, 0.2, 0], [-0.3j, 0, -0.1]]),
        ),
        SecularDrive(
            "b",
            np.array([[0.1, 0.3, 0.2j], [0.3, 0.1, 0], [-0.2j, 0, -0.1]]),
            np.array([[0.2, 0, 0.4], [0, -0.3, 0.1j], [0.4, -0.1j, 0.1]]),
        ),
    )


def _source(*drives: SecularDrive):
    base = source()
    return ModelSource(
        base.initial_density,
        lambda level: replace(
            base.model(level),
            drives=tuple(
                replace(drive, in_phase_ghz=drive.in_phase_ghz * (1 + 0.2 * level[0]))
                for drive in drives
            ),
        ),
    )


def test_equal_support_at_one_carrier_admits_a_single_tone_family():
    pulse = replace(
        track(1, EnvelopeSpec("constant")),
        port="a",
        frequency_ghz=np.array([-0.02]),
        amplitude=np.array([-0.3]),
        phase_rad=np.array([0.4]),
    )
    other = replace(
        pulse, port="b", amplitude=np.array([0.2]), phase_rad=np.array([-0.9])
    )
    timeline = replace(batch(1), tracks=(pulse, other))

    result = run(
        Engine(1, families=("slow_envelope",)), _source(*_drives()), (timeline,)
    )

    assert result.paths.segments_by_kernel == {"slow_envelope": 1}


def _sampled_pair() -> tuple[DriveTrack, DriveTrack]:
    samples = np.array([0j, 0.25 + 0.1j, 0.5 - 0.14j, 0j])
    times = np.array([0.0, 1.0, 2.0, 3.0])
    envelope = EnvelopeSpec("sampled", samples, times)
    scaled = EnvelopeSpec("sampled", (0.3 - 0.8j) * samples, times)
    pulse = replace(track(1, envelope), port="a")
    return pulse, replace(pulse, port="b", envelopes=(scaled,))


def test_complex_scaled_samples_on_the_same_clock_admit_a_shaped_family():
    timeline = replace(batch(1), tracks=_sampled_pair())

    # Test the route without requiring this arbitrary model to pass a gate guard.
    result = run(
        Engine(1, kernels=(RecordingKernel("gate_cache"),), families=("gate_cache",)),
        _source(*_drives()),
        (timeline,),
    )

    assert result.paths.segments_by_kernel == {"gate_cache": 1}


@pytest.mark.parametrize("level", [0, 2])
def test_fold_matches_unfolded_dop853_with_signed_gain_and_phase(level: int):
    pulse, other = _sampled_pair()
    pulse = replace(
        pulse,
        start_ns=np.array([1.125]),
        frequency_ghz=np.array([-0.12]),
        amplitude=np.array([-0.31]),
        phase_rad=np.array([0.4]),
    )
    other = replace(
        other,
        start_ns=pulse.start_ns,
        frequency_ghz=pulse.frequency_ghz,
        amplitude=np.array([0.2]),
        phase_rad=np.array([-0.9]),
    )
    timeline = batch(1, duration=4.125)
    folded = replace(
        timeline,
        tracks=(pulse, other),
        flux=replace(timeline.flux, level=np.array([[level]])),
        segment_boundaries=((2.5,),),
    )
    # A redundant knot leaves the physical interpolation identical, but gives
    # the two channels different native sample clocks so they cannot fold.
    envelope = other.envelopes[0]
    times = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    resampled = EnvelopeSpec("sampled", envelope.values(times), times)
    unfurled = replace(folded, tracks=(pulse, replace(other, envelopes=(resampled,))))

    with (
        closing(_source(*_drives())) as src,
        closing(Engine(1, backend="cpu", families=("t4", "dop853"))) as engine,
    ):
        actual = engine.evolve(src, (folded,), budget=1e-10)
        expected = engine.evolve(src, (unfurled,), budget=1e-10)

    np.testing.assert_allclose(actual.densities, expected.densities, rtol=0, atol=5e-10)


@pytest.mark.parametrize("mismatch", ["shape", "support", "signed_carrier"])
def test_incompatible_lines_stay_in_the_existing_multitone_route(mismatch: str):
    pulse, other = _sampled_pair()
    if mismatch == "shape":
        # One pair of a three-line group is proportional. Keep the entire group
        # multitone rather than merging only the matching pair.
        other = replace(other, envelopes=(pulse.envelopes[0],))
        third = replace(pulse, envelopes=(EnvelopeSpec("constant"),))
        tracks = (pulse, other, third)
    elif mismatch == "support":
        tracks = (pulse, replace(other, duration_ns=np.array([4.0])))
    else:
        tracks = (pulse, replace(other, frequency_ghz=-pulse.frequency_ghz))
    timeline = replace(batch(1), tracks=tracks)
    kernels = (RecordingKernel("gate_cache"), RecordingKernel("dop853"))
    # The multitone row must skip the single-tone offer and still accept direct.
    with (
        closing(Engine(1, kernels=kernels, families=("gate_cache",))) as engine,
        pytest.raises(NoValidKernel, match="domain multiple"),
    ):
        engine.evolve(_source(*_drives()), (timeline,))
    result = run(Engine(1, kernels=kernels), _source(*_drives()), (timeline,))
    assert result.paths.segments_by_kernel == {"dop853": 1}


def test_unrelated_fold_does_not_change_two_tone_floquet_frame():
    class FrameSizes(FloquetKernel):
        def __init__(self) -> None:
            self.sizes: list[int] = []

        def guard_prepared(
            self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
        ) -> GuardResult:
            result = super().guard_prepared(table, rows, context)
            self.sizes.extend(item.prepared.nbytes for item in result.preparations)
            return result

    case = floquet_case("sideband")
    floquet = FrameSizes()
    # This separate DC pair exercises the virtual bank, not Floquet playback.
    dc = replace(
        case.batch.tracks[0], frequency_ghz=np.zeros(1), amplitude=np.full(1, 1e18)
    )
    other = replace(dc, port=case.batch.tracks[1].port)
    folded = replace(case.batch, points=np.array([1]), tracks=(dc, other))
    # Observe routing of the real Floquet row without evolving the strong DC pair.
    with (
        closing(case.source),
        closing(
            Engine(
                1,
                backend="cpu",
                kernels=(floquet, RecordingKernel("slow_envelope")),
            )
        ) as engine,
    ):
        standalone = engine.evolve(case.source, (case.batch,))
        standalone_sizes = tuple(floquet.sizes)
        floquet.sizes.clear()
        combined = engine.evolve(case.source, (case.batch, folded))
    assert standalone.paths.segments_by_kernel == {"floquet": 1}
    assert combined.paths.segments_by_kernel == {"floquet": 1, "slow_envelope": 1}
    np.testing.assert_array_equal(combined.densities[0], standalone.densities[0])
    assert standalone_sizes and tuple(floquet.sizes) == standalone_sizes


@pytest.mark.parametrize("folded_slot", ["stark", "probe"])
def test_floquet_uses_its_folded_port_after_an_unrelated_virtual_port(
    folded_slot: str,
) -> None:
    case = floquet_case("sideband")
    stark, probe = case.batch.tracks
    # Floquet picks its Stark tone by gain, and a folded slot always has gain 1.
    # Rescale the physical operator and gain inversely to retain the same pump.
    with closing(case.source):
        native = case.source.model((0,))
        pump, read = native.drives
        model = replace(
            native, drives=(replace(pump, in_phase_ghz=pump.in_phase_ghz / 10), read)
        )
        initial = case.source.initial_density
    stark = replace(stark, amplitude=10 * stark.amplitude, phase_rad=np.zeros(1))
    probe = replace(probe, phase_rad=np.zeros(1))
    standalone = replace(case.batch, tracks=(stark, probe))
    pulse = stark if folded_slot == "stark" else probe
    half = replace(pulse, amplitude=pulse.amplitude / 2)
    tracks = (half, half, probe) if folded_slot == "stark" else (stark, half, half)
    folded = replace(standalone, tracks=tracks)
    dc = replace(stark, frequency_ghz=np.zeros(1))
    unrelated = replace(
        standalone, points=np.array([1]), tracks=(dc, replace(dc, port=probe.port))
    )
    with (
        closing(ModelSource(initial, lambda level: model)) as src,
        closing(
            Engine(
                1,
                backend="cpu",
                kernels=(FloquetKernel(), RecordingKernel("slow_envelope")),
            )
        ) as engine,
    ):
        expected = engine.evolve(src, (standalone,))
        actual = engine.evolve(src, (unrelated, folded))
    assert expected.paths.segments_by_kernel == {"floquet": 1}
    assert actual.paths.segments_by_kernel == {"floquet": 1, "slow_envelope": 1}
    np.testing.assert_allclose(
        actual.densities[1], expected.densities[0], rtol=0, atol=1e-10
    )


def test_expanded_pair_history_follows_fold_recipe_not_encounter_order() -> None:
    class UsageHistory(RoutingHistory):
        def __init__(self) -> None:
            super().__init__()
            self.usage: dict[bytes, int] = {}

        def record_gate_usage(
            self, inventory: Mapping[bytes, int], *, content: bytes
        ) -> dict[bytes, int]:
            self.usage = super().record_gate_usage(inventory, content=content)
            return self.usage

    # The 60 MHz detuning is outside the initial 50 MHz window. The 4 MHz
    # circular component exceeds the omission-transfer limit. At 75 ns the
    # restored RWA product wins the price comparison at two uses, not one.
    positive = np.array([[0, 0.004], [0, 0]], dtype=complex)
    zero = np.zeros_like(positive)
    model = SecularModel(
        (0,),
        "folded-expanded-pair",
        np.array([0.0, 0.36]),
        np.eye(2, dtype=complex),
        (
            SecularDrive("a", positive + positive.T, 1j * (positive - positive.T)),
            *(SecularDrive(name, zero, zero) for name in ("b", "c", "probe")),
        ),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    pulse = replace(
        track(1, EnvelopeSpec("constant"), duration=75.0),
        port="a",
        frequency_ghz=np.array([0.3]),
    )
    probe = replace(pulse, port="probe", frequency_ghz=np.array([0.9]))
    template = batch(1, duration=75.0)
    first, other = (
        replace(
            template,
            points=np.array([point]),
            tracks=(pulse, replace(pulse, port=channel), probe),
        )
        for point, channel in enumerate(("b", "c"))
    )
    # Both folds produce identical quadratures but use different physical lines.
    history = UsageHistory()
    initial = np.diag(np.full(2, 0.5, dtype=np.complex128))
    with (
        closing(ModelSource(initial, lambda level: model, history=history)) as src,
        closing(
            Engine(
                1,
                kernels=(MultitoneRWAKernel(), RecordingKernel("dop853")),
                backend="cpu",
            )
        ) as engine,
    ):
        cold = engine.evolve(src, (first,), reuse_key=b"cold")
        assert cold.paths.segments_by_kernel == {"dop853": 1}
        assert engine.reuse(src, b"cold")
        assert list(history.usage.values()) == [1]
        (first_key,) = history.usage

        warming = engine.evolve(src, (other, first))
        assert first_key in history.usage, "the warm recipe lost its history key"
        (other_key,) = history.usage.keys() - {first_key}
        assert history.usage == {first_key: 2, other_key: 1}
        assert warming.paths.segments_by_kernel == {"dop853": 1, "multitone_rwa": 1}
        assert not engine.reuse(src, b"cold")

        warm = engine.evolve(src, (first, other))
        assert history.usage == {first_key: 3, other_key: 2}
        assert warm.paths.segments_by_kernel == {"multitone_rwa": 2}
        repeated = engine.evolve(src, (first, other))
        assert history.usage == {first_key: 3, other_key: 2}
        assert repeated.paths.segments_by_kernel == warm.paths.segments_by_kernel


@pytest.fixture
def channel_sets():
    a, b = _drives()
    c = replace(b, name="c", in_phase_ghz=-b.in_phase_ghz)
    pulse = replace(
        track(1, EnvelopeSpec("constant")),
        port="a",
        frequency_ghz=np.zeros(1),
        amplitude=np.array([0.2]),
    )
    batches = tuple(
        replace(
            batch(1),
            points=np.array([point]),
            tracks=(pulse, replace(pulse, port=channel)),
        )
        for point, channel in enumerate(("b", "c", "b", "c"))
    )
    return (a, b, c), batches


def test_channel_sets_keep_distinct_preparations_across_batches(channel_sets):
    drives, batches = channel_sets
    with (
        closing(_source(*drives)) as src,
        closing(Engine(1, backend="cpu", families=("slow_envelope",))) as engine,
    ):
        result = engine.evolve(src, batches, telemetry=True, reuse_key=b"test")
        assert engine.reuse(src, b"test")

    assert result.paths.routing["slow_envelope"].prepare_key_builds == 2
    np.testing.assert_array_equal(result.densities[:2], result.densities[2:])
    assert np.max(np.abs(result.densities[0] - result.densities[1])) > 1e-3


def test_model_only_prepare_shares_dop853_without_aliasing_fold_recipes(channel_sets):
    drives, batches = channel_sets
    folded = batches[:2]
    # Identical interpolation does not imply an identical sample clock.
    times = np.array([0.0, 1.5, 3.0])
    redundant_clock = np.array([0.0, 0.75, 1.5, 3.0])
    envelope = EnvelopeSpec("sampled", np.ones(len(times), dtype=complex), times)
    resampled = EnvelopeSpec(
        "sampled", np.ones(len(redundant_clock), dtype=complex), redundant_clock
    )
    folded = tuple(
        replace(
            item, tracks=tuple(replace(t, envelopes=(envelope,)) for t in item.tracks)
        )
        for item in folded
    )
    unfurled = tuple(
        replace(
            item,
            tracks=(item.tracks[0], replace(item.tracks[1], envelopes=(resampled,))),
        )
        for item in folded
    )
    with (
        closing(_source(*drives)) as src,
        closing(Engine(1, backend="cpu", families=("dop853",))) as engine,
    ):
        actual = engine.evolve(src, folded, budget=1e-10, telemetry=True)
        expected = engine.evolve(src, unfurled, budget=1e-10)

    assert actual.paths.routing["dop853"].prepare_key_builds == 1
    np.testing.assert_allclose(actual.densities, expected.densities, rtol=0, atol=5e-10)
    assert np.max(np.abs(actual.densities[0] - actual.densities[1])) > 1e-3


def test_prefix_reuses_only_the_same_channels_and_capture_keeps_physics(channel_sets):
    drives, batches = channel_sets
    suffix = replace(batches[0].tracks[0], start_ns=np.array([3.0]))
    timelines = tuple(
        replace(item, tracks=(*item.tracks, suffix), end_ns=np.array([6.0]))
        for item in batches
    )
    capture = CaptureBuffer()
    with (
        closing(_source(*drives)) as src,
        closing(Engine(1, backend="cpu", families=("slow_envelope",))) as engine,
    ):
        result = engine.evolve(src, timelines, capture=capture)
        physical = tuple(
            canonical_json(physical_input(src, item)) for item in timelines
        )

    assert (
        tuple(canonical_json(record.physical) for record in capture.records) == physical
    )
    adopted = [
        {adopter["batch"] for adopter in prefix["adopters"]}
        for record in capture.records
        for prefix in read_capture_metadata(record)["prefix_runs"]
    ]
    assert adopted == [{0, 2}, {1, 3}]
    np.testing.assert_array_equal(result.densities[:2], result.densities[2:])
    assert np.max(np.abs(result.densities[0] - result.densities[1])) > 1e-3
