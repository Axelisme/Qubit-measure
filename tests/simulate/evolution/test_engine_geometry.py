"""Engine timeline geometry and physical row projection through kernel observations."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    SecularModel,
)

from ._engine import RecordingKernel, batch, run, source, track
from .contract.kernel_suite import synthetic_model


def test_geometry_preserves_track_order_mux_plateaus_and_flux_levels():
    from zcu_tools.simulate.evolution.kernels import REGISTRY

    base = synthetic_model()
    model = replace(base, drives=(base.drives[0], replace(base.drives[0], name="mux")))
    src = ModelSource(
        source().initial_density, lambda level: replace(model, level=level)
    )
    env = EnvelopeSpec(
        "sampled", np.array([0, 1j, 1j, 0], complex), np.array([0.0, 2.0, 12.0, 14.0])
    )
    pulse = replace(track(1, env, duration=20), frequency_ghz=np.array([-0.1]))
    mux = replace(
        track(1, EnvelopeSpec("constant"), start=4, duration=12),
        port="mux",
        phase_rad=np.array([0.7]),
    )
    silent = replace(track(1, env, start=1, duration=22), amplitude=np.zeros(1))
    timeline = replace(
        batch(1, duration=24),
        tracks=(pulse, silent, mux),
        flux=FluxSchedule(np.array([0, 2]), np.array([0.0, 8.0]), np.array([[0], [1]])),
    )
    kernels = tuple(RecordingKernel(k.name) for k in REGISTRY)
    result = run(Engine(1, 0, kernels=kernels), src, (timeline,))
    observations = {
        float(table.start_ns[row]): (table, int(row), kernel.name)
        for kernel in kernels
        for table, rows, _ in kernel.calls
        for row in rows
    }
    assert sorted(observations) == [0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    for start, (table, row, family) in observations.items():
        count = (start < 20) + (4 <= start < 16)
        assert table.tones[row] == count
        assert table.level[row].tolist() == [int(start >= 8)]
        assert family == ("flux_dop853" if count else "t4")
        if count:
            assert table.tone_port[row, :count].tolist() == (
                [0, 1] if count == 2 else [0]
            )
            flat = 2 <= start < 12
            assert table.envelopes[table.tone_envelope[row, 0]].kind == (
                "constant" if flat else "sampled"
            )
            assert table.tone_envelope_offset_ns[row, 0] == (0 if flat else start)
            assert table.tone_theta_rad[row, 0] == (
                2 * np.pi * -0.1 * start + (np.pi / 2 if flat else 0)
            )
            if count == 2:
                assert table.tone_envelope_offset_ns[row, 1] == start - 4
                assert table.tone_theta_rad[row, 1] == 2 * np.pi * 0.1 * start + 0.7
    np.testing.assert_array_equal(result.densities[0], src.initial_density)


@pytest.mark.parametrize(
    ("times", "values"),
    [
        ([0.0, 10.0, 30.0, 240.0], [0.0, 1j, 1j, 0.3j]),
        ([0.0, 12.0, 32.0, 240.0], [0.0, 1j, 1j, 0.3j]),
        ([0.0, 10.0, 30.0, 240.0], [0.0, 1j, 1j, 0.4j]),
    ],
)
def test_equal_waveform_copies_keep_physical_rows_across_batches(times, values):
    from zcu_tools.simulate.evolution.kernels import REGISTRY

    base = synthetic_model()
    drives = (base.drives[0], replace(base.drives[0], name="mux"))

    def model(level):
        return replace(
            base, level=level, drives=drives if level == (0,) else drives[::-1]
        )

    src = ModelSource(source().initial_density, model)
    batches = []
    for point, duration in enumerate((254.0, 256.0, 254.0, 256.0)):
        # Equal content is deliberately reconstructed as a distinct record.
        env = EnvelopeSpec("sampled", np.array(values), np.array(times))
        batches.append(
            replace(
                batch(1, duration=duration),
                points=np.array([point]),
                flux=FluxSchedule(
                    np.array([0, 2]),
                    np.array([0.0, 100.0]),
                    np.array([[0], [1]]),
                ),
                tracks=(
                    replace(
                        track(1, env, duration=duration),
                        phase_rad=np.array([point * 0.13]),
                    ),
                    replace(
                        track(1, env, duration=duration),
                        port="absent",
                        active=np.array([False]),
                    ),
                    replace(
                        track(1, EnvelopeSpec("constant"), start=4, duration=238),
                        port="mux",
                    ),
                ),
            )
        )
    kernels = tuple(RecordingKernel(k.name) for k in REGISTRY)
    result = run(Engine(1, 0, kernels=kernels), src, tuple(batches))
    observations = {
        (int(table.timeline[row]), float(table.start_ns[row])): (table, int(row))
        for kernel in kernels
        for table, rows, _ in kernel.calls
        for row in rows
    }
    for point, duration in enumerate((254.0, 256.0, 254.0, 256.0)):
        boundaries = [0.0, 4.0, times[1], times[2], 100.0, 240.0, 242.0, duration]
        assert (
            sorted(start for p, start in observations if p == point) == boundaries[:-1]
        )
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            table, row = observations[point, start]
            assert table.duration_ns[row] == end - start
            count = 1 + (4 <= start < 242)
            level = int(start >= 100)
            assert table.level[row].tolist() == [level]
            assert table.tones[row] == count
            assert (
                table.tone_port[row, :count].tolist()
                == ([0, 1] if not level else [1, 0])[:count]
            )
            physical = table.envelopes[table.tone_envelope[row, 0]]
            flat = times[1] <= start < times[2] or start >= 240
            assert physical.kind == ("constant" if flat else "sampled")
            if flat:
                value = values[-1] if start >= 240 else 1j
                assert table.tone_amplitude[row, 0] == abs(value)
                assert table.tone_envelope_offset_ns[row, 0] == 0
                assert table.tone_theta_rad[
                    row, 0
                ] == 2 * np.pi * 0.1 * start + point * 0.13 + np.angle(value)
            else:
                np.testing.assert_array_equal(physical.samples, values)
                np.testing.assert_array_equal(physical.sample_ns, times)
                assert table.tone_envelope_offset_ns[row, 0] == start
    np.testing.assert_array_equal(result.points, np.arange(4))
    np.testing.assert_array_equal(
        result.densities, np.repeat(src.initial_density[None], 4, axis=0)
    )


def test_waveform_time_and_complex_samples_remain_distinct_in_one_acquire():
    envelopes = (
        EnvelopeSpec("sampled", np.array([0.2, 1j]), np.array([0.0, 2.0])),
        EnvelopeSpec("sampled", np.array([0.2, 1j]), np.array([0.0, 3.0])),
        EnvelopeSpec("sampled", np.array([0.2, -1j]), np.array([0.0, 2.0])),
    )
    timelines = tuple(
        replace(
            batch(1),
            points=np.array([point]),
            tracks=(track(1, env),),
        )
        for point, env in enumerate(envelopes)
    )
    direct = RecordingKernel("dop853")
    src = source()
    result = run(Engine(1, 0, kernels=(direct,)), src, timelines)
    for table, rows, _ in direct.calls:
        for row in rows:
            expected = envelopes[table.timeline[row]]
            physical = table.envelopes[table.tone_envelope[row, 0]]
            np.testing.assert_array_equal(physical.samples, expected.samples)
            np.testing.assert_array_equal(physical.sample_ns, expected.sample_ns)
    np.testing.assert_array_equal(
        result.densities, np.repeat(src.initial_density[None], 3, axis=0)
    )


def test_active_unknown_port_raises_value_error():
    timeline = replace(
        batch(1),
        tracks=(replace(track(1, EnvelopeSpec("constant")), port="absent"),),
    )
    with (
        closing(Engine(1, 0, kernels=(RecordingKernel("t6"),))) as engine,
        pytest.raises(ValueError, match="absent"),
    ):
        engine.evolve(source(), (timeline,), cancel=Event())


def test_flux_basis_changes_and_readout_truncation():
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    zero = np.zeros((2, 2))
    swap = np.array([[0, 1], [1, 0]], dtype=complex)

    def build(level):
        return SecularModel(
            level,
            str(level),
            np.zeros(2),
            np.eye(2, dtype=complex) if level == (0,) else swap,
            (),
            zero,
            zero,
        )

    src = ModelSource(np.diag([1.0, 0.0]).astype(complex), build)
    b = replace(
        batch(1),
        flux=FluxSchedule(
            np.array([0, 3]), np.array([0.0, 1.0, 100.0]), np.array([[0], [1], [99]])
        ),
        tracks=(replace(track(1, EnvelopeSpec("constant"), start=20), port="absent"),),
    )
    result = run(Engine(1, 1000, kernels=(T4(),)), src, (b,))
    np.testing.assert_allclose(result.densities[0], src.initial_density, atol=1e-14)
    assert result.paths.segments_by_kernel == {"t4": 2}


def test_flat_top_split_phase_offset_and_small_plateau():
    envelope = EnvelopeSpec(
        "sampled", np.array([0, 1j, 1j, 0], complex), np.array([0.0, 2.0, 12.0, 14.0])
    )
    direct, t6 = RecordingKernel("dop853"), RecordingKernel("t6")
    result = run(
        Engine(
            1,
            1000,
            kernels=(
                direct,
                RecordingKernel("dyson", accepts=False),
                RecordingKernel("floquet_period", accepts=False),
                t6,
            ),
        ),
        source(),
        (replace(batch(1, duration=14), tracks=(track(1, envelope, duration=14),)),),
    )
    assert result.paths.segments_by_kernel == {"dop853": 2, "t6": 1}
    table, rows, _ = t6.calls[0]
    assert table.start_ns[rows].tolist() == [2.0]
    assert table.duration_ns[rows].tolist() == [10.0]
    np.testing.assert_allclose(
        table.tone_theta_rad[rows, 0], 2 * np.pi * 0.1 * 2 + np.pi / 2
    )
    assert table.envelopes[table.tone_envelope[rows[0], 0]].kind == "constant"
    assert direct.calls[-1][0].tone_envelope_offset_ns[
        direct.calls[-1][1], 0
    ].tolist() == [12.0]
    short = replace(envelope, sample_ns=np.array([0.0, 2.0, 11.0, 14.0]))
    k = RecordingKernel("dop853")
    result = run(
        Engine(1, 0, kernels=(k,)),
        source(),
        (replace(batch(1, duration=14), tracks=(track(1, short, duration=14),)),),
    )
    assert result.paths.segments_by_kernel == {"dop853": 1}


@pytest.mark.parametrize("kind", ["sampled", "constant"])
def test_pulse_support_ends_at_its_absolute_event(kind):
    envelope = (
        EnvelopeSpec("sampled", np.array([0.2, 1.0]), np.array([0.0, 0.8]))
        if kind == "sampled"
        else EnvelopeSpec("constant")
    )
    idle, direct, constant, flux = (
        RecordingKernel(name) for name in ("t4", "dop853", "t6", "flux_dop853")
    )
    timeline = replace(
        batch(1, duration=4.0),
        # A level switch exactly at the first pulse's end does not overlap it.
        flux=FluxSchedule(
            np.array([0, 3]),
            np.array([0.0, 2.8, 3.2]),
            np.array([[0], [1], [1]]),
        ),
        tracks=(
            track(1, envelope, start=2.0, duration=0.8),
            track(1, envelope, start=3.2, duration=0.8),
        ),
    )
    result = run(
        Engine(
            1,
            0,
            kernels=(
                idle,
                direct,
                constant,
                flux,
                RecordingKernel("dyson", accepts=False),
                RecordingKernel("floquet_period", accepts=False),
            ),
        ),
        source(),
        (timeline,),
    )
    observations = sorted(
        (float(table.start_ns[row]), int(table.tones[row]))
        for kernel in (idle, direct, constant, flux)
        for table, rows, _ in kernel.calls
        for row in rows
    )
    assert observations == [(0.0, 0), (2.0, 1), (2.8, 0), (3.2, 1)]
    driven = direct if kind == "sampled" else constant
    assert result.paths.segments_by_kernel == {"t4": 2, driven.name: 2}
    for table, rows, _ in driven.calls:
        np.testing.assert_array_equal(table.tone_envelope_offset_ns[rows, 0], 0.0)


@pytest.mark.parametrize(
    ("start", "plateau_start", "plateau_end", "duration"),
    [(2.0, 0.8, 10.8, 12.8), (0.1, 0.8, 20.8, 22.8)],
)
def test_plateau_support_uses_absolute_events(
    start, plateau_start, plateau_end, duration
):
    envelope = EnvelopeSpec(
        "sampled",
        np.array([0.0, 1j, 1j, 0.0]),
        np.array([0.0, plateau_start, plateau_end, duration]),
    )
    direct, constant = RecordingKernel("dop853"), RecordingKernel("t6")
    result = run(
        Engine(
            1,
            0,
            kernels=(
                RecordingKernel("t4"),
                direct,
                constant,
                RecordingKernel("dyson", accepts=False),
                RecordingKernel("floquet_period", accepts=False),
            ),
        ),
        source(),
        (
            replace(
                batch(1, duration=start + duration),
                tracks=(track(1, envelope, start=start, duration=duration),),
            ),
        ),
    )
    assert result.paths.segments_by_kernel == {"t4": 1, "dop853": 2, "t6": 1}
    table, rows, _ = constant.calls[0]
    np.testing.assert_allclose(table.start_ns[rows], start + plateau_start)
    np.testing.assert_allclose(table.duration_ns[rows], plateau_end - plateau_start)
    np.testing.assert_array_equal(table.tone_envelope_offset_ns[rows, 0], 0.0)
    offsets = sorted(
        float(table.tone_envelope_offset_ns[row, 0])
        for table, rows, _ in direct.calls
        for row in rows
    )
    np.testing.assert_allclose(offsets, [0.0, plateau_end])
