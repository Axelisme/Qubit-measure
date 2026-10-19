"""Multi-timeline tone projection observed through injected kernel collaborators."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    FluxWaveform,
    ModelSource,
)
from zcu_tools.simulate.evolution.kernels import REGISTRY

from ._engine import RecordingKernel, batch, run, source, track
from .contract.kernel_suite import synthetic_model


@pytest.mark.parametrize("continuous", [False, True])
def test_sparse_tracks_pack_in_input_order_with_half_open_support(continuous):
    model = synthetic_model()
    drives = (model.drives[0], replace(model.drives[0], name="mux"))
    src = ModelSource(
        source().initial_density,
        lambda level: replace(
            model, level=level, drives=drives if level == (0,) else drives[::-1]
        ),
    )
    shaped = EnvelopeSpec(
        "sampled",
        np.array([0.2, 1j, 1j, -0.5j]),
        np.array([0.0, 2.0, 14.0, 16.0]),
    )
    envelopes = (shaped, EnvelopeSpec("constant"))
    tracks = tuple(
        replace(
            track(4, shaped, duration=30),
            port="drive" if i % 2 else "mux",
            start_ns=np.array([i * 4, i * 4 + 2, i * 4, i * 4 + 6], float),
            duration_ns=np.array([30, 0 if i % 3 == 0 else 24, 20, 34], float),
            amplitude=np.array([0.7, -0.4, 0 if i % 3 == 1 else 0.8, 1.0]),
            phase_rad=np.arange(4) * 0.13 + i * 0.07,
            frequency_ghz=np.array([0.1, -0.2, 0.3, 0.4]),
            envelopes=envelopes,
            envelope_index=np.array([0, 1, i % 2, 0]),
            active=np.array([True, i % 2 == 0, True, False]),
        )
        for i in range(24)
    )
    flux = FluxSchedule(
        np.arange(0, 13, 3),
        np.tile([0.0, 18.0, 50.0], 4),
        np.tile([[0], [1], [1]], (4, 1)),
    )
    waveform = FluxWaveform(
        np.arange(0, 13, 3),
        np.tile([0.0, 10.0, 40.0], 4),
        np.tile([[0.0], [0.001], [0.001]], (4, 1)),
    )
    timeline = replace(
        batch(4),
        points=np.array([8, 2, 5, 1]),
        end_ns=np.array([78.0, 66.0, 0.0, 40.0]),
        tracks=tracks,
        flux=flux,
        flux_waveform=waveform if continuous else None,
    )
    kernels = tuple(RecordingKernel(k.name) for k in REGISTRY)
    result = run(Engine(1, 0, kernels=kernels), src, (timeline,))
    observations = {
        (int(table.timeline[row]), float(table.start_ns[row])): (table, row, k.name)
        for k in kernels
        for table, rows, _ in k.calls
        for row in rows
    }
    assert {point for point, _ in observations} == {0, 1, 3}
    for (point, start), (table, row, family) in observations.items():
        end = start + table.duration_ns[row]
        playing = [
            t
            for t in tracks
            if t.active[point]
            and t.amplitude[point] != 0
            and t.start_ns[point] <= start < t.start_ns[point] + t.duration_ns[point]
            and end <= t.start_ns[point] + t.duration_ns[point]
        ]
        assert table.tones[row] == len(playing)
        changing = any(
            t.start_ns[point] < 18 < t.start_ns[point] + t.duration_ns[point]
            for t in playing
        )
        changing |= continuous and start < 10 and end > 0
        if playing and changing:
            assert family == "flux_dop853"
        elif playing:
            assert family != "flux_dop853"
        else:
            assert family == "t4"
        for slot, t in enumerate(playing):
            local = start - t.start_ns[point]
            local_end = end - t.start_ns[point]
            env = t.envelopes[t.envelope_index[point]]
            flat = env.kind == "sampled" and (
                2 <= local < 14
                and local_end <= 14
                or local >= 16
                and t.duration_ns[point] - 16 >= 10
            )
            value = 1j if local < 14 else -0.5j
            expected_env = envelopes[1] if flat else env
            assert (
                table.envelopes[table.tone_envelope[row, slot]].identity
                == expected_env.identity
            )
            assert table.tone_port[row, slot] == (
                [d.name for d in drives][::-1]
                if start >= 18
                else [d.name for d in drives]
            ).index(t.port)
            assert table.tone_frequency_ghz[row, slot] == t.frequency_ghz[point]
            assert table.tone_amplitude[row, slot] == t.amplitude[point] * (
                abs(value) if flat else 1
            )
            assert table.tone_theta_rad[row, slot] == 2 * np.pi * t.frequency_ghz[
                point
            ] * start + t.phase_rad[point] + (np.angle(value) if flat else 0)
            assert table.tone_envelope_offset_ns[row, slot] == (0 if flat else local)
        assert np.all(table.tone_port[row, len(playing) :] == -1)
    np.testing.assert_array_equal(result.points, timeline.points)
    np.testing.assert_array_equal(
        result.densities, np.repeat(src.initial_density[None], 4, axis=0)
    )
