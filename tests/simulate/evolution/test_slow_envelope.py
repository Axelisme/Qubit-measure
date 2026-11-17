"""Low-frequency plateau routing through the public engine interface."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec

from ._engine import RecordingKernel, batch, run, source, track
from .contract.accuracy_cases import coherent_source, drive_track, timeline
from .contract.synthetic_models import synthetic_model


@pytest.mark.parametrize("frequency", [0.001, 0.003, 0.01, -0.01, 0.0])
def test_low_frequency_plateau_uses_slow_envelope(frequency: float) -> None:
    kernels = tuple(
        RecordingKernel(name)
        for name in (
            "slow_envelope",
            "dyson",
            "single_tone_rwa",
            "floquet_period",
            "t6",
        )
    )
    pulse = replace(
        track(1, EnvelopeSpec("constant"), duration=50.0),
        frequency_ghz=np.array([frequency]),
    )
    timeline = replace(batch(1, duration=50.0), tracks=(pulse,))
    result = run(Engine(1, kernels=kernels), source(), (timeline,))
    assert result.paths.segments_by_kernel == {"slow_envelope": 1}


def test_single_use_dissipative_dc_avoids_dense_channel_build() -> None:
    model = synthetic_model(32)
    single = timeline(1, 100.0)
    pulse = drive_track(single, EnvelopeSpec("constant"), frequency=0.0, gain=0.02)
    single = replace(single, tracks=(pulse,))
    with closing(coherent_source(model)) as src, closing(Engine(1)) as engine:
        query = engine.inspect_dispatch(src, (single,))
        assert [row.family for row in query.rows] == ["dop853"]
        result = engine.evolve(src, (single,), telemetry=True)
    assert result.paths.segments_by_kernel == {"dop853": 1}


def test_dc_channel_build_is_amortized_only_over_this_acquires_exact_keys() -> None:
    model = synthetic_model(32)
    repeated = timeline(400, 100.0)
    pulse = replace(
        drive_track(repeated, EnvelopeSpec("constant"), frequency=0.0, gain=0.02),
        phase_rad=np.full(400, 0.17),
    )
    repeated = replace(repeated, tracks=(pulse,))
    distinct = replace(
        repeated, tracks=(replace(pulse, phase_rad=np.linspace(0.17, 0.27, 400)),)
    )
    single = timeline(1, 100.0)
    single = replace(
        single,
        tracks=(
            drive_track(single, EnvelopeSpec("constant"), frequency=0.0, gain=0.02),
        ),
    )
    with closing(coherent_source(model)) as src, closing(Engine(1)) as engine:
        shared_query = engine.inspect_dispatch(src, (repeated,))
        assert {row.family for row in shared_query.rows} == {"slow_envelope"}
        distinct_query = engine.inspect_dispatch(src, (distinct,))
        assert {row.family for row in distinct_query.rows} == {"dop853"}
        shared_result = engine.evolve(src, (repeated,), telemetry=True)
        assert shared_result.paths.segments_by_kernel == {"slow_envelope": 400}
        # Even a previous large inventory cannot subsidize the next channel build.
        later_query = engine.inspect_dispatch(src, (single,))
        assert [row.family for row in later_query.rows] == ["dop853"]
        later_result = engine.evolve(src, (single,), telemetry=True)
        assert later_result.paths.segments_by_kernel == {"dop853": 1}
    np.testing.assert_allclose(
        shared_result.densities,
        np.broadcast_to(later_result.densities, shared_result.densities.shape),
        atol=2e-3,
        rtol=0,
    )
