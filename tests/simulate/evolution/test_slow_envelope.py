"""Low-frequency plateau routing through the public engine interface."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec

from ._engine import RecordingKernel, batch, run, source, track


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
