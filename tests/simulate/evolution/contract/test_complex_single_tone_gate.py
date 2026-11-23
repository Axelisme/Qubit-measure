"""Complex sampled single-tone gates at the public Engine seam."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, NoValidKernel

from .accuracy_cases import coherent_source, drive_track, linear_drive_model, timeline


@pytest.mark.sim
@pytest.mark.parametrize(
    "samples",
    [
        pytest.param([0.85 - 0.15j, 0.925 - 0.075j, 1.0], id="slope"),
        pytest.param([1 + 0.035j, 1.0, 1 + 0.035j], id="curvature"),
    ],
)
def test_rough_complex_envelope_is_rejected(samples: list[complex]) -> None:
    # The slope case exceeds the limit only in modulus; the curvature case
    # keeps slopes within range, with a sharp turn in the Q component.
    envelope = EnvelopeSpec("sampled", np.array(samples), np.array([0.0, 4.0, 8.0]))
    batch = timeline(1, 8.0)
    batch = replace(batch, tracks=(drive_track(batch, envelope, frequency=0.6),))
    with (
        closing(coherent_source(linear_drive_model())) as source,
        closing(Engine(1, families=("single_tone_gate",), backend="cpu")) as gate,
        pytest.raises(NoValidKernel) as failure,
    ):
        gate.evolve(source, (batch,), budget=2e-3)
    assert failure.value.attempted == ("single_tone_gate",)
