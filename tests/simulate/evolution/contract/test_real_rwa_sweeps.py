"""Small real-model observations at the existing public guard-sweep seams.

These cases do not change admission or replace the two-level error curves.
The exploratory grid is deliberately not part of pytest.
"""

from contextlib import closing

import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    NoValidKernel,
    single_tone_rwa_diagnostic,
)

from .interaction_reference import interaction_reference
from .real_rwa_sweep_cases import real_rwa_case, real_rwa_source
from .reference import density_error


@pytest.mark.sim
@pytest.mark.parametrize(
    "detuning,gain,duration", [(-0.1, 0.025, 8.0), (0.0, 0.025, 20.0), (0.1, 0.05, 8.0)]
)
def test_real_constant_sweep_density(
    detuning: float, gain: float, duration: float
) -> None:
    with closing(real_rwa_source()) as source:
        case = real_rwa_case(
            source, detuning_ghz=detuning, gain=gain, duration_ns=duration
        )
        track = case.batch.tracks[0]
        diagnostic = single_tone_rwa_diagnostic(
            source.model((0,)),
            track.port,
            float(track.frequency_ghz[0]),
            gain,
            duration,
        )
        assert diagnostic.accepted
        reference = interaction_reference(source, case.batch)
        with closing(Engine(1, families=("single_tone_rwa", "t4"))) as engine:
            result = engine.evolve(source, (case.batch,))
        assert dict(result.paths.segments_by_kernel) == {"single_tone_rwa": 1}
        assert density_error(result.densities, reference.density) <= 2e-3


@pytest.mark.sim
def test_real_constant_sweep_rejects_strong_drive() -> None:
    with closing(real_rwa_source()) as source:
        case = real_rwa_case(source, detuning_ghz=0.0, gain=0.8, duration_ns=20.0)
        track = case.batch.tracks[0]
        diagnostic = single_tone_rwa_diagnostic(
            source.model((0,)), track.port, float(track.frequency_ghz[0]), 0.8, 20.0
        )
        assert not diagnostic.accepted
        with (
            closing(Engine(1, families=("single_tone_rwa", "t4"))) as engine,
            pytest.raises(NoValidKernel, match="domain plateau"),
        ):
            engine.evolve(source, (case.batch,))
