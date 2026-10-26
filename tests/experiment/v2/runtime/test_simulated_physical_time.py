"""Program acquisition facts expose shot time independently of simulator pacing."""

from collections.abc import Iterator
from typing import Literal

import numpy as np
import pytest
from zcu_tools.core.simulation import RoundRequest, RoundSignals
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.simulation import SimulationEnvironment, white_noise
from zcu_tools.experiment.v2.runtime import AcquireOptions, Schedule, SignalBuffer
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import (
    IntegratedRound,
    SimulationBinding,
    SimulationSoc,
)

from tests.experiment.v2.runtime._flow_cfg import make_flow_cfg


@pytest.fixture
def environment() -> Iterator[SimulationEnvironment]:
    env = white_noise(std_adc=0, seed=1)
    try:
        yield env
    finally:
        env.devices.close_all_devices()


@pytest.fixture
def program(environment: SimulationEnvironment) -> ModularProgramV2:
    return ModularProgramV2(
        environment.soccfg,
        modules=[
            PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=100,
                gain=0.1,
                waveform=ConstWaveformCfg(length=2),
                ro_length=0.2,
            ).build("readout")
        ],
        sweep=[("axis", 3)],
        reps=10,
        initial_delay=0,
        relax_delay=8,
    )


@pytest.mark.parametrize("time_scale", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("decimated", [False, True])
def test_simulated_shot_time_is_per_acquisition_and_independent_of_pacing(
    environment: SimulationEnvironment,
    program: ModularProgramV2,
    time_scale: float,
    decimated: bool,
) -> None:
    binding = SimulationBinding(
        environment.soc.binding.capture_snapshot,
        environment.soc.binding.execute,
        backend_version="synthetic",
        seed=1,
        time_scale=time_scale,
    )
    soc = SimulationSoc(environment.soccfg, binding, routes=((0, 0),))
    with Schedule(
        make_flow_cfg(), SignalBuffer((), dtype=np.complex128), scope=Scope("run")
    ) as schedule:
        builder = schedule.prog_builder(soc, environment.soccfg)
        runner = builder.run_program_decimated if decimated else builder.run_program
        for _ in range(2):
            runner(
                program,
                rounds=2,
                raw2signal_fn=lambda raw: np.asarray(raw).mean(),
            )
    assert schedule.outcome.status == "completed", schedule.stop.error
    # Three 2 us timelines plus 8 us relaxation, 10 reps, two rounds = 600 us.
    assert [
        facts.simulated_physical_s for facts in schedule.outcome.acquisitions
    ] == pytest.approx([0.0006, 0.0006], abs=0.2e-6)


@pytest.mark.parametrize(
    "failure, seconds", [("execute", 0.0009), ("validation", 0.0012)]
)
def test_shot_time_counts_returned_rounds_in_discarded_attempts(
    environment: SimulationEnvironment,
    program: ModularProgramV2,
    failure: Literal["execute", "validation"],
    seconds: float,
) -> None:
    calls = 0

    def execute(request: RoundRequest) -> IntegratedRound | RoundSignals:
        nonlocal calls
        calls += 1
        if calls == 2:
            if failure == "execute":
                raise RuntimeError("round did not finish execution")
            # Execute returned, so this round costs shot time even if validation fails.
            return RoundSignals(())
        return environment.soc.binding.execute(request)

    binding = SimulationBinding(
        environment.soc.binding.capture_snapshot,
        execute,
        backend_version="synthetic",
        seed=1,
        time_scale=0,
    )
    soc = SimulationSoc(environment.soccfg, binding, routes=((0, 0),))
    with Schedule(
        make_flow_cfg(), SignalBuffer((), dtype=np.complex128), scope=Scope("run")
    ) as schedule:
        schedule.prog_builder(soc, environment.soccfg).run_program(
            program,
            rounds=2,
            raw2signal_fn=lambda raw: np.asarray(raw).mean(),
            options=AcquireOptions(retry=1),
        )
    assert schedule.outcome.status == "completed", schedule.stop.error
    facts = schedule.outcome.acquisitions[0]
    assert facts.attempts == 2
    assert facts.completed_rounds == 2
    # Allow 10 ns of QICK timing quantization per shot, at most 120 shots.
    assert facts.simulated_physical_s == pytest.approx(seconds, abs=1.2e-6)
