"""Bind injected numeric sources to an already assembled environment's transport."""

from dataclasses import replace

import pytest
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.program.v2.simulation import SimulationBinding, SimulationCallbacks
from zcu_tools.simulators.ftf.config import FTFSettings
from zcu_tools.simulators.ftf.environment import FTFSource, ftf
from zcu_tools.simulators.single_fluxonium.environment.runtime import (
    SingleFluxoniumRuntimeOptions,
    SingleFluxoniumSource,
    single_fluxonium,
)
from zcu_tools.simulators.telemetry import RoutingTelemetry

from tests.simulators._engine import EngineDouble


@pytest.fixture
def six_engines(monkeypatch: pytest.MonkeyPatch):
    """Keep six real independent sources, replacing only their numeric Engines."""
    from zcu_tools.simulators.six_fluxonium.environment import runtime

    created: list[EngineDouble] = []

    def engine_factory(*args, **kwargs):
        engine = EngineDouble(*args, **kwargs)
        created.append(engine)
        return engine

    # SixFluxoniumSource has no public Engine injection point.
    monkeypatch.setattr(runtime, "Engine", engine_factory)
    return created


def bind_source(
    env: SimulationEnvironment,
    source: SingleFluxoniumSource | FTFSource,
    seed: int | None,
) -> SimulationEnvironment:
    """Keep builder-owned devices/routes and replace only the numeric callbacks.

    The original backend has not acquired. Close it before installing the source
    with its injected Engine. The returned environment owns the new source.
    """
    if env.close_backend is not None:
        env.close_backend()
    binding = SimulationBinding(
        source.snapshot,
        source.execute,
        backend_version="environment-test",
        seed=seed,
        callbacks=SimulationCallbacks(
            on_begin=lambda: source.begin_acquire(seed=binding.root_seed),
            on_end=source.finish_acquire if isinstance(source, FTFSource) else None,
        ),
    )
    env.soc.binding = binding
    return replace(env, close_backend=source.close)


def single_environment(configuration, *, soccfg, engine=None, **kwargs):
    """Use the public source injection seam without changing builder wiring."""
    env = single_fluxonium(configuration, soccfg=soccfg, **kwargs)
    source = SingleFluxoniumSource(
        configuration,
        env.devices,
        runtime=SingleFluxoniumRuntimeOptions(
            engine=EngineDouble(families=kwargs.get("families"))
            if engine is None
            else engine,
            history=kwargs.get("history"),
        ),
    )
    return bind_source(env, source, configuration.numerics.seed)


def ftf_environment(
    configuration: FTFSettings,
    *,
    soccfg,
    engine=None,
    telemetry: RoutingTelemetry | None = None,
    **kwargs,
):
    """Bind an FTFSource with a non-evolving collaborator to existing routes."""
    env = ftf(configuration, soccfg=soccfg, **kwargs)
    source = FTFSource(
        configuration,
        env.devices,
        soccfg=soccfg,
        telemetry=telemetry,
        engine=EngineDouble(families=kwargs.get("families"))
        if engine is None
        else engine,
        history=kwargs.get("history"),
    )
    return bind_source(env, source, configuration.numerics.seed)
