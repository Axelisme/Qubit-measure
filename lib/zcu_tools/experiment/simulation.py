"""Explicit Python environments, shared by GUI startup and Notebook callers."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from qick import QickConfig

from zcu_tools.core.simulation import (
    ChannelSignals,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
)
from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.simulation import SimulationBinding, SimulationSoc


@dataclass(frozen=True)
class SimulationEnvironment:
    """Fully bound resources; successful builder return transfers cleanup to host.

    devices is the sole manager of its drivers. GUI injects this exact instance;
    Notebook owners close it themselves. description is public, not model truth.
    Builder failures must close their resources before raising; host installation
    failures close these devices, preserving both startup and cleanup exceptions.
    """

    soc: SimulationSoc
    soccfg: QickConfig
    devices: DeviceManager
    description: str


@dataclass(frozen=True)
class WhiteNoise:
    """Independent zero-mean I/Q Gaussian, std_adc in per-sample ADC units."""

    std_adc: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.std_adc) or self.std_adc < 0:
            raise ValueError("std_adc must be finite and nonnegative")

    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot("white_noise-v1", f"std_adc={self.std_adc!r}")

    def execute(self, request: RoundRequest) -> RoundSignals:
        acquire = request.acquire
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    [acquire.root_seed, acquire.sequence, request.index]
                )
            )
        )
        channels = []
        for window in acquire.readouts:
            shape = (acquire.executions, window.reads)
            if acquire.mode == "decimated":
                shape += (window.samples, 2)
                std = self.std_adc
            else:
                shape += (2,)
                std = self.std_adc * np.sqrt(window.samples)
            channels.append(ChannelSignals(window.channel, rng.normal(0, std, shape)))
        return RoundSignals(tuple(channels))


def white_noise(
    *,
    std_adc: float = 1.0,
    seed: int | None = None,
    n_gens: int = 2,
    n_readouts: int = 1,
    routes: tuple[tuple[int, int], ...] | None = None,
) -> SimulationEnvironment:
    """Build an explicit noise environment using the fixed offline QICK profile.

    Defaults bind each ADC to the same-index generator (requires enough gens).
    routes overrides wiring only, not clocks, DDS, channel types or memory sizes.
    No physical pulse response is modeled. Invalid amplitude/seed/routes raise.
    """
    source = WhiteNoise(std_adc)
    soccfg = make_mock_soccfg(n_gens, n_readouts)
    binding = SimulationBinding(
        source.snapshot, source.execute, backend_version="white_noise-v1", seed=seed
    )
    wiring = tuple((ch, ch) for ch in range(n_readouts)) if routes is None else routes
    soc = SimulationSoc(soccfg, binding, routes=wiring)
    return SimulationEnvironment(
        soc, soccfg, DeviceManager(), "white_noise: independent Gaussian ADC samples"
    )


def load_environment(path: str | Path) -> SimulationEnvironment:
    """Execute trusted Python once and invoke its zero-argument builder once.

    No sandbox, fallback, discovery, or hot reload. Builders own pre-return
    failures; the caller owns the returned manager even if installation fails.
    """
    spec = importlib.util.spec_from_file_location(
        f"_zcu_environment_{uuid4().hex}", Path(path).resolve()
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load environment: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "build_environment", None)
    if not callable(builder):
        raise TypeError("environment must define build_environment()")
    result = builder()
    if not isinstance(result, SimulationEnvironment):
        raise TypeError("build_environment() must return SimulationEnvironment")
    return result
