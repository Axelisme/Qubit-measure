"""Explicit Python environments, shared by GUI startup and Notebook callers."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from qick import QickConfig

from zcu_tools.core.simulation import (
    ChannelSignals,
    ChannelStatistics,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
    RoundStatistics,
    SamplingIdentity,
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

    def __post_init__(self) -> None:
        if self.soc.get_cfg() != self.soccfg.get_cfg():
            raise ValueError("environment soc and soccfg must use matching profiles")


@dataclass(frozen=True)
class WhiteNoise:
    """Independent zero-mean I/Q Gaussian, std_adc in per-sample ADC units."""

    std_adc: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.std_adc) or self.std_adc < 0:
            raise ValueError("std_adc must be finite and nonnegative")

    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot("white_noise-v1", f"std_adc={self.std_adc!r}")

    def execute(self, request: RoundRequest) -> RoundStatistics | RoundSignals:
        """Draw compact integrated totals or per-execution decimated ADC samples.

        Integrated totals include receiver and rounding variance. Zero noise
        remains exactly zero. Deferred shots use a separate reproducible stream.
        """
        if request.acquire.mode == "integrated":
            return self._integrated_statistics(request)
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
            shape = (acquire.executions, window.reads, window.samples, 2)
            channels.append(
                ChannelSignals(window.channel, rng.normal(0, self.std_adc, shape))
            )
        return RoundSignals(tuple(channels))

    def _integrated_statistics(self, request: RoundRequest) -> RoundStatistics:
        acquire = request.acquire
        point_dims = tuple(
            n for axis, n in enumerate(acquire.loop_dims) if axis != acquire.avg_level
        )
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    [acquire.root_seed, acquire.sequence, request.index, 0]
                )
            )
        )
        channels = []
        for window in acquire.readouts:
            variance = self.std_adc**2 * window.samples
            # Zero-noise environments promise exact zero, without rounding noise.
            total_variance = acquire.reps * (variance + 1 / 12) if self.std_adc else 0
            shape = (*point_dims, window.reads, 2)
            channels.append(
                ChannelStatistics(
                    window.channel,
                    rng.normal(0, np.sqrt(total_variance), shape),
                    np.zeros((*point_dims, window.reads, 1, 2)),
                    np.eye(2) * variance,
                )
            )
        return RoundStatistics(
            np.full((*point_dims, 1), acquire.reps, dtype=np.int64),
            tuple(channels),
            SamplingIdentity(acquire.root_seed, (acquire.sequence, request.index, 1)),
        )


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
        source.snapshot, source.execute, backend_version="white_noise-v2", seed=seed
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
    # Ordinary Python import identity is needed by postponed annotations and
    # dataclasses. Retain successful author modules for later type resolution.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        builder = getattr(module, "build_environment", None)
        if not callable(builder):
            raise TypeError("environment must define build_environment()")
        result = builder()
        if not isinstance(result, SimulationEnvironment):
            raise TypeError("build_environment() must return SimulationEnvironment")
        return result
    except Exception:
        del sys.modules[spec.name]
        raise
