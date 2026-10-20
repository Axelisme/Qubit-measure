"""Explicit Python environments, shared by GUI startup and Notebook callers."""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from qick import QickConfig

from zcu_tools.core.simulation import (
    RoundRequest,
    RoundSignals,
)
from zcu_tools.device import DeviceManager, FakeDevice
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.simulation import (
    IntegratedRound,
    SimulationBinding,
    SimulationSoc,
)
from zcu_tools.simulate.white_noise import WhiteNoise

from .warmup import EnvironmentWarmup, WarmupStatus


@dataclass(frozen=True)
class SimulationEnvironment:
    """Fully bound resources; successful builder return transfers cleanup to host.

    devices is the sole manager of its drivers. GUI injects this exact instance;
    Notebook owners close it themselves. description is public, not model truth.
    Builder failures must close their resources before raising; host installation
    failures close these devices, preserving both startup and cleanup exceptions.
    warmup optionally exposes process-shared numeric preparation; None disables
    startup work. It owns no live resources. Poll warmup.status for progress or
    call wait_ready before starting acquisition timing. Explicitly addressed
    startup FakeDevices define the modeled addresses for create_device. Their
    identities live only in FakeDevice's process-wide class container; this
    environment retains only address metadata, not a second instance registry.
    """

    soc: SimulationSoc
    soccfg: QickConfig
    devices: DeviceManager
    description: str
    warmup: EnvironmentWarmup | None = None
    _fake_addresses: frozenset[str] = field(init=False, repr=False)

    def create_device(self, type_name: str, address: str) -> FakeDevice:
        """Retrieve a modeled FakeDevice by its explicit address, without reset.

        Only type FakeDevice and an addressed startup source are supported.
        Unsupported type/address combinations raise ValueError, never open real
        hardware and never create an inert source. Returned identities retain
        settings and model bindings across disconnects and name changes. The
        caller owns registry admission; an already connected identity cannot be
        registered under a second name (ADR-0076).
        """
        if type_name != "FakeDevice" or address not in self._fake_addresses:
            raise ValueError(
                f"Simulation cannot bind device type {type_name!r} at {address!r}: "
                "use an explicitly addressed startup FakeDevice"
            )
        return FakeDevice(address=address)

    def wait_ready(self, timeout: float | None = None) -> WarmupStatus | None:
        """Wait before starting caller-owned acquisition timing.

        Return None when startup preparation is disabled, otherwise its terminal
        status. Failure is reported in the status and allows on-demand compiling.
        timeout is seconds or None; TimeoutError does not cancel preparation.
        warmup is the optional shared preparation controller; polling its status
        never blocks. The environment does not own a second worker or future.
        """
        return None if self.warmup is None else self.warmup.wait_ready(timeout)

    def __post_init__(self) -> None:
        if self.soc.get_cfg() != self.soccfg.get_cfg():
            raise ValueError("environment soc and soccfg must use matching profiles")
        retained = FakeDevice.instance_addresses()
        object.__setattr__(
            self,
            "_fake_addresses",
            frozenset(
                device.address
                for device in self.devices.get_all_devices().values()
                if isinstance(device, FakeDevice)
                and device.address in retained
                and FakeDevice(address=device.address) is device
            ),
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

    def execute(request: RoundRequest) -> IntegratedRound | RoundSignals:
        result = source.execute(request)
        if isinstance(result, RoundSignals):
            return result
        statistics, raw = result
        return IntegratedRound(statistics, raw)

    binding = SimulationBinding(
        source.snapshot, execute, backend_version="white_noise-v2", seed=seed
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
