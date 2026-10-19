"""Small address-bound source for simulation composition contracts."""

from math import prod

import numpy as np
from zcu_tools.core.simulation import (
    ChannelSignals,
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
)
from zcu_tools.device import FakeDevice
from zcu_tools.experiment.simulation import SimulationEnvironment, white_noise


class BiasSource:
    """Capture startup driver identities, as an author-defined model can do."""

    def __init__(self, drivers: dict[str, FakeDevice]) -> None:
        self.drivers = drivers

    def snapshot(self) -> EnvironmentSnapshot:
        """Capture live values independently of session registry membership."""
        return EnvironmentSnapshot(
            "synthetic-bias-v1",
            "synthetic",
            tuple((name, device.get_value()) for name, device in self.drivers.items()),
        )

    def execute(self, request: RoundRequest) -> RoundSignals:
        """Emit a constant ADC level equal to the sum of bound device values."""
        level = sum(value for _name, value in request.snapshot.values)
        return RoundSignals(
            tuple(
                ChannelSignals(
                    window.channel,
                    np.full(
                        (
                            prod(request.acquire.loop_dims),
                            window.reads,
                            window.samples,
                            2,
                        ),
                        level,
                    ),
                )
                for window in request.acquire.readouts
            )
        )


def bias_environment() -> SimulationEnvironment:
    """Build two explicitly addressed sources coupled to a synthetic model."""
    noise = white_noise(std_adc=0)
    drivers = {}
    for name in ("bias", "spare"):
        driver = FakeDevice(address=name, fast_mode=True, value_unit="V")
        noise.devices.register_device(name, driver)
        drivers[name] = driver
    source = BiasSource(drivers)
    noise.soc.binding.capture_snapshot = source.snapshot
    noise.soc.binding.execute = source.execute
    return SimulationEnvironment(
        noise.soc, noise.soccfg, noise.devices, "synthetic address-bound bias"
    )
