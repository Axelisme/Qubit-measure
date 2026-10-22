"""Composition root for six independent fluxonium/cavity models."""

from pathlib import Path

from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.resources.six_fluxonium import SixFluxoniumSettings


def six_fluxonium(configuration: SixFluxoniumSettings | str | Path) -> SimulationEnvironment:
    """Build six independent models, one global DC driver and the mock SoC.

    configuration is validated material or a YAML path relative to cwd. No
    hardware is opened. The caller owns device and backend cleanup after return.
    """
    raise NotImplementedError("six-fluxonium composition is not implemented")
