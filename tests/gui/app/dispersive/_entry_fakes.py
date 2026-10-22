"""Synthetic registered components for dispersive entry integration tests."""

from pydantic import ConfigDict
from zcu_tools.resources.entry import ComponentSchema


class Fluxonium(ComponentSchema):
    """Fluxonium with only energies declared; fit outputs stay in artifacts."""

    model_config = ConfigDict(extra="forbid")
    EJ: float | None = None
    EC: float | None = None
    EL: float | None = None


class CoupledFluxonium(Fluxonium):
    """A kind that explicitly accepts coupling, but not bare_rf."""

    g: float | None = None


class Resonator(ComponentSchema):
    """Readout frequency in MHz, as declared by the component contract."""

    model_config = ConfigDict(extra="forbid")
    freq: float | None = None
