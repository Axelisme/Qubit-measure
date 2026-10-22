"""Synthetic component schemas shared by fluxdep analysis tests."""

from zcu_tools.resources.entry.schema import ComponentSchema


class SyntheticFluxonium(ComponentSchema):
    """Analysis target declaring kind and EJ/EC/EL energies in GHz, but no flux fields."""

    kind: str = "qubit/fluxonium"
    EJ: float | None = None
    EC: float | None = None
    EL: float | None = None
