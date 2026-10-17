"""Lab component models and explicit startup registration.

Importing this package never registers definitions. Call register_all once on
an empty caller-owned registry; duplicate bootstrap raises the registry's error.
"""

from zcu_tools.resources.entry.registry import ComponentRegistry, RoleSpec

from .base import (
    BuiltinComponent,
    Duration,
    Energy,
    Flux,
    Frequency,
    PumpPower,
    WiringSchema,
)
from .fluxonium import QubitSchema
from .jpa import JpaSchema
from .mux import MuxGroupSchema
from .qubit import QubitBaseSchema, QubitWiringSchema, params_for
from .resonator import ResonatorSchema
from .transmon import TransmonSchema

__all__ = [
    "BuiltinComponent",
    "Duration",
    "Energy",
    "Flux",
    "Frequency",
    "JpaSchema",
    "MuxGroupSchema",
    "PumpPower",
    "QubitBaseSchema",
    "QubitSchema",
    "QubitWiringSchema",
    "ResonatorSchema",
    "TransmonSchema",
    "WiringSchema",
    "params_for",
    "register_all",
]


def register_all(registry: ComponentRegistry) -> None:
    """Register definitions into the caller's registry; duplicates fail normally."""
    registry.register("resonator", ResonatorSchema)
    registry.register("group/mux", MuxGroupSchema)
    registry.register("device/current_source", BuiltinComponent)
    registry.register("amplifier/jpa", JpaSchema)
    registry.register("qubit/fluxonium", QubitSchema)
    registry.register("qubit/transmon", TransmonSchema)
    for name, kind in (
        ("qubit", "qubit/*"),
        ("resonator", "resonator"),
        ("qubit1", "qubit/*"),
        ("qubit2", "qubit/*"),
    ):
        registry.roles.register(name, RoleSpec(kind))
    registry.roles.configure(shorthand=("qubit", "resonator"), focus_kinds=("qubit/*",))
