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
from .mux import CoilFluxCalibration, MuxGroupSchema, MuxWiringSchema
from .parameter_upgrades import PARAMETER_SCHEMA_POLICY
from .qubit import QubitBaseSchema, QubitFluxCalibration, QubitWiringSchema
from .resonator import ResonatorSchema
from .transmon import TransmonSchema

__all__ = [
    "BuiltinComponent",
    "CoilFluxCalibration",
    "Duration",
    "Energy",
    "Flux",
    "Frequency",
    "JpaSchema",
    "MuxGroupSchema",
    "MuxWiringSchema",
    "PumpPower",
    "QubitBaseSchema",
    "QubitFluxCalibration",
    "QubitSchema",
    "QubitWiringSchema",
    "ResonatorSchema",
    "TransmonSchema",
    "WiringSchema",
    "register_all",
]


def register_all(registry: ComponentRegistry) -> None:
    """Register lab models, roles and whole-document schema policy.

    Bootstrap once on an empty registry; duplicate model names fail normally.
    The policy declares version 1 and the explicit offline upgrade command.
    Registration performs no file I/O and never upgrades saved parameters.
    """
    registry.configure_schema(PARAMETER_SCHEMA_POLICY)
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
