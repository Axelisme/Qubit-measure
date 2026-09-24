"""Startup-only role composition; retained across experiment reloads."""

from __future__ import annotations

from collections.abc import Callable

from zcu_tools.gui.app.main.adapter import ExpContext
from zcu_tools.gui.app.main.role_catalog import RoleCatalog, RoleEntry, RoleItemKind
from zcu_tools.gui.app.main.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.cfg import (
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.gui.measure_cfg import PROGRAM_SHAPES, ProgramShape

from .adapters._support import ROLE_FACTORIES


def _blank_value_factory(shape: ProgramShape) -> Callable[[ExpContext], ReferenceValue]:
    def _make(_ctx: ExpContext) -> ReferenceValue:
        value = make_default_value(shape.make_spec(MAIN_PROGRAM_SPEC_POLICY))
        return ReferenceValue(make_custom_reference_key(shape.discriminator), value)

    return _make


def _blank_entries() -> list[RoleEntry]:
    entries: list[RoleEntry] = []
    factory: Callable[[ExpContext], ReferenceValue]
    for shape in PROGRAM_SHAPES.modules():
        factory = _blank_value_factory(shape)
        entries.append(
            RoleEntry(
                f"{shape.discriminator}:blank",
                f"Blank: {shape.discriminator}",
                "module",
                lambda shape=shape: shape.make_spec(MAIN_PROGRAM_SPEC_POLICY),
                factory,
            )
        )
    for shape in PROGRAM_SHAPES.waveforms():
        factory = _blank_value_factory(shape)
        entries.append(
            RoleEntry(
                f"{shape.discriminator}:blank",
                f"Blank: {shape.discriminator}",
                "waveform",
                lambda shape=shape: shape.make_spec(MAIN_PROGRAM_SPEC_POLICY),
                factory,
            )
        )
    return entries


# Insertion order is dropdown order. Blank factories never adopt library entries.
_CATALOG_ROLES: list[tuple[str, str, RoleItemKind, str]] = [
    ("res_probe", "Resonator probe", "module", "readout_rf"),
    ("readout", "Pulse readout", "module", "readout_rf"),
    ("readout_dpm", "Optimized readout (DPM)", "module", "readout_dpm"),
    ("direct_readout", "Direct readout", "module", "readout_direct"),
    ("qub_probe", "Qubit probe pulse", "module", "qub_pulse"),
    ("pi_pulse", "Pi pulse", "module", "pi_amp"),
    ("pi2_pulse", "Pi/2 pulse", "module", "pi2_amp"),
    ("none_reset", "No reset", "module", "reset_none"),
    ("reset", "Pulse reset", "module", "reset_10"),
    ("two_pulse_reset", "Two-pulse reset", "module", "reset_120"),
    ("bath_reset", "Bath reset", "module", "reset_bath"),
    ("qub_waveform", "Qubit drive waveform", "waveform", "qub_flat"),
    ("res_waveform", "Res-probe waveform", "waveform", "ro_waveform"),
]

ROLE_ENTRIES: list[RoleEntry] = [
    RoleEntry(
        role_id,
        label,
        kind,
        ROLE_FACTORIES[role_id].shape,
        ROLE_FACTORIES[role_id].blank,
        default_name,
    )
    for role_id, label, kind, default_name in _CATALOG_ROLES
]

ALL_ROLE_ENTRIES: list[RoleEntry] = [*ROLE_ENTRIES, *_blank_entries()]


def register_all_roles(catalog: RoleCatalog) -> None:
    for entry in ALL_ROLE_ENTRIES:
        catalog.register(entry)
