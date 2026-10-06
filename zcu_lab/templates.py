"""Startup-only template composition; retained across experiment reloads."""

from __future__ import annotations

from collections.abc import Callable

from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES, ProgramShape
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.app.measure.template_catalog import (
    TemplateCatalog,
    TemplateEntry,
    TemplateItemKind,
)
from zcu_tools.gui.cfg import (
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES


def _blank_value_factory(shape: ProgramShape) -> Callable[[SessionEnv], ReferenceValue]:
    def _make(_ctx: SessionEnv) -> ReferenceValue:
        value = make_default_value(shape.make_spec(MAIN_PROGRAM_SPEC_POLICY))
        return ReferenceValue(make_custom_reference_key(shape.discriminator), value)

    return _make


def _blank_entries() -> list[TemplateEntry]:
    entries: list[TemplateEntry] = []
    factory: Callable[[SessionEnv], ReferenceValue]
    for shape in PROGRAM_SHAPES.modules():
        factory = _blank_value_factory(shape)
        entries.append(
            TemplateEntry(
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
            TemplateEntry(
                f"{shape.discriminator}:blank",
                f"Blank: {shape.discriminator}",
                "waveform",
                lambda shape=shape: shape.make_spec(MAIN_PROGRAM_SPEC_POLICY),
                factory,
            )
        )
    return entries


# Insertion order is dropdown order. Blank factories never adopt library entries.
_CATALOG_TEMPLATES: list[tuple[str, str, TemplateItemKind, str]] = [
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

TEMPLATE_ENTRIES: list[TemplateEntry] = [
    TemplateEntry(
        template_id,
        label,
        kind,
        ROLE_FACTORIES[template_id].shape,
        ROLE_FACTORIES[template_id].blank,
        default_name,
    )
    for template_id, label, kind, default_name in _CATALOG_TEMPLATES
]

ALL_TEMPLATE_ENTRIES: list[TemplateEntry] = [*TEMPLATE_ENTRIES, *_blank_entries()]


def register_all_templates(catalog: TemplateCatalog) -> None:
    """Append named and blank templates in dropdown order to the caller catalog.

    Duplicate IDs or invalid shapes raise the catalog registration error. Call
    only at startup, not on reload; this function does not clear prior entries.
    """
    for entry in ALL_TEMPLATE_ENTRIES:
        catalog.register(entry)
