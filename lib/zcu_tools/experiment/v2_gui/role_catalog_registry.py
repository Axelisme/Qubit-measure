"""Populate the GUI ``RoleCatalog`` with experiment-role templates.

Mirrors ``registry.register_all``: the gui defines the ``RoleCatalog`` interface,
this module (in the experiment layer, which may import gui) fills it with the
eval-aware ``make_<role>_default`` builders. Wired once at startup in ``app.py``.

Uses the blank ``_default`` factories (not the ``_ref`` variants): creating from
a role means seeding a fresh blank with md-linked defaults, not referencing an
existing library entry.
"""

from __future__ import annotations

from zcu_tools.gui.role_catalog import RoleCatalog, RoleEntry

from .adapters.shared import (
    make_bath_reset_default,
    make_direct_readout_default,
    make_none_reset_default,
    make_pi2_pulse_default,
    make_pi_pulse_default,
    make_pulse_readout_default,
    make_pulse_reset_default,
    make_qub_probe_default,
    make_qub_waveform_default,
    make_res_probe_default,
    make_res_waveform_default,
    make_two_pulse_reset_default,
)

# Insertion order = dropdown order. Readout/probe first, then pulses, then
# resets, then waveforms.
ROLE_ENTRIES: list[RoleEntry] = [
    RoleEntry("res_probe", "Resonator probe", "module", make_res_probe_default),
    RoleEntry("pulse_readout", "Pulse readout", "module", make_pulse_readout_default),
    RoleEntry(
        "direct_readout", "Direct readout", "module", make_direct_readout_default
    ),
    RoleEntry("qub_probe", "Qubit probe pulse", "module", make_qub_probe_default),
    RoleEntry("pi_pulse", "Pi pulse", "module", make_pi_pulse_default),
    RoleEntry("pi2_pulse", "Pi/2 pulse", "module", make_pi2_pulse_default),
    RoleEntry("none_reset", "No reset", "module", make_none_reset_default),
    RoleEntry("pulse_reset", "Pulse reset", "module", make_pulse_reset_default),
    RoleEntry(
        "two_pulse_reset", "Two-pulse reset", "module", make_two_pulse_reset_default
    ),
    RoleEntry("bath_reset", "Bath reset", "module", make_bath_reset_default),
    RoleEntry(
        "qub_waveform", "Qubit drive waveform", "waveform", make_qub_waveform_default
    ),
    RoleEntry(
        "res_waveform", "Res-probe waveform", "waveform", make_res_waveform_default
    ),
]


def register_all_roles(catalog: RoleCatalog) -> None:
    for entry in ROLE_ENTRIES:
        catalog.register(entry)
