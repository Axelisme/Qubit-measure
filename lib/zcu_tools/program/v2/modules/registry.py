from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .pulse import PulseCfg


class PulseRegistry:
    """A registry to manage and share pulse configurations within a program."""

    def __init__(self):
        self._pulses: Dict[str, "PulseCfg"] = {}

    def register(self, name: str, cfg: "PulseCfg"):
        """Registers a pulse configuration."""
        if name in self._pulses:
            warnings.warn(
                f"Pulse '{name}' is being re-registered. Overwriting previous configuration."
            )
        self._pulses[name] = cfg

    def has(self, name: str) -> bool:
        """Checks if a pulse is registered."""
        return name in self._pulses

    def check_valid_mixer_freq(self, new_pulse_name: str, new_pulse_cfg: "PulseCfg"):
        """
        Checks if a new pulse's mixer frequency is consistent with other pulses
        on the same channel.
        """
        if "mixer_freq" not in new_pulse_cfg:
            return

        ch = new_pulse_cfg["ch"]
        mixer_freq = new_pulse_cfg["mixer_freq"]

        for name, cfg in self._pulses.items():
            if cfg["ch"] != ch:
                continue

            # All pulses on a channel being checked must have a mixer_freq if the new one does.
            if "mixer_freq" not in cfg:
                raise ValueError(
                    f"Pulse '{name}' on channel {ch} is missing 'mixer_freq', "
                    f"which is required for comparison with '{new_pulse_name}'."
                )

            if cfg["mixer_freq"] != mixer_freq:
                warnings.warn(
                    f"Mixer frequency mismatch on channel {ch}: "
                    f"Pulse '{name}' ({cfg['mixer_freq']} MHz) and "
                    f"Pulse '{new_pulse_name}' ({mixer_freq} MHz). "
                    "This may lead to unexpected behavior."
                )
