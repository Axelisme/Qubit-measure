from __future__ import annotations

import logging
from copy import deepcopy
from collections import OrderedDict
import hashlib
import json
import warnings

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from .pulse import PulseCfg

logger = logging.getLogger(__name__)


class PulseRegistry:
    """A registry to manage and share pulse configurations within a program."""

    HASH_KEYS = [
        "waveform",
        "ch",
        "nqz",
        "freq",
        "phase",
        "gain",
        "mixer_freq",
        "mux_freqs",
        "mux_gains",
        "mux_phases",
        "mask",
        "outsel",
        "ro_ch",
    ]

    def __init__(self) -> None:
        self._pulses: dict[str, tuple[str, PulseCfg]] = {}

    def calc_name(self, cfg: PulseCfg) -> str:
        def sort_dict(d: dict) -> dict:
            sorted_dict = OrderedDict()
            for key in sorted(d.keys()):
                value = d[key]
                if isinstance(value, QickParam):
                    value = OrderedDict(
                        [
                            ("type", "QickParam"),
                            ("start", value.start),
                            ("spans", sort_dict(value.spans)),
                        ]
                    )
                if isinstance(value, dict):
                    value = sort_dict(value)

                sorted_dict[key] = value
            return sorted_dict

        filter_cfg = {k: cfg.get(k) for k in PulseRegistry.HASH_KEYS if k in cfg}
        cfg_json = json.dumps(sort_dict(filter_cfg), separators=(",", ":"))
        hash_name = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

        return f"pulse_{hash_name[:16]}"

    def register(self, name: str, cfg: PulseCfg) -> bool:
        """Registers a pulse configuration. returns True if the pulse is newly registered, False if it already exists."""
        pulse_name = self.calc_name(cfg)
        if pulse_name in self._pulses:
            logger.debug(
                "PulseRegistry: reuse '%s' (module=%s, ch=%s, phase=%s, gain=%s)",
                pulse_name, name, cfg["ch"], cfg["phase"], cfg["gain"],
            )
            return False

        self._pulses[pulse_name] = (name, deepcopy(cfg))
        waveform = cfg["waveform"]
        waveform_style = waveform["style"] if isinstance(waveform, dict) else "?"
        logger.debug(
            "PulseRegistry: new '%s' (module=%s, ch=%s, freq=%s, phase=%s, gain=%s, "
            "waveform_style=%s) [total=%d]",
            pulse_name, name, cfg["ch"], cfg["freq"],
            cfg["phase"], cfg["gain"],
            waveform_style,
            len(self._pulses),
        )

        return True

    def check_valid_mixer_freq(self, name: str, pulse_cfg: PulseCfg) -> None:
        """
        Checks if a new pulse's mixer frequency is consistent with other pulses
        on the same channel.
        """
        ch = pulse_cfg["ch"]
        has_mixer_freq = "mixer_freq" in pulse_cfg
        mixer_freq = pulse_cfg.get("mixer_freq")

        for p_name, p_cfg in self._pulses.values():
            if p_cfg["ch"] != ch:
                continue

            registered_has_mixer_freq = "mixer_freq" in p_cfg
            registered_mixer_freq = p_cfg.get("mixer_freq")

            if has_mixer_freq != registered_has_mixer_freq:
                raise ValueError(
                    f"Pulse '{p_name}' on channel {ch} has "
                    f"{'no ' if not registered_has_mixer_freq else ''}'mixer_freq', "
                    f"which is required for comparison with '{name}'."
                )

            if has_mixer_freq and registered_mixer_freq != mixer_freq:
                warnings.warn(
                    f"Mixer frequency mismatch on channel {ch}: "
                    f"Pulse '{p_name}' ({registered_mixer_freq} MHz) and "
                    f"Pulse '{name}' ({mixer_freq} MHz). "
                    "This may lead to unexpected behavior."
                )
