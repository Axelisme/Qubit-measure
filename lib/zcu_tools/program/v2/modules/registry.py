from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

from qick.asm_v2 import QickParam

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    ]

    def __init__(self) -> None:
        self._pulses: dict[str, tuple[str, PulseCfg]] = {}

    @property
    def count(self) -> int:
        """Number of unique pulse configurations registered."""
        return len(self._pulses)

    def calc_name(self, cfg: PulseCfg, *, ro_ch: int | None = None) -> str:
        """Return a stable hardware-definition name, including runtime ADC matching.

        cfg supplies generator/waveform settings. ro_ch is None for unpaired
        playback or an ADC channel; delays and description do not affect identity.
        """

        def sort_dict(d: Mapping[str, object]) -> OrderedDict[str, object]:
            sorted_dict: OrderedDict[str, object] = OrderedDict()
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

        cfg_dict = cfg.to_dict()
        filter_cfg = {
            k: cfg_dict.get(k) for k in PulseRegistry.HASH_KEYS if k in cfg_dict
        }
        filter_cfg["ro_ch"] = ro_ch
        cfg_json = json.dumps(sort_dict(filter_cfg), separators=(",", ":"))
        hash_name = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

        return f"pulse_{hash_name[:16]}"

    def register(self, name: str, cfg: PulseCfg, *, ro_ch: int | None = None) -> bool:
        """Capture cfg with runtime ADC matching; return False for a reused definition.

        name labels diagnostics. ro_ch has the same meaning as in calc_name.
        """
        pulse_name = self.calc_name(cfg, ro_ch=ro_ch)
        if pulse_name in self._pulses:
            logger.debug(
                "PulseRegistry: reuse '%s' (module=%s, ch=%s, phase=%s, gain=%s)",
                pulse_name,
                name,
                cfg.ch,
                cfg.phase,
                cfg.gain,
            )
            return False

        self._pulses[pulse_name] = (name, deepcopy(cfg))
        waveform = cfg.waveform
        logger.debug(
            "PulseRegistry: new '%s' (module=%s, ch=%s, freq=%s, phase=%s, gain=%s, "
            "style=%s, len=%s)",
            pulse_name,
            name,
            cfg.ch,
            cfg.freq,
            cfg.phase,
            cfg.gain,
            waveform.style,
            waveform.length,
        )

        return True
