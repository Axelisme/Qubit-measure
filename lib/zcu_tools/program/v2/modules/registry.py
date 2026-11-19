from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

from qick.asm_v2 import QickParam

from ..table_param import TableParam
from .waveform import (
    ArbWaveformCfg,
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    InMemoryArbWaveformCfg,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qick.asm_v2 import QickProgramV2

    from .pulse import PulseCfg
    from .waveform import QickWaveformKwargs

logger = logging.getLogger(__name__)


class PulseRegistry:
    """Share pulse definitions and sampled envelopes within one program build."""

    HASH_KEYS = [
        "waveform",
        "ch",
        "nqz",
        "freq",
        "phase",
        "gain",
        "mixer_freq",
    ]

    def __init__(self) -> None:
        self._pulses: dict[str, tuple[str, PulseCfg]] = {}
        self._envelopes: dict[tuple[int, bool, str], str] = {}

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
                if isinstance(value, TableParam):
                    value = OrderedDict(
                        [
                            ("type", "TableParam"),
                            ("loop", value.loop),
                            ("values", value.values),
                        ]
                    )
                elif isinstance(value, QickParam):
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

        if isinstance(cfg.waveform, InMemoryArbWaveformCfg):
            cfg_dict = cfg.model_dump(
                mode="python", exclude={"waveform"}, exclude_none=True
            )
            cfg_dict["waveform"] = _memory_envelope_definition(cfg.waveform)
        else:
            cfg_dict = cfg.to_dict()
        filter_cfg = {
            k: cfg_dict.get(k) for k in PulseRegistry.HASH_KEYS if k in cfg_dict
        }
        filter_cfg["ro_ch"] = ro_ch
        cfg_json = json.dumps(sort_dict(filter_cfg), separators=(",", ":"))
        hash_name = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

        return f"pulse_{hash_name[:16]}"

    def declare_waveform(
        self, prog: QickProgramV2, cfg: PulseCfg, *, pulse_id: str
    ) -> QickWaveformKwargs:
        """Return QICK pulse kwargs after declaring or reusing envelope samples.

        prog owns this registry and must have declared cfg.ch. cfg.waveform
        supplies the shape and playback length. pulse_id names a newly stored
        envelope; later pulses with the same sample definition reuse its name.
        Const stores no samples. QICK and asset errors propagate.
        """
        waveform_cfg = cfg.waveform
        name = f"{pulse_id}_waveform"
        if isinstance(waveform_cfg, ConstWaveformCfg):
            return waveform_cfg.build(name).to_wav_kwargs()

        # Flat-top ramps round to even clocks; the plateau adds no samples.
        flat_top = isinstance(waveform_cfg, FlatTopWaveformCfg)
        shape = waveform_cfg.raise_waveform if flat_top else waveform_cfg
        if isinstance(shape, InMemoryArbWaveformCfg):
            definition = _memory_envelope_definition(shape)
        elif isinstance(shape, ArbWaveformCfg):
            # File-backed playback reads the path, not the asset label.
            definition = {"style": shape.style, "path": shape.path}
        else:
            definition = shape.to_dict()
        key = (cfg.ch, flat_top, json.dumps(definition, sort_keys=True))
        shared_name = self._envelopes.get(key)
        waveform = waveform_cfg.build(name if shared_name is None else shared_name)
        if shared_name is None:
            waveform.create(prog, cfg.ch)
            self._envelopes[key] = name
        return waveform.to_wav_kwargs()

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


def _memory_envelope_definition(cfg: InMemoryArbWaveformCfg) -> dict[str, str]:
    return {"style": cfg.style, "sample_hash": cfg.sample_hash}
