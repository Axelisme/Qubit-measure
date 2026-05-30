from __future__ import annotations

import logging
from typing import Any, Optional, cast

from zcu_tools.gui.adapter import CfgSchema
from zcu_tools.gui.cfg_schemas import module_cfg_to_value
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.program.v2 import (
    AbsModuleCfg,
    AbsWaveformCfg,
    ModuleCfgFactory,
    WaveformCfgFactory,
)

logger = logging.getLogger(__name__)


def schema_from_module(proposed: object | None) -> Optional[CfgSchema]:
    """Edit CfgSchema from a proposed module/waveform cfg (module-agnostic).

    Delegates to ``module_cfg_to_value``, which auto-routes waveform cfgs to
    ``waveform_cfg_to_value``. Returns None when ``proposed`` is None (no edit
    button shown). Unknown module types fast-fail (``module_cfg_to_value``
    raises) — deliberate.
    """
    if proposed is None:
        return None
    spec, value = module_cfg_to_value(proposed)
    return CfgSchema(spec=spec, value=value)


def build_readout_for_frequency(
    readout: object | None,
    *,
    freq: float,
    pulse_ch: int,
    ro_ch: int,
    ml: ModuleLibrary | None,
) -> object | None:
    """Return a readout config updated to the fitted frequency."""
    if isinstance(readout, AbsModuleCfg):
        try:
            updates: dict[str, object] = {}
            pulse_cfg = getattr(readout, "pulse_cfg", None)
            if pulse_cfg is not None:
                updates["pulse_cfg"] = pulse_cfg.with_updates(freq=freq)

            ro_cfg = getattr(readout, "ro_cfg", None)
            if ro_cfg is not None:
                updates["ro_cfg"] = ro_cfg.with_updates(ro_freq=freq)

            if hasattr(readout, "ro_freq"):
                updates["ro_freq"] = freq

            if updates:
                return cast(Any, readout).with_updates(**updates)
            return readout
        except Exception as exc:
            logger.warning("build_readout_for_frequency failed: %s", exc)

    fallback_raw = {
        "type": "readout/pulse",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": 1.0},
            "ch": pulse_ch,
            "nqz": 2,
            "freq": freq,
            "gain": 0.2,
        },
        "ro_cfg": {
            "ro_ch": ro_ch,
            "ro_freq": freq,
            "ro_length": 1.0,
            "trig_offset": 0.5,
        },
    }
    try:
        return ModuleCfgFactory.from_raw(fallback_raw, ml=ml)
    except Exception:
        return None


def build_waveform_for_length(
    readout: object | None,
    *,
    length: float,
    ml: ModuleLibrary | None,
) -> object | None:
    """Return the probe waveform updated to the requested length."""
    try:
        pulse_cfg = getattr(readout, "pulse_cfg", None)
        waveform = getattr(pulse_cfg, "waveform", None)
        if isinstance(waveform, AbsWaveformCfg):
            return waveform.with_updates(length=length)
    except Exception:
        pass

    fallback_raw = {
        "style": "flat_top",
        "raise_waveform": {"style": "cosine", "length": 0.1},
        "length": length,
    }
    try:
        return WaveformCfgFactory.from_raw(fallback_raw, ml=ml)
    except Exception:
        return None
