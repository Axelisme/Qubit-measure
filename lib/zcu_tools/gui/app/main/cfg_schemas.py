"""cfg_schemas.py — convert Module/Waveform dicts to (spec, value) pairs.

Strictly accepts dictionaries or module config objects and represents missing
fields with explicit unset scalar values.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

from zcu_tools.gui.app.main.adapter import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarValue,
    WaveformRefValue,
    make_default_value,
)
from zcu_tools.gui.app.main.specs import (
    make_bath_reset_spec,
    make_const_waveform_spec,
    make_direct_readout_spec,
    make_none_reset_spec,
    make_pulse_readout_spec,
    make_pulse_reset_spec,
    make_pulse_spec,
    make_two_pulse_reset_spec,
    make_waveform_spec_by_style,
)
from zcu_tools.program.v2.modules.base import AbsModuleCfg
from zcu_tools.program.v2.modules.waveform import AbsWaveformCfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _val(cfg: dict, key: str, default: Any = None) -> ScalarValue:
    """Extract a scalar value from dict, marking as is_unset if key is missing."""
    if key not in cfg:
        return DirectValue(value=default, is_unset=True)
    return DirectValue(value=cfg[key], is_unset=False)


# ---------------------------------------------------------------------------
# Waveform value builders
# ---------------------------------------------------------------------------


def waveform_cfg_to_value(cfg_input: Any) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Return (spec, value) from a raw waveform dictionary or object."""
    if isinstance(cfg_input, AbsWaveformCfg):
        cfg = cfg_input.to_dict()
    elif isinstance(cfg_input, dict):
        cfg = cfg_input
    else:
        raise TypeError(f"Expected dict or AbsWaveformCfg, got {type(cfg_input)}")
    style = cfg.get("style", "const")
    logger.debug("waveform_cfg_to_value: style=%r keys=%r", style, list(cfg.keys()))
    spec = make_waveform_spec_by_style(style)

    if style == "const":
        val = CfgSectionValue(
            fields={
                "style": DirectValue("const"),
                "length": _val(cfg, "length", 1.0),
            }
        )
    elif style == "cosine":
        val = CfgSectionValue(
            fields={
                "style": DirectValue("cosine"),
                "length": _val(cfg, "length", 0.1),
            }
        )
    elif style == "gauss":
        val = CfgSectionValue(
            fields={
                "style": DirectValue("gauss"),
                "length": _val(cfg, "length", 1.0),
                "sigma": _val(cfg, "sigma", 0.25),
            }
        )
    elif style == "drag":
        val = CfgSectionValue(
            fields={
                "style": DirectValue("drag"),
                "length": _val(cfg, "length", 1.0),
                "sigma": _val(cfg, "sigma", 0.25),
                "delta": _val(cfg, "delta", -200.0),
                "alpha": _val(cfg, "alpha", 0.5),
            }
        )
    elif style == "flat_top":
        raise_wav = cfg.get("raise_waveform")
        if isinstance(raise_wav, dict):
            _, raise_val = waveform_cfg_to_value(raise_wav)
            raise_style = raise_wav.get("style", "cosine")
            raise_spec = make_waveform_spec_by_style(raise_style)
        else:
            from zcu_tools.gui.app.main.specs.waveform import make_cosine_waveform_spec

            raise_spec = make_cosine_waveform_spec()
            raise_val = make_default_value(raise_spec)

        val = CfgSectionValue(
            fields={
                "style": DirectValue("flat_top"),
                "length": _val(cfg, "length", 1.0),
                "raise_waveform": WaveformRefValue(
                    chosen_key=f"<Custom:{raise_spec.label}>",
                    value=raise_val,
                ),
            }
        )
    elif style == "arb":
        val = CfgSectionValue(
            fields={
                "style": DirectValue("arb"),
                "length": _val(cfg, "length", 1.0),
                "data": _val(cfg, "data"),
            }
        )
    else:
        raise RuntimeError(f"Unsupported waveform style {style!r}")

    return spec, val


# ---------------------------------------------------------------------------
# Module value builders
# ---------------------------------------------------------------------------


def _pulse_to_value(cfg: dict) -> CfgSectionValue:
    wav = cfg.get("waveform")
    if isinstance(wav, dict):
        wav_spec, wav_val = waveform_cfg_to_value(wav)
    else:
        wav_spec = make_const_waveform_spec()
        wav_val = make_default_value(wav_spec)

    return CfgSectionValue(
        fields={
            "type": DirectValue("pulse"),
            "waveform": WaveformRefValue(
                chosen_key=f"<Custom:{wav_spec.label}>",
                value=wav_val,
            ),
            "ch": DirectValue(cfg.get("ch", 0)),
            "nqz": _val(cfg, "nqz", 2),
            "freq": _val(cfg, "freq", 6000.0),
            "phase": _val(cfg, "phase", 0.0),
            "gain": _val(cfg, "gain", 0.5),
            "pre_delay": _val(cfg, "pre_delay", 0.0),
            "post_delay": _val(cfg, "post_delay", 0.0),
        }
    )


def _direct_readout_to_value(cfg: dict) -> CfgSectionValue:
    return CfgSectionValue(
        fields={
            "type": DirectValue("readout/direct"),
            "ro_ch": DirectValue(cfg.get("ro_ch", 0)),
            "ro_freq": _val(cfg, "ro_freq", 6000.0),
            "ro_length": _val(cfg, "ro_length", 1.0),
            "trig_offset": _val(cfg, "trig_offset", 0.0),
        }
    )


def _pulse_readout_to_value(cfg: dict) -> CfgSectionValue:
    pulse_cfg = cfg.get("pulse_cfg")
    ro_cfg = cfg.get("ro_cfg")

    pulse_val = (
        _pulse_to_value(pulse_cfg)
        if isinstance(pulse_cfg, dict)
        else make_default_value(make_pulse_spec())
    )
    ro_val = (
        _direct_readout_to_value(ro_cfg)
        if isinstance(ro_cfg, dict)
        else make_default_value(make_direct_readout_spec())
    )
    return CfgSectionValue(
        fields={
            "type": DirectValue("readout/pulse"),
            "pulse_cfg": pulse_val,
            "ro_cfg": ro_val,
        }
    )


def _none_reset_to_value(cfg: dict) -> CfgSectionValue:  # noqa: ARG001
    return CfgSectionValue(fields={"type": DirectValue("reset/none")})


def _pulse_reset_to_value(cfg: dict) -> CfgSectionValue:
    pulse_cfg = cfg.get("pulse_cfg")
    pulse_val = (
        _pulse_to_value(pulse_cfg)
        if isinstance(pulse_cfg, dict)
        else make_default_value(make_pulse_spec())
    )
    return CfgSectionValue(
        fields={
            "type": DirectValue("reset/pulse"),
            "pulse_cfg": pulse_val,
        }
    )


def _two_pulse_reset_to_value(cfg: dict) -> CfgSectionValue:
    p1 = cfg.get("pulse1_cfg")
    p2 = cfg.get("pulse2_cfg")
    return CfgSectionValue(
        fields={
            "type": DirectValue("reset/two_pulse"),
            "pulse1_cfg": _pulse_to_value(p1)
            if isinstance(p1, dict)
            else make_default_value(make_pulse_spec()),
            "pulse2_cfg": _pulse_to_value(p2)
            if isinstance(p2, dict)
            else make_default_value(make_pulse_spec()),
        }
    )


def _bath_reset_to_value(cfg: dict) -> CfgSectionValue:
    cav = cfg.get("cavity_tone_cfg")
    qub = cfg.get("qubit_tone_cfg")
    pi2 = cfg.get("pi2_cfg")
    return CfgSectionValue(
        fields={
            "type": DirectValue("reset/bath"),
            "cavity_tone_cfg": _pulse_to_value(cav)
            if isinstance(cav, dict)
            else make_default_value(make_pulse_spec()),
            "qubit_tone_cfg": _pulse_to_value(qub)
            if isinstance(qub, dict)
            else make_default_value(make_pulse_spec()),
            "pi2_cfg": _pulse_to_value(pi2)
            if isinstance(pi2, dict)
            else make_default_value(make_pulse_spec()),
            "relax_delay": _val(cfg, "relax_delay", 1.0),
        }
    )


_MODULE_VALUE_BUILDERS = {
    "pulse": _pulse_to_value,
    "readout/direct": _direct_readout_to_value,
    "readout/pulse": _pulse_readout_to_value,
    "reset/none": _none_reset_to_value,
    "reset/pulse": _pulse_reset_to_value,
    "reset/two_pulse": _two_pulse_reset_to_value,
    "reset/bath": _bath_reset_to_value,
}

_MODULE_SPEC_FACTORIES = {
    "pulse": make_pulse_spec,
    "readout/direct": make_direct_readout_spec,
    "readout/pulse": make_pulse_readout_spec,
    "reset/none": make_none_reset_spec,
    "reset/pulse": make_pulse_reset_spec,
    "reset/two_pulse": make_two_pulse_reset_spec,
    "reset/bath": make_bath_reset_spec,
}


def module_cfg_to_value(cfg_input: Any) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Return (spec, value) from a raw module dictionary or object."""
    # 1. Normalize input to a dictionary
    if isinstance(cfg_input, (AbsModuleCfg, AbsWaveformCfg)):
        cfg = cfg_input.to_dict()
    elif isinstance(cfg_input, dict):
        cfg = cfg_input
    else:
        raise TypeError(f"Expected dict or ModuleCfg, got {type(cfg_input)}")
    logger.debug(
        "module_cfg_to_value: type=%r style=%r keys=%r",
        cfg.get("type"),
        cfg.get("style"),
        list(cfg.keys()),
    )

    # 2. Check for waveform styles
    if "style" in cfg:
        return waveform_cfg_to_value(cfg)

    # 3. Check for module types
    type_val = str(cfg.get("type", ""))
    spec_factory = _MODULE_SPEC_FACTORIES.get(type_val)
    builder = _MODULE_VALUE_BUILDERS.get(type_val)
    if spec_factory is not None and builder is not None:
        return spec_factory(), builder(cfg)

    # 4. Unknown module type — fast fail
    raise RuntimeError(f"Unsupported module type {type_val!r}")
