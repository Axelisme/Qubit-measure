"""cfg_schemas.py — convert ModuleCfg / WaveformCfg objects to (spec, value) pairs.

The Spec side comes from gui/specs/ static constants.
This module only handles the value extraction: reading actual field values from
a live config object and producing a CfgSectionValue that matches its spec.

Public entry points:
    module_cfg_to_value(cfg) -> (CfgSectionSpec, CfgSectionValue)
    waveform_cfg_to_value(cfg) -> (CfgSectionSpec, CfgSectionValue)
"""

from __future__ import annotations

from typing import Any

from zcu_tools.gui.adapter import (
    CfgSectionSpec,
    CfgSectionValue,
    ChannelValue,
    ScalarValue,
    WaveformRefValue,
    make_default_value,
)
from zcu_tools.gui.specs import (
    BATH_RESET_SPEC,
    CONST_WAVEFORM_SPEC,
    DIRECT_READOUT_SPEC,
    NONE_RESET_SPEC,
    PULSE_READOUT_SPEC,
    PULSE_RESET_SPEC,
    PULSE_SPEC,
    TWO_PULSE_RESET_SPEC,
    WAVEFORM_SPEC_BY_STYLE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _v(cfg: Any, attr: str, default: Any = 0) -> Any:
    return getattr(cfg, attr, default)


# ---------------------------------------------------------------------------
# Waveform value builders
# ---------------------------------------------------------------------------


def waveform_cfg_to_value(cfg: Any) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Return (spec, value) for any AbsWaveformCfg subtype."""
    style = str(getattr(cfg, "style", "const"))
    spec = WAVEFORM_SPEC_BY_STYLE.get(style, CONST_WAVEFORM_SPEC)

    if style == "const":
        val = CfgSectionValue(
            fields={
                "style": ScalarValue("const"),
                "length": ScalarValue(float(_v(cfg, "length", 1.0))),
            }
        )
    elif style == "cosine":
        val = CfgSectionValue(
            fields={
                "style": ScalarValue("cosine"),
                "length": ScalarValue(float(_v(cfg, "length", 0.1))),
            }
        )
    elif style == "gauss":
        val = CfgSectionValue(
            fields={
                "style": ScalarValue("gauss"),
                "length": ScalarValue(float(_v(cfg, "length", 1.0))),
                "sigma": ScalarValue(float(_v(cfg, "sigma", 0.25))),
            }
        )
    elif style == "drag":
        val = CfgSectionValue(
            fields={
                "style": ScalarValue("drag"),
                "length": ScalarValue(float(_v(cfg, "length", 1.0))),
                "sigma": ScalarValue(float(_v(cfg, "sigma", 0.25))),
                "delta": ScalarValue(float(_v(cfg, "delta", 0.0))),
                "alpha": ScalarValue(float(_v(cfg, "alpha", 0.5))),
            }
        )
    elif style == "arb":
        val = CfgSectionValue(
            fields={
                "style": ScalarValue("arb"),
                "length": ScalarValue(float(_v(cfg, "length", 1.0))),
                "data": ScalarValue(str(_v(cfg, "data", ""))),
            }
        )
    elif style == "flat_top":
        raise_wav = getattr(cfg, "raise_waveform", None)
        if raise_wav is not None:
            _, raise_val = waveform_cfg_to_value(raise_wav)
            raise_spec = WAVEFORM_SPEC_BY_STYLE.get(
                str(getattr(raise_wav, "style", "cosine")), CONST_WAVEFORM_SPEC
            )
        else:
            from zcu_tools.gui.specs.waveform import COSINE_WAVEFORM_SPEC

            raise_spec = COSINE_WAVEFORM_SPEC
            raise_val = make_default_value(raise_spec)

        raise_label = raise_spec.label
        val = CfgSectionValue(
            fields={
                "style": ScalarValue("flat_top"),
                "length": ScalarValue(float(_v(cfg, "length", 5.0))),
                "raise_waveform": WaveformRefValue(
                    chosen_key=f"<Custom:{raise_label}>",
                    value=raise_val,
                ),
            }
        )
    else:
        val = make_default_value(spec)

    return spec, val


# ---------------------------------------------------------------------------
# Module value builders
# ---------------------------------------------------------------------------


def _pulse_to_value(cfg: Any) -> CfgSectionValue:
    wav = getattr(cfg, "waveform", None)
    if wav is not None:
        wav_spec, wav_val = waveform_cfg_to_value(wav)
    else:
        wav_spec = CONST_WAVEFORM_SPEC
        wav_val = make_default_value(wav_spec)

    wav_label = wav_spec.label
    return CfgSectionValue(
        fields={
            "type": ScalarValue("pulse"),
            "waveform": WaveformRefValue(
                chosen_key=f"<Custom:{wav_label}>",
                value=wav_val,
            ),
            "ch": ChannelValue(chosen=int(_v(cfg, "ch", 0)), resolved=None),
            "nqz": ScalarValue(int(_v(cfg, "nqz", 2))),
            "freq": ScalarValue(float(_v(cfg, "freq", 6000.0))),
            "phase": ScalarValue(float(_v(cfg, "phase", 0.0))),
            "gain": ScalarValue(float(_v(cfg, "gain", 0.5))),
            "pre_delay": ScalarValue(float(_v(cfg, "pre_delay", 0.0))),
            "post_delay": ScalarValue(float(_v(cfg, "post_delay", 0.0))),
        }
    )


def _direct_readout_to_value(cfg: Any) -> CfgSectionValue:
    return CfgSectionValue(
        fields={
            "type": ScalarValue("readout/direct"),
            "ro_ch": ChannelValue(chosen=int(_v(cfg, "ro_ch", 0)), resolved=None),
            "ro_freq": ScalarValue(float(_v(cfg, "ro_freq", 6000.0))),
            "ro_length": ScalarValue(float(_v(cfg, "ro_length", 1.0))),
            "trig_offset": ScalarValue(float(_v(cfg, "trig_offset", 0.0))),
        }
    )


def _pulse_readout_to_value(cfg: Any) -> CfgSectionValue:
    pulse_cfg = getattr(cfg, "pulse_cfg", None)
    ro_cfg = getattr(cfg, "ro_cfg", None)
    pulse_val = (
        _pulse_to_value(pulse_cfg)
        if pulse_cfg is not None
        else make_default_value(PULSE_SPEC)
    )
    ro_val = (
        _direct_readout_to_value(ro_cfg)
        if ro_cfg is not None
        else make_default_value(DIRECT_READOUT_SPEC)
    )
    return CfgSectionValue(
        fields={
            "type": ScalarValue("readout/pulse"),
            "pulse_cfg": pulse_val,
            "ro_cfg": ro_val,
        }
    )


def _none_reset_to_value(cfg: Any) -> CfgSectionValue:  # noqa: ARG001
    return CfgSectionValue(fields={"type": ScalarValue("reset/none")})


def _pulse_reset_to_value(cfg: Any) -> CfgSectionValue:
    pulse_cfg = getattr(cfg, "pulse_cfg", None)
    pulse_val = (
        _pulse_to_value(pulse_cfg)
        if pulse_cfg is not None
        else make_default_value(PULSE_SPEC)
    )
    return CfgSectionValue(
        fields={
            "type": ScalarValue("reset/pulse"),
            "pulse_cfg": pulse_val,
        }
    )


def _two_pulse_reset_to_value(cfg: Any) -> CfgSectionValue:
    p1 = getattr(cfg, "pulse1_cfg", None)
    p2 = getattr(cfg, "pulse2_cfg", None)
    return CfgSectionValue(
        fields={
            "type": ScalarValue("reset/two_pulse"),
            "pulse1_cfg": _pulse_to_value(p1)
            if p1 is not None
            else make_default_value(PULSE_SPEC),
            "pulse2_cfg": _pulse_to_value(p2)
            if p2 is not None
            else make_default_value(PULSE_SPEC),
        }
    )


def _bath_reset_to_value(cfg: Any) -> CfgSectionValue:
    cav = getattr(cfg, "cavity_tone_cfg", None)
    qub = getattr(cfg, "qubit_tone_cfg", None)
    pi2 = getattr(cfg, "pi2_cfg", None)
    return CfgSectionValue(
        fields={
            "type": ScalarValue("reset/bath"),
            "cavity_tone_cfg": _pulse_to_value(cav)
            if cav is not None
            else make_default_value(PULSE_SPEC),
            "qubit_tone_cfg": _pulse_to_value(qub)
            if qub is not None
            else make_default_value(PULSE_SPEC),
            "pi2_cfg": _pulse_to_value(pi2)
            if pi2 is not None
            else make_default_value(PULSE_SPEC),
        }
    )


_MODULE_SPEC_BY_TYPE: dict[str, CfgSectionSpec] = {
    "pulse": PULSE_SPEC,
    "readout/direct": DIRECT_READOUT_SPEC,
    "readout/pulse": PULSE_READOUT_SPEC,
    "reset/none": NONE_RESET_SPEC,
    "reset/pulse": PULSE_RESET_SPEC,
    "reset/two_pulse": TWO_PULSE_RESET_SPEC,
    "reset/bath": BATH_RESET_SPEC,
}

_MODULE_VALUE_BUILDERS: dict[str, Any] = {
    "pulse": _pulse_to_value,
    "readout/direct": _direct_readout_to_value,
    "readout/pulse": _pulse_readout_to_value,
    "reset/none": _none_reset_to_value,
    "reset/pulse": _pulse_reset_to_value,
    "reset/two_pulse": _two_pulse_reset_to_value,
    "reset/bath": _bath_reset_to_value,
}


def module_cfg_to_value(cfg: Any) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Return (spec, value) for any AbsModuleCfg or AbsWaveformCfg subtype.

    Used by CfgFormWidget._on_ref_changed when loading a named module from ml.
    Falls back to a generic dict-based extraction for unknown types.
    """
    try:
        from zcu_tools.program.v2.modules.waveform import AbsWaveformCfg

        if isinstance(cfg, AbsWaveformCfg):
            return waveform_cfg_to_value(cfg)
    except Exception:
        pass

    type_val = str(getattr(cfg, "type", ""))
    spec = _MODULE_SPEC_BY_TYPE.get(type_val)
    builder = _MODULE_VALUE_BUILDERS.get(type_val)
    if spec is not None and builder is not None:
        return spec, builder(cfg)

    # Generic fallback: mirror to_dict() structure
    if hasattr(cfg, "to_dict"):
        d = cfg.to_dict()
    else:
        d = dict(cfg)
    return _dict_to_spec_value(d)


def _dict_to_spec_value(
    data: dict,
) -> tuple[CfgSectionSpec, CfgSectionValue]:
    from zcu_tools.gui.adapter import ScalarSpec

    spec_fields: dict = {}
    val_fields: dict = {}
    for k, v in data.items():
        if isinstance(v, dict):
            sub_spec, sub_val = _dict_to_spec_value(v)
            spec_fields[k] = sub_spec
            val_fields[k] = sub_val
        elif isinstance(v, (int, float, bool, str)) or v is None:
            is_discriminator = k in ("type", "style")
            spec_fields[k] = ScalarSpec(
                label=k.replace("_", " ").title(),
                type=type(v) if v is not None else str,
                hidden=is_discriminator,
            )
            val_fields[k] = ScalarValue(v)
    return CfgSectionSpec(fields=spec_fields), CfgSectionValue(fields=val_fields)
