"""Autoflux policy adapter and ModuleLibrary value conversion."""

from __future__ import annotations

from typing import Any

from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.gui.measure_cfg import (
    PROGRAM_SHAPES,
    ProgramSpecPolicy,
    UnknownProgramShapeError,
)
from zcu_tools.program.v2.modules.base import AbsModuleCfg
from zcu_tools.program.v2.modules.waveform import AbsWaveformCfg

AUTOFLUX_PROGRAM_SPEC_POLICY = ProgramSpecPolicy()


def _value(cfg: dict[str, Any], key: str) -> ScalarValue:
    if key not in cfg:
        return DirectValue(value=None)
    return DirectValue(value=cfg[key])


def _normalize_waveform_cfg(cfg_input: object) -> dict[str, Any]:
    if isinstance(cfg_input, AbsWaveformCfg):
        return cfg_input.to_dict()
    if isinstance(cfg_input, dict):
        return cfg_input
    raise TypeError(f"Expected dict or AbsWaveformCfg, got {type(cfg_input)}")


def _normalize_module_cfg(cfg_input: object) -> dict[str, Any]:
    if isinstance(cfg_input, (AbsModuleCfg, AbsWaveformCfg)):
        return cfg_input.to_dict()
    if isinstance(cfg_input, dict):
        return cfg_input
    raise TypeError(f"Expected dict or ModuleCfg, got {type(cfg_input)}")


def _make_waveform_spec(style: str) -> CfgSectionSpec:
    try:
        shape = PROGRAM_SHAPES.waveform(style)
    except UnknownProgramShapeError as exc:
        raise RuntimeError(f"Unsupported waveform style {style!r}") from exc
    return shape.make_spec(AUTOFLUX_PROGRAM_SPEC_POLICY)


def _make_const_waveform_spec() -> CfgSectionSpec:
    return _make_waveform_spec("const")


def _make_cosine_waveform_spec() -> CfgSectionSpec:
    return _make_waveform_spec("cosine")


def _make_pulse_spec(label: str = "Pulse") -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("pulse").make_spec(
        AUTOFLUX_PROGRAM_SPEC_POLICY, label=label
    )


def _make_direct_readout_spec() -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("readout/direct").make_spec(
        AUTOFLUX_PROGRAM_SPEC_POLICY
    )


def _make_pulse_readout_spec() -> CfgSectionSpec:
    return PROGRAM_SHAPES.module("readout/pulse").make_spec(
        AUTOFLUX_PROGRAM_SPEC_POLICY
    )


def pulse_module_ref_spec(
    label: str = "Pulse", optional: bool = False
) -> ReferenceSpec:
    return ReferenceSpec(
        kind="module",
        allowed=[_make_pulse_spec()],
        label=label,
        optional=optional,
    )


def pulse_readout_module_ref_spec(
    label: str = "Readout", optional: bool = False
) -> ReferenceSpec:
    return ReferenceSpec(
        kind="module",
        allowed=[_make_pulse_readout_spec()],
        label=label,
        optional=optional,
    )


def waveform_cfg_to_value(
    cfg_input: object,
) -> tuple[CfgSectionSpec, CfgSectionValue]:
    cfg = _normalize_waveform_cfg(cfg_input)
    style = str(cfg.get("style", "const"))
    spec = _make_waveform_spec(style)

    if style in {"const", "cosine"}:
        value = CfgSectionValue(
            fields={
                "style": DirectValue(style),
                "length": _value(cfg, "length"),
            }
        )
    elif style == "gauss":
        value = CfgSectionValue(
            fields={
                "style": DirectValue("gauss"),
                "length": _value(cfg, "length"),
                "sigma": _value(cfg, "sigma"),
            }
        )
    elif style == "drag":
        value = CfgSectionValue(
            fields={
                "style": DirectValue("drag"),
                "length": _value(cfg, "length"),
                "sigma": _value(cfg, "sigma"),
                "delta": _value(cfg, "delta"),
                "alpha": _value(cfg, "alpha"),
            }
        )
    elif style == "arb":
        value = CfgSectionValue(
            fields={
                "style": DirectValue("arb"),
                "data": _value(cfg, "data"),
            }
        )
    elif style == "flat_top":
        raise_waveform = cfg.get("raise_waveform")
        if isinstance(raise_waveform, dict):
            raise_spec, raise_value = waveform_cfg_to_value(raise_waveform)
        else:
            raise_spec = _make_cosine_waveform_spec()
            raise_value = make_default_value(raise_spec)
        value = CfgSectionValue(
            fields={
                "style": DirectValue("flat_top"),
                "length": _value(cfg, "length"),
                "raise_waveform": ReferenceValue(
                    chosen_key=make_custom_reference_key(raise_spec.label),
                    value=raise_value,
                ),
            }
        )
    else:
        raise RuntimeError(f"Unsupported waveform style {style!r}")

    return spec, value


def _pulse_to_value(cfg: dict[str, Any]) -> CfgSectionValue:
    waveform = cfg.get("waveform")
    if isinstance(waveform, dict):
        waveform_spec, waveform_value = waveform_cfg_to_value(waveform)
    else:
        waveform_spec = _make_const_waveform_spec()
        waveform_value = make_default_value(waveform_spec)
    return CfgSectionValue(
        fields={
            "type": DirectValue("pulse"),
            "waveform": ReferenceValue(
                chosen_key=make_custom_reference_key(waveform_spec.label),
                value=waveform_value,
            ),
            "ch": DirectValue(cfg.get("ch", 0)),
            "nqz": _value(cfg, "nqz"),
            "freq": _value(cfg, "freq"),
            "phase": _value(cfg, "phase"),
            "gain": _value(cfg, "gain"),
            "pre_delay": _value(cfg, "pre_delay"),
            "post_delay": _value(cfg, "post_delay"),
            "mixer_freq": _value(cfg, "mixer_freq"),
        }
    )


def _direct_readout_to_value(cfg: dict[str, Any]) -> CfgSectionValue:
    return CfgSectionValue(
        fields={
            "type": DirectValue("readout/direct"),
            "ro_ch": DirectValue(cfg.get("ro_ch", 0)),
            "ro_freq": _value(cfg, "ro_freq"),
            "ro_length": _value(cfg, "ro_length"),
            "trig_offset": _value(cfg, "trig_offset"),
            "gen_ch": _value(cfg, "gen_ch"),
        }
    )


def _pulse_readout_to_value(cfg: dict[str, Any]) -> CfgSectionValue:
    pulse_cfg = cfg.get("pulse_cfg")
    readout_cfg = cfg.get("ro_cfg")
    pulse_value = (
        _pulse_to_value(pulse_cfg)
        if isinstance(pulse_cfg, dict)
        else make_default_value(_make_pulse_spec())
    )
    readout_value = (
        _direct_readout_to_value(readout_cfg)
        if isinstance(readout_cfg, dict)
        else make_default_value(_make_direct_readout_spec())
    )
    return CfgSectionValue(
        fields={
            "type": DirectValue("readout/pulse"),
            "pulse_cfg": pulse_value,
            "ro_cfg": readout_value,
        }
    )


def module_cfg_to_value(
    cfg_input: object,
) -> tuple[CfgSectionSpec, CfgSectionValue]:
    """Convert only the pulse shapes accepted by autoflux defaults."""
    cfg = _normalize_module_cfg(cfg_input)
    type_value = str(cfg.get("type", ""))
    if type_value == "pulse":
        return _make_pulse_spec(), _pulse_to_value(cfg)
    if type_value == "readout/pulse":
        return _make_pulse_readout_spec(), _pulse_readout_to_value(cfg)
    raise RuntimeError(f"Unsupported module type {type_value!r}")


def module_cfg_shape_label(cfg_input: object) -> str:
    """Resolve every legal measure module discriminator without materializing it."""
    cfg = _normalize_module_cfg(cfg_input)
    if "style" in cfg:
        return waveform_cfg_shape_label(cfg)
    type_value = str(cfg.get("type", ""))
    try:
        return PROGRAM_SHAPES.module(type_value).label
    except UnknownProgramShapeError as exc:
        raise RuntimeError(f"Unsupported module type {type_value!r}") from exc


def waveform_cfg_shape_label(cfg_input: object) -> str:
    cfg = _normalize_waveform_cfg(cfg_input)
    style = str(cfg.get("style", "const"))
    try:
        return PROGRAM_SHAPES.waveform(style).label
    except UnknownProgramShapeError as exc:
        raise RuntimeError(f"Unsupported waveform style {style!r}") from exc
