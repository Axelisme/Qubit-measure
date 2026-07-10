"""Autoflux-local pulse/readout specs and ModuleLibrary value conversion."""

from __future__ import annotations

from typing import Any

from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    LiteralSpec,
    ModuleRefSpec,
    ScalarSpec,
    ScalarValue,
    WaveformRefSpec,
    WaveformRefValue,
    make_default_value,
)
from zcu_tools.program.v2.modules.base import AbsModuleCfg
from zcu_tools.program.v2.modules.waveform import AbsWaveformCfg

_MODULE_LABELS: dict[str, str] = {
    "pulse": "Pulse",
    "readout/direct": "Direct Readout",
    "readout/pulse": "Pulse Readout",
    "reset/none": "None Reset",
    "reset/pulse": "Pulse Reset",
    "reset/two_pulse": "Two-Pulse Reset",
    "reset/bath": "Bath Reset",
}

_WAVEFORM_LABELS: dict[str, str] = {
    "const": "Const",
    "cosine": "Cosine",
    "gauss": "Gauss",
    "drag": "DRAG",
    "arb": "Arb",
    "flat_top": "FlatTop",
}


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


def _make_const_waveform_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Const",
        fields={
            "style": LiteralSpec("const"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
        },
    )


def _make_cosine_waveform_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Cosine",
        fields={
            "style": LiteralSpec("cosine"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
        },
    )


def _make_gauss_waveform_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Gauss",
        fields={
            "style": LiteralSpec("gauss"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
            "sigma": ScalarSpec(label="Sigma (us)", type=float, decimals=3),
        },
    )


def _make_drag_waveform_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="DRAG",
        fields={
            "style": LiteralSpec("drag"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
            "sigma": ScalarSpec(label="Sigma (us)", type=float, decimals=3),
            "delta": ScalarSpec(label="Delta (MHz)", type=float, decimals=2),
            "alpha": ScalarSpec(label="Alpha", type=float, decimals=4),
        },
    )


def _make_arb_waveform_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Arb",
        fields={
            "style": LiteralSpec("arb"),
            "data": ScalarSpec(
                label="Data key",
                type=str,
                choices_source="arb_waveforms",
            ),
        },
    )


def _make_flat_top_waveform_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="FlatTop",
        fields={
            "style": LiteralSpec("flat_top"),
            "length": ScalarSpec(label="Length (us)", type=float, decimals=3),
            "raise_waveform": WaveformRefSpec(
                allowed=[
                    _make_cosine_waveform_spec(),
                    _make_gauss_waveform_spec(),
                    _make_drag_waveform_spec(),
                    _make_arb_waveform_spec(),
                ],
                label="Raise Waveform",
            ),
        },
    )


def _make_waveform_spec(style: str) -> CfgSectionSpec:
    factories = {
        "const": _make_const_waveform_spec,
        "cosine": _make_cosine_waveform_spec,
        "gauss": _make_gauss_waveform_spec,
        "drag": _make_drag_waveform_spec,
        "arb": _make_arb_waveform_spec,
        "flat_top": _make_flat_top_waveform_spec,
    }
    factory = factories.get(style)
    if factory is None:
        raise RuntimeError(f"Unsupported waveform style {style!r}")
    return factory()


def _make_pulse_spec(label: str = "Pulse") -> CfgSectionSpec:
    return CfgSectionSpec(
        label=label,
        fields={
            "type": LiteralSpec("pulse"),
            "waveform": WaveformRefSpec(
                allowed=[
                    _make_const_waveform_spec(),
                    _make_cosine_waveform_spec(),
                    _make_gauss_waveform_spec(),
                    _make_drag_waveform_spec(),
                    _make_arb_waveform_spec(),
                    _make_flat_top_waveform_spec(),
                ],
                label="Waveform",
            ),
            "ch": ScalarSpec(label="Gen ch", type=int),
            "nqz": ScalarSpec(label="NQZ", type=int, choices=[1, 2], group="Advanced"),
            "freq": ScalarSpec(label="Freq (MHz)", type=float, decimals=2),
            "gain": ScalarSpec(label="Gain", type=float, decimals=4),
            "phase": ScalarSpec(
                label="Phase (deg)", type=float, decimals=2, group="Advanced"
            ),
            "pre_delay": ScalarSpec(
                label="Pre-delay (us)", type=float, decimals=3, group="Advanced"
            ),
            "post_delay": ScalarSpec(
                label="Post-delay (us)", type=float, decimals=3, group="Advanced"
            ),
            "mixer_freq": ScalarSpec(
                label="Mixer freq (MHz)",
                type=float,
                decimals=2,
                optional=True,
                group="Advanced",
            ),
        },
    )


def _make_direct_readout_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Direct Readout",
        fields={
            "type": LiteralSpec("readout/direct"),
            "ro_ch": ScalarSpec(label="RO ch", type=int),
            "ro_freq": ScalarSpec(label="RO Freq (MHz)", type=float, decimals=2),
            "ro_length": ScalarSpec(label="RO length (us)", type=float, decimals=3),
            "trig_offset": ScalarSpec(label="Trig offset (us)", type=float, decimals=3),
            "gen_ch": ScalarSpec(
                label="Gen ch",
                type=int,
                optional=True,
                group="Advanced",
            ),
        },
    )


def _make_pulse_readout_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Pulse Readout",
        fields={
            "type": LiteralSpec("readout/pulse"),
            "pulse_cfg": _make_pulse_spec(),
            "ro_cfg": _make_direct_readout_spec(),
        },
    )


def pulse_module_ref_spec(
    label: str = "Pulse", optional: bool = False
) -> ModuleRefSpec:
    return ModuleRefSpec(
        allowed=[_make_pulse_spec()],
        label=label,
        optional=optional,
    )


def pulse_readout_module_ref_spec(
    label: str = "Readout", optional: bool = False
) -> ModuleRefSpec:
    return ModuleRefSpec(
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
                "raise_waveform": WaveformRefValue(
                    chosen_key=f"<Custom:{raise_spec.label}>",
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
            "waveform": WaveformRefValue(
                chosen_key=f"<Custom:{waveform_spec.label}>",
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
    label = _MODULE_LABELS.get(type_value)
    if label is None:
        raise RuntimeError(f"Unsupported module type {type_value!r}")
    return label


def waveform_cfg_shape_label(cfg_input: object) -> str:
    cfg = _normalize_waveform_cfg(cfg_input)
    style = str(cfg.get("style", "const"))
    label = _WAVEFORM_LABELS.get(style)
    if label is None:
        raise RuntimeError(f"Unsupported waveform style {style!r}")
    return label
