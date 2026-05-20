"""cfg_schemas.py — typed CfgSection builders for every ModuleCfg / WaveformCfg subtype.

Each concrete config class (PulseCfg, PulseReadoutCfg, FlatTopWaveformCfg, …) gets its
own _xxx_to_section() function that produces a CfgSection with:
  - correct `choices` for Literal-typed discriminator / enum fields
  - `editable=False` on discriminator-only fields ("type" / "style")
  - proper numeric types (float vs int) so the right spin-box is rendered
  - recursive sub-sections for nested config fields

Public entry points:
  waveform_cfg_to_schema(cfg) → CfgSection   (for any AbsWaveformCfg subtype)
  module_cfg_to_schema(cfg)   → CfgSection   (for any AbsModuleCfg subtype)

Both fall back to the generic module_cfg_to_section() from adapter.py when the subtype
is not explicitly handled (so new types still work, just without choices).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from zcu_tools.gui.adapter import CfgSection, ScalarField, module_cfg_to_section

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import CfgSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WAVEFORM_STYLE_CHOICES = ["const", "cosine", "gauss", "drag", "arb", "flat_top"]
_RAISE_WAVEFORM_STYLE_CHOICES = ["cosine", "gauss", "drag", "arb"]

_MODULE_TYPE_CHOICES = [
    "pulse",
    "readout/direct",
    "readout/pulse",
    "reset/none",
    "reset/pulse",
    "reset/two_pulse",
    "reset/bath",
]


def _locked(value: Any, label: str, choices: list) -> ScalarField:
    return ScalarField(
        value=value, label=label, type=str, editable=False, choices=choices
    )


def _sf(
    value: Any,
    label: str,
    typ: type,
    choices: list | None = None,
    decimals: int | None = None,
) -> ScalarField:
    return ScalarField(
        value=value, label=label, type=typ, choices=choices, decimals=decimals
    )


def _v(cfg: Any, attr: str, default: Any = 0) -> Any:
    return getattr(cfg, attr, default)


# ---------------------------------------------------------------------------
# Waveform section builders
# ---------------------------------------------------------------------------


def _const_to_section(cfg: Any) -> CfgSection:
    return CfgSection(
        label="Const Waveform",
        fields={
            "style": _locked("const", "Style", ["const"]),
            "length": _sf(_v(cfg, "length", 1.0), "Length (us)", float, decimals=3),
        },
    )


def _cosine_to_section(cfg: Any) -> CfgSection:
    return CfgSection(
        label="Cosine Waveform",
        fields={
            "style": _locked("cosine", "Style", _RAISE_WAVEFORM_STYLE_CHOICES),
            "length": _sf(_v(cfg, "length", 0.1), "Length (us)", float, decimals=3),
        },
    )


def _gauss_to_section(cfg: Any) -> CfgSection:
    return CfgSection(
        label="Gauss Waveform",
        fields={
            "style": _locked("gauss", "Style", _RAISE_WAVEFORM_STYLE_CHOICES),
            "length": _sf(_v(cfg, "length", 1.0), "Length (us)", float, decimals=3),
            "sigma": _sf(_v(cfg, "sigma", 0.25), "Sigma (us)", float, decimals=3),
        },
    )


def _drag_to_section(cfg: Any) -> CfgSection:
    return CfgSection(
        label="DRAG Waveform",
        fields={
            "style": _locked("drag", "Style", _RAISE_WAVEFORM_STYLE_CHOICES),
            "length": _sf(_v(cfg, "length", 1.0), "Length (us)", float, decimals=3),
            "sigma": _sf(_v(cfg, "sigma", 0.25), "Sigma (us)", float, decimals=3),
            "delta": _sf(_v(cfg, "delta", 0.0), "Delta (MHz)", float, decimals=2),
            "alpha": _sf(_v(cfg, "alpha", 0.5), "Alpha", float, decimals=4),
        },
    )


def _arb_to_section(cfg: Any) -> CfgSection:
    return CfgSection(
        label="Arb Waveform",
        fields={
            "style": _locked("arb", "Style", _RAISE_WAVEFORM_STYLE_CHOICES),
            "length": _sf(_v(cfg, "length", 1.0), "Length (us)", float, decimals=3),
            "data": _sf(_v(cfg, "data", ""), "Data key", str),
        },
    )


def _flat_top_to_section(cfg: Any) -> CfgSection:
    raise_wav = getattr(cfg, "raise_waveform", None)
    raise_sec = (
        waveform_cfg_to_section(raise_wav)
        if raise_wav is not None
        else _cosine_to_section(None)
    )
    return CfgSection(
        label="FlatTop Waveform",
        fields={
            "style": _locked("flat_top", "Style", _WAVEFORM_STYLE_CHOICES),
            "length": _sf(_v(cfg, "length", 5.0), "Length (us)", float, decimals=3),
            "raise_waveform": raise_sec,
        },
    )


_WAVEFORM_BUILDERS: dict[str, Any] = {
    "const": _const_to_section,
    "cosine": _cosine_to_section,
    "gauss": _gauss_to_section,
    "drag": _drag_to_section,
    "arb": _arb_to_section,
    "flat_top": _flat_top_to_section,
}


def waveform_cfg_to_section(cfg: Any) -> CfgSection:
    """Return a typed CfgSection for any AbsWaveformCfg subtype."""
    style = getattr(cfg, "style", None)
    builder = _WAVEFORM_BUILDERS.get(str(style)) if style else None
    if builder is not None:
        return builder(cfg)
    return module_cfg_to_section(cfg)


# ---------------------------------------------------------------------------
# Module section builders — pulse
# ---------------------------------------------------------------------------


def _pulse_to_section(cfg: Any) -> CfgSection:
    wav = getattr(cfg, "waveform", None)
    wav_sec = (
        waveform_cfg_to_section(wav) if wav is not None else _const_to_section(None)
    )
    return CfgSection(
        label="Pulse",
        fields={
            "type": _locked("pulse", "Type", ["pulse"]),
            "waveform": wav_sec,
            "ch": _sf(_v(cfg, "ch", 0), "Gen ch", int),
            "nqz": _sf(_v(cfg, "nqz", 2), "NQZ", int, [1, 2]),
            "freq": _sf(_v(cfg, "freq", 6000.0), "Freq (MHz)", float, decimals=2),
            "phase": _sf(_v(cfg, "phase", 0.0), "Phase (deg)", float, decimals=2),
            "gain": _sf(_v(cfg, "gain", 0.5), "Gain", float, decimals=4),
            "pre_delay": _sf(
                _v(cfg, "pre_delay", 0.0), "Pre-delay (us)", float, decimals=3
            ),
            "post_delay": _sf(
                _v(cfg, "post_delay", 0.0), "Post-delay (us)", float, decimals=3
            ),
        },
    )


# ---------------------------------------------------------------------------
# Module section builders — readout
# ---------------------------------------------------------------------------


def _direct_readout_to_section(cfg: Any) -> CfgSection:
    return CfgSection(
        label="Direct Readout",
        fields={
            "type": _locked("readout/direct", "Type", ["readout/direct"]),
            "ro_ch": _sf(_v(cfg, "ro_ch", 0), "RO ch", int),
            "ro_freq": _sf(
                _v(cfg, "ro_freq", 6000.0), "RO freq (MHz)", float, decimals=2
            ),
            "ro_length": _sf(
                _v(cfg, "ro_length", 1.0), "RO length (us)", float, decimals=3
            ),
            "trig_offset": _sf(
                _v(cfg, "trig_offset", 0.0), "Trig offset (us)", float, decimals=3
            ),
        },
    )


def _pulse_readout_to_section(cfg: Any) -> CfgSection:
    pulse_cfg = getattr(cfg, "pulse_cfg", None)
    ro_cfg = getattr(cfg, "ro_cfg", None)
    pulse_sec = (
        _pulse_to_section(pulse_cfg)
        if pulse_cfg is not None
        else _pulse_to_section(None)
    )
    ro_sec = (
        _direct_readout_to_section(ro_cfg)
        if ro_cfg is not None
        else _direct_readout_to_section(None)
    )
    return CfgSection(
        label="Pulse Readout",
        fields={
            "type": _locked(
                "readout/pulse", "Type", ["readout/direct", "readout/pulse"]
            ),
            "pulse_cfg": pulse_sec,
            "ro_cfg": ro_sec,
        },
    )


# ---------------------------------------------------------------------------
# Module section builders — reset
# ---------------------------------------------------------------------------


def _none_reset_to_section(_cfg: Any) -> CfgSection:
    return CfgSection(
        label="None Reset",
        fields={
            "type": _locked("reset/none", "Type", _MODULE_TYPE_CHOICES),
        },
    )


def _pulse_reset_to_section(cfg: Any) -> CfgSection:
    pulse_cfg = getattr(cfg, "pulse_cfg", None)
    pulse_sec = (
        _pulse_to_section(pulse_cfg)
        if pulse_cfg is not None
        else _pulse_to_section(None)
    )
    return CfgSection(
        label="Pulse Reset",
        fields={
            "type": _locked("reset/pulse", "Type", _MODULE_TYPE_CHOICES),
            "pulse_cfg": pulse_sec,
        },
    )


def _two_pulse_reset_to_section(cfg: Any) -> CfgSection:
    p1 = getattr(cfg, "pulse1_cfg", None)
    p2 = getattr(cfg, "pulse2_cfg", None)
    return CfgSection(
        label="Two-Pulse Reset",
        fields={
            "type": _locked("reset/two_pulse", "Type", _MODULE_TYPE_CHOICES),
            "pulse1_cfg": _pulse_to_section(p1)
            if p1 is not None
            else _pulse_to_section(None),
            "pulse2_cfg": _pulse_to_section(p2)
            if p2 is not None
            else _pulse_to_section(None),
        },
    )


def _bath_reset_to_section(cfg: Any) -> CfgSection:
    cav = getattr(cfg, "cavity_tone_cfg", None)
    qub = getattr(cfg, "qubit_tone_cfg", None)
    pi2 = getattr(cfg, "pi2_cfg", None)
    return CfgSection(
        label="Bath Reset",
        fields={
            "type": _locked("reset/bath", "Type", _MODULE_TYPE_CHOICES),
            "cavity_tone_cfg": _pulse_to_section(cav)
            if cav is not None
            else _pulse_to_section(None),
            "qubit_tone_cfg": _pulse_to_section(qub)
            if qub is not None
            else _pulse_to_section(None),
            "pi2_cfg": _pulse_to_section(pi2)
            if pi2 is not None
            else _pulse_to_section(None),
        },
    )


_MODULE_BUILDERS: dict[str, Any] = {
    "pulse": _pulse_to_section,
    "readout/direct": _direct_readout_to_section,
    "readout/pulse": _pulse_readout_to_section,
    "reset/none": _none_reset_to_section,
    "reset/pulse": _pulse_reset_to_section,
    "reset/two_pulse": _two_pulse_reset_to_section,
    "reset/bath": _bath_reset_to_section,
}


def module_cfg_to_schema(cfg: Any) -> CfgSection:
    """Return a typed CfgSection for any AbsModuleCfg subtype."""
    type_val = getattr(cfg, "type", None)
    builder = _MODULE_BUILDERS.get(str(type_val)) if type_val else None
    if builder is not None:
        return builder(cfg)
    return module_cfg_to_section(cfg)


# ---------------------------------------------------------------------------
# Public schema factories — build CfgSchema with typed fields from raw values
# Used by Adapters to provide WritebackItem.edit_template
# ---------------------------------------------------------------------------


class _SimpleNamespace:
    """Minimal namespace to pass values to section builders via getattr."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def make_pulse_readout_schema(
    *,
    pulse_ch: int = 0,
    pulse_nqz: int = 2,
    pulse_freq: float = 6000.0,
    pulse_gain: float = 0.2,
    waveform_style: str = "const",
    waveform_length: float = 1.0,
    ro_ch: int = 0,
    ro_length: float = 0.9,
    trig_offset: float = 0.335,
) -> "CfgSchema":
    """Build a CfgSchema for a PulseReadoutCfg with given default values."""
    from zcu_tools.gui.adapter import CfgSchema

    wav_builder = _WAVEFORM_BUILDERS.get(waveform_style, _const_to_section)
    wav_ns = _SimpleNamespace(style=waveform_style, length=waveform_length)
    wav_sec = wav_builder(wav_ns)

    pulse_ns = _SimpleNamespace(
        waveform=None,
        ch=pulse_ch,
        nqz=pulse_nqz,
        freq=pulse_freq,
        phase=0.0,
        gain=pulse_gain,
        pre_delay=0.0,
        post_delay=0.0,
    )
    pulse_sec = _pulse_to_section(pulse_ns)
    pulse_sec.fields["waveform"] = wav_sec

    ro_ns = _SimpleNamespace(
        ro_ch=ro_ch, ro_freq=pulse_freq, ro_length=ro_length, trig_offset=trig_offset
    )
    ro_sec = _direct_readout_to_section(ro_ns)

    root = CfgSection(
        label="Pulse Readout",
        fields={
            "type": _locked(
                "readout/pulse", "Type", ["readout/direct", "readout/pulse"]
            ),
            "pulse_cfg": pulse_sec,
            "ro_cfg": ro_sec,
        },
    )
    return CfgSchema(root=root)


def make_flat_top_waveform_schema(
    *,
    length: float = 5.0,
    raise_style: str = "cosine",
    raise_length: float = 0.1,
) -> "CfgSchema":
    """Build a CfgSchema for a FlatTopWaveformCfg with given default values."""
    from zcu_tools.gui.adapter import CfgSchema

    raise_builder = _WAVEFORM_BUILDERS.get(raise_style, _cosine_to_section)
    raise_ns = _SimpleNamespace(style=raise_style, length=raise_length)
    raise_sec = raise_builder(raise_ns)

    root = CfgSection(
        label="FlatTop Waveform",
        fields={
            "style": _locked("flat_top", "Style", _WAVEFORM_STYLE_CHOICES),
            "length": _sf(length, "Length (us)", float, decimals=3),
            "raise_waveform": raise_sec,
        },
    )
    return CfgSchema(root=root)
