"""Persistent module/waveform creation templates, retained across reloads.

Only Expr seeds evaluate; ordinary strings, numbers and bools remain literal.
Slot seeds retain library links. The composition root injects TEMPLATES once
into its measure Registry; definitions projects that catalog into the library
creation menu. This module does not register anything at import time.
"""

from __future__ import annotations

from zcu_tools.cfg_model import (
    ArbWaveformCfg,
    BathResetSpec,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DirectReadoutCfg,
    DragWaveformCfg,
    Expr,
    FlatTopWaveformSpec,
    GaussWaveformCfg,
    NoneResetCfg,
    PulseReadoutSpec,
    PulseResetSpec,
    PulseSpec,
    Slot,
    Template,
    TemplateDef,
    TemplateSeed,
    TwoPulseResetSpec,
)

_QUBIT_FREQ = Expr("{qubit}.freq")
_QUBIT_CH = Expr("{qubit}.wiring.L01.ch")
_RESONATOR_FREQ = Expr("{resonator}.freq")
_RESONATOR_RO_CH = Expr("{resonator}.wiring.ro_ch")
_TIME_OF_FLIGHT_PLUS_005_US = Expr("{resonator}.ext.time_of_fly + 0.05")


def pulse(
    *,
    freq: TemplateSeed = _QUBIT_FREQ,
    ch: TemplateSeed = _QUBIT_CH,
    gain: TemplateSeed = 0.1,
    length: TemplateSeed = 5.1,
    at: str = "",
) -> dict[str, TemplateSeed]:
    """Return partial pulse seeds relative to a persistent module root.

    ``freq`` is MHz, ``ch`` is a generator channel, ``gain`` is normalized
    amplitude, and ``length`` is waveform duration in us. ``at`` is an optional
    dotted path to the pulse section; an empty path seeds a root PulseSpec.
    The Nyquist zone is 2 and the waveform style is explicitly const, preserving
    the legacy creation value without adding a persistent Spec default.
    No library lookup or evaluation occurs here.
    Wrap expressions in Expr; ordinary strings, including those starting with
    ``$``, are literal values. Both string kinds may use ``{qubit}`` and
    ``{resonator}``; TemplateDef validates and expands these placeholders.
    Seed paths and value compatibility are checked by the consuming catalog.
    """
    prefix = f"{at}." if at else ""
    return {
        f"{prefix}freq": freq,
        f"{prefix}ch": ch,
        f"{prefix}nqz": 2,
        f"{prefix}gain": gain,
        f"{prefix}waveform.style": "const",
        f"{prefix}waveform.length": length,
    }


def ro(
    *,
    ro_freq: TemplateSeed = _RESONATOR_FREQ,
    ro_ch: TemplateSeed = _RESONATOR_RO_CH,
    ro_length: TemplateSeed = 0.9,
    trig_offset: TemplateSeed = _TIME_OF_FLIGHT_PLUS_005_US,
    at: str = "ro_cfg",
) -> dict[str, TemplateSeed]:
    """Return partial acquisition seeds relative to a persistent module root.

    ``ro_freq`` is MHz, ``ro_ch`` is an ADC channel, and ``ro_length`` and
    ``trig_offset`` are us. ``at`` is the dotted acquisition section path;
    use an empty path for DirectReadoutCfg. The default trigger offset is the
    resonator time of flight plus 0.05 us; missing calibration remains Invalid
    in the consumer, rather than substituting a numeric fallback.
    Wrap expressions in Expr; ordinary strings, numbers and bools are literal.
    Strings and Expr may use ``{qubit}`` and ``{resonator}``, validated and
    expanded by TemplateDef. This helper does not evaluate or read a library.
    Seed paths and value compatibility are checked by the consuming catalog.
    """
    prefix = f"{at}." if at else ""
    return {
        f"{prefix}ro_freq": ro_freq,
        f"{prefix}ro_ch": ro_ch,
        f"{prefix}ro_length": ro_length,
        f"{prefix}trig_offset": trig_offset,
    }


# Insertion order preserves library creation menus. Empty seeds preserve blanks.
TEMPLATES: dict[str, TemplateDef] = {
    "res_probe": TemplateDef(
        PulseSpec,
        "Resonator probe",
        pulse(
            freq=Expr("{resonator}.freq"),
            ch=Expr("{resonator}.wiring.ch"),
            gain=0.05,
            length=1.0,
        ),
        default_name="readout_rf",
    ),
    "readout": TemplateDef(
        PulseReadoutSpec,
        "Pulse readout",
        {
            **pulse(
                freq=Expr("{resonator}.freq"),
                ch=Expr("{resonator}.wiring.ch"),
                length=1.0,
                at="pulse_cfg",
            ),
            **ro(),
        },
        default_name="readout_rf",
    ),
    "readout_dpm": TemplateDef(
        PulseReadoutSpec,
        "Optimized readout (DPM)",
        {
            **pulse(
                freq=Expr("{qubit}.ext.best_ro_freq"),
                ch=Expr("{resonator}.wiring.ch"),
                gain=Expr("{qubit}.ext.best_ro_gain"),
                length=Expr("{qubit}.ext.best_ro_length + 0.1"),
                at="pulse_cfg",
            ),
            **ro(
                ro_freq=Expr("{qubit}.ext.best_ro_freq"),
                ro_length=Expr("{qubit}.ext.best_ro_length"),
            ),
        },
        default_name="readout_dpm",
    ),
    "direct_readout": TemplateDef(
        DirectReadoutCfg, "Direct readout", ro(at=""), default_name="readout_direct"
    ),
    "qub_probe": TemplateDef(
        PulseSpec, "Qubit probe pulse", pulse(), default_name="qub_pulse"
    ),
    "pi_pulse": TemplateDef(PulseSpec, "Pi pulse", pulse(), default_name="pi_amp"),
    "pi2_pulse": TemplateDef(PulseSpec, "Pi/2 pulse", pulse(), default_name="pi2_amp"),
    "none_reset": TemplateDef(NoneResetCfg, "No reset", {}, default_name="reset_none"),
    "reset": TemplateDef(
        PulseResetSpec,
        "Pulse reset",
        pulse(gain=0.2, length=1.0, at="pulse_cfg"),
        default_name="reset_10",
    ),
    "two_pulse_reset": TemplateDef(
        TwoPulseResetSpec,
        "Two-pulse reset",
        {
            **pulse(gain=0.2, length=1.0, at="pulse1_cfg"),
            **pulse(gain=0.2, length=1.0, at="pulse2_cfg"),
        },
        default_name="reset_120",
    ),
    "bath_reset": TemplateDef(
        BathResetSpec,
        "Bath reset",
        {
            **pulse(
                freq=Expr("{resonator}.freq"),
                ch=Expr("{resonator}.wiring.ch"),
                gain=0.1,
                length=1.0,
                at="cavity_tone_cfg",
            ),
            **pulse(gain=0.2, length=1.0, at="qubit_tone_cfg"),
            "pi2_cfg": Slot("{qubit}.module.x90", fallback=Template("pi2_pulse")),
        },
        default_name="reset_bath",
    ),
    "qub_waveform": TemplateDef(
        CosineWaveformCfg,
        "Qubit drive waveform",
        {"length": 0.1},
        default_name="qub_flat",
    ),
    "res_waveform": TemplateDef(
        ConstWaveformCfg,
        "Res-probe waveform",
        {"length": 1.0},
        default_name="ro_waveform",
    ),
    "pulse:blank": TemplateDef(PulseSpec, "Blank: pulse", {}),
    "readout/direct:blank": TemplateDef(DirectReadoutCfg, "Blank: readout/direct", {}),
    "readout/pulse:blank": TemplateDef(PulseReadoutSpec, "Blank: readout/pulse", {}),
    "reset/none:blank": TemplateDef(NoneResetCfg, "Blank: reset/none", {}),
    "reset/pulse:blank": TemplateDef(PulseResetSpec, "Blank: reset/pulse", {}),
    "reset/two_pulse:blank": TemplateDef(
        TwoPulseResetSpec, "Blank: reset/two_pulse", {}
    ),
    "reset/bath:blank": TemplateDef(BathResetSpec, "Blank: reset/bath", {}),
    "const:blank": TemplateDef(ConstWaveformCfg, "Blank: const", {}),
    "cosine:blank": TemplateDef(CosineWaveformCfg, "Blank: cosine", {}),
    "gauss:blank": TemplateDef(GaussWaveformCfg, "Blank: gauss", {}),
    "drag:blank": TemplateDef(DragWaveformCfg, "Blank: drag", {}),
    "flat_top:blank": TemplateDef(FlatTopWaveformSpec, "Blank: flat_top", {}),
    "arb:blank": TemplateDef(ArbWaveformCfg, "Blank: arb", {}),
}
