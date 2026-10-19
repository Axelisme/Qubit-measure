"""Persistent module/waveform creation templates, retained across reloads.

Only Expr seeds evaluate; ordinary strings, numbers and bools remain literal.
Slot seeds retain library links. The composition root injects TEMPLATES once
into its measure Registry. This module does not register anything at import time.
Placeholder names are lab declarations, not framework defaults. Callers must
supply Template.bindings or register defaults through the startup context.
"""

from __future__ import annotations

from zcu_tools.cfg_model import (
    BathResetSpec,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    Expr,
    FlatTopWaveformSpec,
    GaussWaveformCfg,
    MuxReadoutSpec,
    NoneResetCfg,
    PulseReadoutSpec,
    PulseResetSpec,
    PulseSpec,
    Slot,
    TemplateDef,
    TemplateSeed,
    TwoPulseResetSpec,
)

_QUBIT_FREQ = Expr("{qubit}.freq")
_QUBIT_CH = Expr("{qubit}.wiring.ch")
_RESONATOR_FREQ = Expr("{resonator}.freq")
_RESONATOR_CH = Expr("{resonator}.wiring.ch")
_RESONATOR_RO_CH = Expr("{resonator}.wiring.ro_ch")
_TIME_OF_FLIGHT_PLUS_005_US = Expr("{resonator}.ext.time_of_fly + 0.05")

# Initial pulse duration in us for the readout template and single-tone RO windows.
READOUT_PULSE_LENGTH_US = 1.0


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
    freq: TemplateSeed = _RESONATOR_FREQ,
    gen_ch: TemplateSeed = _RESONATOR_CH,
    ro_ch: TemplateSeed = _RESONATOR_RO_CH,
    ro_length: TemplateSeed = 0.9,
    trig_offset: TemplateSeed = _TIME_OF_FLIGHT_PLUS_005_US,
    at: str = "",
) -> dict[str, TemplateSeed]:
    """Return partial flat readout seeds beneath the optional dotted path ``at``.

    ``freq`` is the shared generator/ADC frequency in MHz. ``gen_ch`` and
    ``ro_ch`` select generator and ADC channels. ``ro_length`` and
    ``trig_offset`` are us. Waveform, gain and Nyquist zone remain caller-owned.
    The default trigger offset is the resonator time
    of flight plus 0.05 us; missing calibration remains Invalid in the consumer.
    Expr seeds evaluate only in the consuming Form. Ordinary strings are literal;
    TemplateDef validates and expands ``{qubit}`` and ``{resonator}`` placeholders.
    This helper does not evaluate or read a library. The consuming catalog checks
    seed paths and value compatibility.
    """
    prefix = f"{at}." if at else ""
    return {
        f"{prefix}freq": freq,
        f"{prefix}gen_ch": gen_ch,
        f"{prefix}ro_ch": ro_ch,
        f"{prefix}ro_length": ro_length,
        f"{prefix}trig_offset": trig_offset,
    }


# Insertion order preserves library creation menus. Blank seeds retain legacy zeros.
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
            **ro(),
            "nqz": 2,
            "gain": 0.1,
            "waveform.style": "const",
            "waveform.length": READOUT_PULSE_LENGTH_US,
        },
        default_name="readout_rf",
    ),
    "readout_dpm": TemplateDef(
        PulseReadoutSpec,
        "Optimized readout (DPM)",
        {
            **ro(
                freq=Expr("{qubit}.ext.best_ro_freq"),
                ro_length=Expr("{qubit}.ext.best_ro_length"),
            ),
            "nqz": 2,
            "gain": Expr("{qubit}.ext.best_ro_gain"),
            "waveform.style": "const",
            "waveform.length": Expr("{qubit}.ext.best_ro_length + 0.1"),
        },
        default_name="readout_dpm",
    ),
    "qub_probe": TemplateDef(
        PulseSpec, "Qubit probe pulse", pulse(length=1.0), default_name="qub_pulse"
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
            "pi2_cfg": Slot("{qubit}.module.x90"),
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
    "flux_pulse": TemplateDef(
        PulseSpec,
        "Flux pulse",
        {
            "ch": Expr("{qubit}.wiring.lo_flux_ch"),
            "freq": 0.0,
            "gain": 1.0,
            "nqz": 2,
            "waveform.style": "const",
            "waveform.length": 1.0,
        },
        default_name="flux_pulse",
    ),
    "pulse:blank": TemplateDef(
        PulseSpec,
        "Blank: pulse",
        {
            "ch": 0,
            "freq": 0.0,
            "gain": 0.0,
            "nqz": 1,
            "phase": 0.0,
            "pre_delay": 0.0,
            "post_delay": 0.0,
            "waveform.style": "const",
            "waveform.length": 0.0,
        },
    ),
    "readout/pulse:blank": TemplateDef(
        PulseReadoutSpec,
        "Blank: readout/pulse",
        {
            "gen_ch": 0,
            "freq": 0.0,
            "gain": 0.0,
            "nqz": 1,
            "phase": 0.0,
            "pre_delay": 0.0,
            "post_delay": 0.0,
            "waveform.style": "const",
            "waveform.length": 0.0,
            "ro_ch": 0,
            "ro_length": 0.0,
            "trig_offset": 0.0,
        },
    ),
    # Editing seeds only: authors must supply tones and verify hardware wiring.
    "readout/mux:blank": TemplateDef(
        MuxReadoutSpec,
        "Blank: readout/mux",
        {
            "gen_ch": 0,
            "nqz": 1,
            "mixer_freq": 0.0,
            "length": 1.0,
            "ro_length": 1.0,
            "pre_delay": 0.0,
            "post_delay": 0.0,
            "trig_offset": 0.0,
        },
        default_name="readout_rf",
    ),
    "reset/none:blank": TemplateDef(
        NoneResetCfg, "Blank: reset/none", {"type": "reset/none"}
    ),
    "reset/pulse:blank": TemplateDef(
        PulseResetSpec,
        "Blank: reset/pulse",
        {
            "pulse_cfg.ch": 0,
            "pulse_cfg.freq": 0.0,
            "pulse_cfg.gain": 0.0,
            "pulse_cfg.nqz": 1,
            "pulse_cfg.phase": 0.0,
            "pulse_cfg.pre_delay": 0.0,
            "pulse_cfg.post_delay": 0.0,
            "pulse_cfg.waveform.style": "const",
            "pulse_cfg.waveform.length": 0.0,
        },
    ),
    "reset/two_pulse:blank": TemplateDef(
        TwoPulseResetSpec,
        "Blank: reset/two_pulse",
        {
            "pulse1_cfg.ch": 0,
            "pulse1_cfg.freq": 0.0,
            "pulse1_cfg.gain": 0.0,
            "pulse1_cfg.nqz": 1,
            "pulse1_cfg.phase": 0.0,
            "pulse1_cfg.pre_delay": 0.0,
            "pulse1_cfg.post_delay": 0.0,
            "pulse1_cfg.waveform.style": "const",
            "pulse1_cfg.waveform.length": 0.0,
            "pulse2_cfg.ch": 0,
            "pulse2_cfg.freq": 0.0,
            "pulse2_cfg.gain": 0.0,
            "pulse2_cfg.nqz": 1,
            "pulse2_cfg.phase": 0.0,
            "pulse2_cfg.pre_delay": 0.0,
            "pulse2_cfg.post_delay": 0.0,
            "pulse2_cfg.waveform.style": "const",
            "pulse2_cfg.waveform.length": 0.0,
        },
    ),
    "reset/bath:blank": TemplateDef(
        BathResetSpec,
        "Blank: reset/bath",
        {
            "cavity_tone_cfg.ch": 0,
            "cavity_tone_cfg.freq": 0.0,
            "cavity_tone_cfg.gain": 0.0,
            "cavity_tone_cfg.nqz": 1,
            "cavity_tone_cfg.phase": 0.0,
            "cavity_tone_cfg.pre_delay": 0.0,
            "cavity_tone_cfg.post_delay": 0.0,
            "cavity_tone_cfg.waveform.style": "const",
            "cavity_tone_cfg.waveform.length": 0.0,
            "qubit_tone_cfg.ch": 0,
            "qubit_tone_cfg.freq": 0.0,
            "qubit_tone_cfg.gain": 0.0,
            "qubit_tone_cfg.nqz": 1,
            "qubit_tone_cfg.phase": 0.0,
            "qubit_tone_cfg.pre_delay": 0.0,
            "qubit_tone_cfg.post_delay": 0.0,
            "qubit_tone_cfg.waveform.style": "const",
            "qubit_tone_cfg.waveform.length": 0.0,
            "pi2_cfg.ch": 0,
            "pi2_cfg.freq": 0.0,
            "pi2_cfg.gain": 0.0,
            "pi2_cfg.nqz": 1,
            "pi2_cfg.phase": 0.0,
            "pi2_cfg.pre_delay": 0.0,
            "pi2_cfg.post_delay": 0.0,
            "pi2_cfg.waveform.style": "const",
            "pi2_cfg.waveform.length": 0.0,
        },
    ),
    "const:blank": TemplateDef(ConstWaveformCfg, "Blank: const", {"length": 0.0}),
    "cosine:blank": TemplateDef(CosineWaveformCfg, "Blank: cosine", {"length": 0.0}),
    "gauss:blank": TemplateDef(
        GaussWaveformCfg, "Blank: gauss", {"length": 0.0, "sigma": 0.0}
    ),
    "drag:blank": TemplateDef(
        DragWaveformCfg,
        "Blank: drag",
        {"length": 0.0, "sigma": 0.0, "delta": 0.0, "alpha": 0.0},
    ),
    "flat_top:blank": TemplateDef(
        FlatTopWaveformSpec,
        "Blank: flat_top",
        {"length": 0.0, "raise_waveform.style": "cosine", "raise_waveform.length": 0.0},
    ),
}
