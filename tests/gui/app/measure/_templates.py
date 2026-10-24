"""Synthetic persistent templates for shared library creation."""

from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    Expr,
    PulseReadoutSpec,
    PulseResetSpec,
    PulseSpec,
    TemplateDef,
)

TEMPLATES: dict[str, TemplateDef] = {
    "res_probe": TemplateDef(
        PulseSpec,
        "Resonator probe",
        {
            "ch": Expr("$resonator.wiring.ch"),
            "freq": Expr("$resonator.freq"),
            "nqz": 1,
            "gain": 0.0,
            "waveform.style": "const",
            "waveform.length": 0.0,
        },
        "readout_rf",
    ),
    "readout": TemplateDef(
        PulseReadoutSpec,
        "Readout",
        {
            "gen_ch": Expr("$resonator.wiring.ch"),
            "freq": Expr("$resonator.freq"),
            "nqz": 2,
            "gain": 0.1,
            "waveform.style": "const",
            "waveform.length": 1.0,
            "ro_ch": Expr("$resonator.wiring.ro_ch"),
            "ro_length": 1.0,
            "trig_offset": Expr("$resonator.ext.time_of_fly"),
        },
        "readout_rf",
    ),
    "reset": TemplateDef(
        PulseResetSpec, "Pulse reset", {"pulse_cfg": "Q1.pulses.seed"}, "reset"
    ),
    "const": TemplateDef(ConstWaveformCfg, "Constant", {"length": 0.0}),
}
