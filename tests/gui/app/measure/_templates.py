"""Synthetic persistent templates for shared library creation."""

from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    Expr,
    NoneResetCfg,
    PulseReadoutSpec,
    PulseSpec,
    TemplateDef,
)

TEMPLATES: dict[str, TemplateDef] = {
    "res_probe": TemplateDef(
        PulseSpec,
        "Resonator probe",
        {
            "ch": Expr("{resonator}.wiring.ch"),
            "freq": Expr("{resonator}.freq"),
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
            "pulse_cfg.ch": Expr("{resonator}.wiring.ch"),
            "pulse_cfg.freq": Expr("{resonator}.freq"),
            "pulse_cfg.nqz": 2,
            "pulse_cfg.gain": 0.1,
            "pulse_cfg.waveform.style": "const",
            "pulse_cfg.waveform.length": 1.0,
            "ro_cfg.ro_ch": Expr("{resonator}.wiring.ro_ch"),
            "ro_cfg.ro_freq": Expr("{resonator}.freq"),
            "ro_cfg.ro_length": 1.0,
            "ro_cfg.trig_offset": Expr("{resonator}.ext.time_of_fly"),
        },
        "readout_rf",
    ),
    "none_reset": TemplateDef(NoneResetCfg, "No reset", {}, "reset_none"),
    "const": TemplateDef(ConstWaveformCfg, "Constant", {"length": 0.0}),
}
