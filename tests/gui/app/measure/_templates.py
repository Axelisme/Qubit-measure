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
            "gen_ch": Expr("{resonator}.wiring.ch"),
            "freq": Expr("{resonator}.freq"),
            "nqz": 2,
            "gain": 0.1,
            "waveform.style": "const",
            "waveform.length": 1.0,
            "ro_ch": Expr("{resonator}.wiring.ro_ch"),
            "ro_length": 1.0,
            "trig_offset": Expr("{resonator}.ext.time_of_fly"),
        },
        "readout_rf",
    ),
    "none_reset": TemplateDef(NoneResetCfg, "No reset", {}, "reset_none"),
    "const": TemplateDef(ConstWaveformCfg, "Constant", {"length": 0.0}),
}
