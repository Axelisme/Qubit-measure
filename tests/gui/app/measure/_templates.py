"""Synthetic persistent templates and unchanged legacy cfg role factories."""

from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    Expr,
    NoneResetCfg,
    PulseReadoutSpec,
    PulseSpec,
    TemplateDef,
)
from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.role_factories import RoleFactorySpec
from zcu_tools.gui.app.measure.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
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


def readout_shape() -> CfgSectionSpec:
    return PROGRAM_SHAPES.get("module", "readout/pulse").make_spec(
        MAIN_PROGRAM_SPEC_POLICY
    )


def readout_value(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(readout_shape())
    # Fully specified inline values make run/config tests independent of calibration.
    for path, item in {
        "pulse_cfg.ch": 0,
        "pulse_cfg.nqz": 2,
        "pulse_cfg.freq": 0.0,
        "pulse_cfg.gain": 0.1,
        "pulse_cfg.waveform.length": 1.0,
        "ro_cfg.ro_ch": 0,
        "ro_cfg.ro_freq": 0.0,
        "ro_cfg.ro_length": 1.0,
        "ro_cfg.trig_offset": 0.0,
    }.items():
        value.with_field(path, item)
    return ReferenceValue(make_custom_reference_key("Pulse Readout"), value)


ROLE_FACTORIES = {
    "readout": RoleFactorySpec("module", readout_shape, readout_value),
}
