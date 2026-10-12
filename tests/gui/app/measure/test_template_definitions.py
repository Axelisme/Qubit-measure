"""Injected definitions drive the existing library creation catalog."""

from zcu_tools import cfg_model
from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES
from zcu_tools.gui.app.measure.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog
from zcu_tools.gui.cfg import (
    CfgSchema,
    DirectValue,
    EvalValue,
    make_default_value,
    read_value_path,
    schema_to_raw,
)
from zcu_tools.gui.session.types import SessionEnv


def test_injected_definitions_provide_module_and_waveform_creation_values() -> None:
    catalog = TemplateCatalog()
    catalog.register_definitions(
        {
            "drive": cfg_model.TemplateDef(
                cfg_model.PulseSpec,
                "Synthetic drive",
                {
                    "freq": cfg_model.Expr("{qubit}.freq"),
                    "gain": 0.2,
                    "waveform.style": "gauss",
                    "waveform.length": 0.1,
                    "waveform.sigma": 0.02,
                },
                default_name="drive_pi",
            ),
            "ramp": cfg_model.TemplateDef(
                cfg_model.CosineWaveformCfg,
                "Synthetic ramp",
                {"length": cfg_model.Expr("{resonator}.length")},
            ),
        }
    )
    assert catalog.list_meta() == [
        {
            "role_id": "drive",
            "label": "Synthetic drive",
            "item_kind": "module",
            "default_name": "drive_pi",
        },
        {
            "role_id": "ramp",
            "label": "Synthetic ramp",
            "item_kind": "waveform",
            "default_name": "",
        },
    ]
    assert [e.template_id for e in catalog.entries_for("waveform")] == ["ramp"]
    env = SessionEnv(parameters=None, soc=None, soccfg=None)
    drive = catalog.get("drive")
    first = drive.make_value(env)
    assert read_value_path(first.value, "freq") == EvalValue("$qubit.freq")
    assert read_value_path(first.value, "gain") == DirectValue(0.2)
    assert read_value_path(first.value, "waveform.style") == DirectValue("gauss")
    assert read_value_path(first.value, "waveform.length") == DirectValue(0.1)
    assert read_value_path(first.value, "waveform.sigma") == DirectValue(0.02)
    first.value.with_field("gain", 0.9)
    second = drive.make_value(env)
    assert read_value_path(second.value, "gain") == DirectValue(0.2)
    ramp = catalog.get("ramp").make_value(env)
    assert read_value_path(ramp.value, "length") == EvalValue("$resonator.length")


def test_empty_seeds_keep_structural_blank_defaults_and_no_name_suggestion() -> None:
    catalog = TemplateCatalog()
    catalog.register_definitions(
        {"pulse:blank": cfg_model.TemplateDef(cfg_model.PulseSpec, "Blank pulse", {})}
    )
    entry = catalog.get("pulse:blank")
    env = SessionEnv(parameters=None, soc=None, soccfg=None)
    expected_spec = PROGRAM_SHAPES.module("pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)
    assert schema_to_raw(CfgSchema(entry.shape(), entry.make_value(env).value)) == (
        schema_to_raw(CfgSchema(expected_spec, make_default_value(expected_spec)))
    )
    assert entry.default_name == ""
