"""Measure adapter integration for shared finished-cfg lowering ports."""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    EvalValue,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import SweepCfg

_PULSE = {
    "type": "pulse",
    "ch": 3,
    "nqz": 2,
    "freq": 5000.0,
    "gain": 0.75,
    "phase": 0.0,
    "pre_delay": 0.0,
    "post_delay": 0.0,
    "waveform": {"style": "const", "length": 0.05},
}


def test_measure_ports_integrate_metadict_and_sweepcfg() -> None:
    md = MetaDict()
    md.start = 1.0
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "frequency": ScalarSpec("Frequency", float),
                "sweep": SweepSpec("Sweep"),
            }
        ),
        value=CfgSectionValue(
            fields={
                "frequency": EvalValue("start + 1"),
                "sweep": SweepValue(EvalValue("start"), 2.0, 5),
            }
        ),
    )

    raw = schema_to_raw_dict(schema, md, ModuleLibrary())

    assert raw["frequency"] == 2.0
    sweep = raw["sweep"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.model_dump() == {
        "start": 1.0,
        "stop": 2.0,
        "expts": 5,
        "step": 0.25,
    }


def test_measure_reference_missing_then_relinks_with_embedded_snapshot() -> None:
    snapshot_raw = {**_PULSE, "gain": 0.25}
    _, snapshot = module_cfg_to_value(snapshot_raw)
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={"drive": ModuleRefSpec(allowed=[make_pulse_spec()])}
        ),
        value=CfgSectionValue(fields={"drive": ModuleRefValue("drive", snapshot)}),
    )
    ml = ModuleLibrary()
    ml.register_module(drive=_PULSE)
    ml.delete_module("drive")

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, None, ml)
    assert str(exc_info.value) == "Unknown module reference: 'drive'"

    ml.register_module(drive={**_PULSE, "gain": 0.9})
    raw = schema_to_raw_dict(schema, None, ml)
    assert raw["drive"]["gain"] == 0.25  # type: ignore[index]
