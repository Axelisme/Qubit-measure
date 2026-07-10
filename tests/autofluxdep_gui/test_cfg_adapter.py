"""Autoflux-local module conversion and lowering adapter tests."""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.autofluxdep.cfg.lowering import schema_to_raw_dict
from zcu_tools.gui.app.autofluxdep.cfg.module_adapter import (
    module_cfg_to_value,
    pulse_module_ref_spec,
    pulse_readout_module_ref_spec,
    waveform_cfg_to_value,
)
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    EvalValue,
    ReferenceSpec,
    ReferenceValue,
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

_READOUT = {
    "type": "readout/pulse",
    "pulse_cfg": {
        "ch": 0,
        "nqz": 2,
        "freq": 7000.0,
        "gain": 0.5,
        "phase": 0.0,
        "pre_delay": 0.0,
        "post_delay": 0.0,
        "waveform": {"style": "const", "length": 1.0},
    },
    "ro_cfg": {
        "ro_ch": 0,
        "ro_freq": 7000.0,
        "ro_length": 0.9,
        "trig_offset": 0.6,
    },
}

_LOWERED_READOUT = {
    "type": "readout/pulse",
    "pulse_cfg": {
        "type": "pulse",
        "waveform": {"style": "const", "length": 1.0},
        "ch": 0,
        "nqz": 2,
        "freq": 7000.0,
        "gain": 0.5,
        "phase": 0.0,
        "pre_delay": 0.0,
        "post_delay": 0.0,
    },
    "ro_cfg": {
        "type": "readout/direct",
        "ro_ch": 0,
        "ro_freq": 7000.0,
        "ro_length": 0.9,
        "trig_offset": 0.6,
    },
}


def _lower_converted(spec: CfgSectionSpec, value: CfgSectionValue) -> dict[str, object]:
    return schema_to_raw_dict(CfgSchema(spec=spec, value=value), None, None)


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ({"style": "const", "length": 1.0}, "Const"),
        ({"style": "cosine", "length": 1.0}, "Cosine"),
        ({"style": "gauss", "length": 1.0, "sigma": 0.2}, "Gauss"),
        (
            {
                "style": "drag",
                "length": 1.0,
                "sigma": 0.2,
                "delta": 3.0,
                "alpha": 0.5,
            },
            "DRAG",
        ),
        ({"style": "arb", "data": "waveform-key"}, "Arb"),
        (
            {
                "style": "flat_top",
                "length": 1.0,
                "raise_waveform": {"style": "cosine", "length": 0.1},
            },
            "FlatTop",
        ),
    ],
)
def test_autoflux_converts_all_supported_waveform_styles(
    raw: dict[str, object], label: str
) -> None:
    spec, value = waveform_cfg_to_value(raw)

    assert spec.label == label
    assert _lower_converted(spec, value) == raw


@pytest.mark.parametrize(
    ("raw", "label", "expected"),
    [
        (_PULSE, "Pulse", _PULSE),
        (_READOUT, "Pulse Readout", _LOWERED_READOUT),
    ],
)
def test_autoflux_converts_supported_default_modules(
    raw: dict[str, object], label: str, expected: dict[str, object]
) -> None:
    spec, value = module_cfg_to_value(raw)

    assert spec.label == label
    assert _lower_converted(spec, value) == expected


def test_autoflux_default_conversion_rejects_direct_readout() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        module_cfg_to_value(
            {
                "type": "readout/direct",
                "ro_ch": 0,
                "ro_freq": 7000.0,
                "ro_length": 1.0,
            }
        )
    assert str(exc_info.value) == "Unsupported module type 'readout/direct'"


def test_autoflux_ports_integrate_expression_and_sweepcfg() -> None:
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


def test_autoflux_reference_missing_then_relinks_with_embedded_snapshot() -> None:
    snapshot_raw = {**_PULSE, "gain": 0.25}
    _, snapshot = module_cfg_to_value(snapshot_raw)
    schema = CfgSchema(
        spec=CfgSectionSpec(fields={"drive": pulse_module_ref_spec()}),
        value=CfgSectionValue(fields={"drive": ReferenceValue("drive", snapshot)}),
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


def _unknown_reference_schema(*, disabled: bool) -> CfgSchema:
    asset_spec = CfgSectionSpec(label="Asset", fields={})
    ref_spec = ReferenceSpec(
        kind="unknown/asset",
        allowed=[asset_spec],
        optional=disabled,
    )
    value = (
        None
        if disabled
        else ReferenceValue("<Custom:Asset>", CfgSectionValue(fields={}))
    )
    return CfgSchema(
        spec=CfgSectionSpec(fields={"asset": ref_spec}),
        value=CfgSectionValue(fields={"asset": value}),
    )


def test_autoflux_rejects_unknown_custom_reference_kind_exactly() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(_unknown_reference_schema(disabled=False), None, None)

    assert str(exc_info.value) == (
        "Config field 'asset' uses unsupported reference kind 'unknown/asset'; "
        "allowed kinds: module, waveform"
    )


def test_autoflux_rejects_unknown_disabled_reference_kind_exactly() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(_unknown_reference_schema(disabled=True), None, None)

    assert str(exc_info.value) == (
        "Config field 'asset' uses unsupported reference kind 'unknown/asset'; "
        "allowed kinds: module, waveform"
    )


def test_autoflux_waveform_missing_then_relinks_with_snapshot_precedence() -> None:
    snapshot_raw = {"style": "gauss", "length": 0.4, "sigma": 0.08}
    snapshot_spec, snapshot = waveform_cfg_to_value(snapshot_raw)
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "waveform": ReferenceSpec(
                    kind="waveform", allowed=[snapshot_spec], label="Waveform"
                )
            }
        ),
        value=CfgSectionValue(fields={"waveform": ReferenceValue("rise", snapshot)}),
    )
    ml = ModuleLibrary()
    ml.register_waveform(rise=snapshot_raw)
    ml.delete_waveform("rise")

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, None, ml)
    assert str(exc_info.value) == "Unknown waveform reference: 'rise'"

    ml.register_waveform(rise={"style": "gauss", "length": 0.8, "sigma": 0.16})

    assert schema_to_raw_dict(schema, None, ml) == {"waveform": snapshot_raw}


def test_autoflux_waveform_reference_reports_unsupported_shape_exactly() -> None:
    snapshot_spec, snapshot = waveform_cfg_to_value(
        {"style": "gauss", "length": 0.4, "sigma": 0.08}
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "waveform": ReferenceSpec(
                    kind="waveform", allowed=[snapshot_spec], label="Waveform"
                )
            }
        ),
        value=CfgSectionValue(fields={"waveform": ReferenceValue("rise", snapshot)}),
    )
    ml = ModuleLibrary()
    ml.register_waveform(rise={"style": "const", "length": 0.4})

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, None, ml)

    assert str(exc_info.value) == (
        "Library reference 'rise' resolved to unsupported spec "
        "'Const'; allowed labels: Gauss"
    )


def test_autoflux_legal_direct_readout_shape_is_reported_as_unsupported() -> None:
    schema = CfgSchema(
        spec=CfgSectionSpec(fields={"readout": pulse_readout_module_ref_spec()}),
        value=CfgSectionValue(
            fields={"readout": ReferenceValue("readout", CfgSectionValue())}
        ),
    )
    ml = ModuleLibrary()
    ml.register_module(
        readout={
            "type": "readout/direct",
            "ro_ch": 0,
            "ro_freq": 7000.0,
            "ro_length": 1.0,
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(schema, None, ml)

    assert str(exc_info.value) == (
        "Library reference 'readout' resolved to unsupported spec "
        "'Direct Readout'; allowed labels: Pulse Readout"
    )
