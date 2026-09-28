"""Measure adapter integration for shared finished-cfg lowering ports."""

from __future__ import annotations

import pytest
from zcu_tools.device import FakeDeviceInfo, GlobalDeviceManager
from zcu_tools.experiment.v2_gui.adapters.onetone.flux_dep import OneToneFluxDepAdapter
from zcu_tools.experiment.v2_gui.adapters.onetone.freq import OneToneFreqAdapter
from zcu_tools.experiment.v2_gui.adapters.twotone.flux_dep import FluxDepAdapter
from zcu_tools.gui.app.main.adapter import ExpContext, RunRequest
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.app.main.specs import make_pulse_spec
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
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
            fields={"drive": ReferenceSpec(kind="module", allowed=[make_pulse_spec()])}
        ),
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


@pytest.mark.parametrize("adapter_type", [OneToneFluxDepAdapter, FluxDepAdapter])
def test_flux_run_assembles_lowered_cfg_with_frozen_device_snapshot(
    adapter_type: type[OneToneFluxDepAdapter] | type[FluxDepAdapter],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = adapter_type()
    ctx = ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    schema = adapter.make_default_cfg(ctx)
    raw = schema_to_raw_dict(schema, ctx.md, ctx.ml)
    device = FakeDeviceInfo(address="frozen", value=0.125)
    request = RunRequest(soc=None, soccfg=None, device_snapshot={"flux_yoko": device})

    def unexpected_device_read():
        pytest.fail("Frozen Run must not read live devices")

    monkeypatch.setattr(GlobalDeviceManager, "get_all_info", unexpected_device_read)
    cfg = adapter.build_exp_cfg(raw, request)

    assert cfg.dev["flux_yoko"].label == "flux_dev"
    assert isinstance(cfg.dev["flux_yoko"], FakeDeviceInfo)
    assert cfg.dev["flux_yoko"].value == 0.125
    assert cfg.dev["flux_yoko"].address == "frozen"
    assert device.label != "flux_dev"
    assert raw["dev"] == {"flux_dev": "flux_yoko"}
    assert cfg.sweep.flux.expts > 0
    assert cfg.sweep.freq.expts > 0


def _frequency_schema(mode: str) -> tuple[OneToneFreqAdapter, ExpContext, CfgSchema]:
    adapter = OneToneFreqAdapter()
    ctx = ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    schema = adapter.make_default_cfg(ctx)
    schema.value.fields["sampling_mode"] = DirectValue(mode)
    return adapter, ctx, schema


@pytest.mark.parametrize("mode", ["linear", "homophasal"])
def test_optional_calibration_parse_error_is_not_ignored_by_mode(mode: str) -> None:
    _, ctx, schema = _frequency_schema(mode)
    calibration = schema.value.fields["homophasal"]
    assert isinstance(calibration, CfgSectionValue)
    calibration.fields["r_f"] = EvalValue("missing_calibration")

    with pytest.raises(RuntimeError, match="homophasal.r_f|missing_calibration"):
        schema_to_raw_dict(schema, ctx.md, ctx.ml)


@pytest.mark.parametrize("mode", ["linear", "homophasal"])
def test_resolved_calibration_is_assembled_from_cfg_not_metadict(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, ctx, schema = _frequency_schema(mode)
    calibration = schema.value.fields["homophasal"]
    assert isinstance(calibration, CfgSectionValue)
    for key, value in {"r_f": 6000.0, "rf_w": 10.0, "theta0": 0.1}.items():
        calibration.fields[key] = DirectValue(value)
    ctx.md.r_f = 7000.0
    ctx.md.rf_w = 20.0
    ctx.md.theta0 = 0.2
    monkeypatch.setattr(GlobalDeviceManager, "get_all_info", lambda: {})

    raw = schema_to_raw_dict(schema, ctx.md, ctx.ml)
    cfg = adapter.build_exp_cfg(
        raw, RunRequest(soc=None, soccfg=None, device_snapshot={})
    )

    if mode == "linear":
        assert cfg.homophasal is None
    else:
        assert cfg.homophasal is not None
        assert cfg.homophasal.model_dump() == {
            "r_f": 6000.0,
            "rf_w": 10.0,
            "theta0": 0.1,
        }


def test_empty_optional_calibration_allows_linear_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, ctx, schema = _frequency_schema("linear")
    monkeypatch.setattr(GlobalDeviceManager, "get_all_info", lambda: {})
    raw = schema_to_raw_dict(schema, ctx.md, ctx.ml)
    cfg = adapter.build_exp_cfg(
        raw, RunRequest(soc=None, soccfg=None, device_snapshot={})
    )
    assert cfg.sampling_mode == "linear"
    assert cfg.homophasal is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("r_f", None),
        ("rf_w", None),
        ("theta0", None),
        ("r_f", 0.0),
        ("r_f", -1.0),
        ("rf_w", 0.0),
        ("rf_w", -1.0),
    ],
)
def test_adapter_run_rejects_invalid_calibration_before_device_io(
    field: str, value: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, ctx, schema = _frequency_schema("homophasal")
    calibration = schema.value.fields["homophasal"]
    assert isinstance(calibration, CfgSectionValue)
    calibration.fields["r_f"] = DirectValue(6000.0)
    calibration.fields["rf_w"] = DirectValue(10.0)
    calibration.fields["theta0"] = DirectValue(0.1)
    calibration.fields[field] = DirectValue(value)

    def unexpected_device_read():
        pytest.fail("Invalid calibration reached device I/O")

    monkeypatch.setattr(GlobalDeviceManager, "get_all_info", unexpected_device_read)
    with pytest.raises(ValueError, match=field):
        adapter.run(
            RunRequest(soc=None, soccfg=None, device_snapshot={}),
            schema_to_raw_dict(schema, ctx.md, ctx.ml),
        )


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


def test_measure_rejects_unknown_custom_reference_kind_exactly() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(_unknown_reference_schema(disabled=False), None, None)

    assert str(exc_info.value) == (
        "Config field 'asset' uses unsupported reference kind 'unknown/asset'; "
        "allowed kinds: module, waveform"
    )


def test_measure_rejects_unknown_disabled_reference_kind_exactly() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(_unknown_reference_schema(disabled=True), None, None)

    assert str(exc_info.value) == (
        "Config field 'asset' uses unsupported reference kind 'unknown/asset'; "
        "allowed kinds: module, waveform"
    )
