"""Measure adapter integration for shared finished-cfg lowering ports."""

from __future__ import annotations

import pytest
from zcu_tools.device import DeviceManager, FakeDeviceInfo
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.gui.app.measure.adapter import RunRequest, SessionEnv
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.app.measure.specs import make_pulse_spec
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
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from tests.gui.app.measure.adapter._cfg_fakes import CalibrationAdapter, CalibrationCfg

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


@pytest.mark.parametrize("device_name", ["sensor", "probe"])
def test_run_cfg_assembly_uses_only_the_frozen_device_snapshot(
    device_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = CalibrationAdapter()
    ctx = SessionEnv(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "dev": CfgSectionSpec(
                    fields={
                        device_name: CfgSectionSpec(
                            fields={"label": ScalarSpec("Label", str)}
                        ),
                    }
                )
            }
        ),
        value=CfgSectionValue(
            fields={
                "dev": CfgSectionValue(
                    fields={
                        device_name: CfgSectionValue(
                            fields={"label": DirectValue("frozen_label")}
                        ),
                    }
                )
            }
        ),
    )
    raw = schema_to_raw_dict(schema, ctx.md, ctx.ml)
    device = FakeDeviceInfo(address="frozen", value=0.125)
    request = RunRequest(soc=None, soccfg=None, device_snapshot={device_name: device})
    device.value = 99.0

    def unexpected_device_read(_manager):
        pytest.fail("Frozen Run must not read live devices")

    monkeypatch.setattr(DeviceManager, "get_all_info", unexpected_device_read)
    cfg = adapter.build_exp_cfg(raw, request)

    assert cfg.dev is not None
    observed = cfg.dev[device_name]
    assert isinstance(observed, FakeDeviceInfo)
    assert observed.label == "frozen_label"
    assert observed.value == 0.125
    assert observed.address == "frozen"
    assert observed is not device
    assert device.label != "frozen_label"
    assert raw["dev"] == {device_name: {"label": "frozen_label"}}


def _calibration_schema(mode: str) -> tuple[CalibrationAdapter, SessionEnv, CfgSchema]:
    adapter = CalibrationAdapter()
    ctx = SessionEnv(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    schema = adapter.make_default_cfg(ctx)
    schema.value.fields["mode"] = DirectValue(mode)
    return adapter, ctx, schema


@pytest.mark.parametrize("mode", ["plain", "calibrated"])
def test_optional_calibration_parse_error_is_not_ignored_by_mode(mode: str) -> None:
    _, ctx, schema = _calibration_schema(mode)
    calibration = schema.value.fields["calibration"]
    assert isinstance(calibration, CfgSectionValue)
    calibration.fields["frequency"] = EvalValue("missing_calibration")

    with pytest.raises(RuntimeError, match="calibration.frequency|missing_calibration"):
        schema_to_raw_dict(schema, ctx.md, ctx.ml)


@pytest.mark.parametrize("mode", ["plain", "calibrated"])
def test_resolved_calibration_is_assembled_from_cfg_not_metadict(mode: str) -> None:
    adapter, ctx, schema = _calibration_schema(mode)
    calibration = schema.value.fields["calibration"]
    assert isinstance(calibration, CfgSectionValue)
    for key, value in {"frequency": 6000.0, "width": 10.0, "phase": 0.1}.items():
        calibration.fields[key] = DirectValue(value)
    ctx.md.frequency = 7000.0
    ctx.md.width = 20.0
    ctx.md.phase = 0.2
    raw = schema_to_raw_dict(schema, ctx.md, ctx.ml)
    cfg = adapter.build_exp_cfg(
        raw, RunRequest(soc=None, soccfg=None, device_snapshot={})
    )

    assert cfg.mode == mode
    assert cfg.calibration.model_dump() == {
        "frequency": 6000.0,
        "width": 10.0,
        "phase": 0.1,
    }


def test_empty_optional_calibration_uses_model_defaults() -> None:
    adapter, ctx, schema = _calibration_schema("plain")
    raw = schema_to_raw_dict(schema, ctx.md, ctx.ml)
    cfg = adapter.build_exp_cfg(
        raw, RunRequest(soc=None, soccfg=None, device_snapshot={})
    )
    assert cfg.mode == "plain"
    assert cfg.calibration == CalibrationCfg()


@pytest.mark.parametrize(
    "field, value",
    [
        ("frequency", None),
        ("width", None),
        ("phase", None),
        ("frequency", 0.0),
        ("frequency", -1.0),
        ("width", 0.0),
        ("width", -1.0),
    ],
)
def test_adapter_run_rejects_invalid_calibration_before_device_io(
    field: str, value: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, ctx, schema = _calibration_schema("calibrated")
    raw = schema_to_raw_dict(schema, ctx.md, ctx.ml)
    raw["calibration"] = {
        "frequency": 6000.0,
        "width": 10.0,
        "phase": 0.1,
        field: value,
    }

    def unexpected_device_read(_manager):
        pytest.fail("Invalid calibration reached device I/O")

    monkeypatch.setattr(DeviceManager, "get_all_info", unexpected_device_read)
    with pytest.raises(ValueError, match=field):
        adapter.run(
            RunRequest(soc=None, soccfg=None, device_snapshot={}),
            raw,
            context=RunContext(
                None,
                None,
                Plots(NonPresentingHost()),
                devices={},
                cancel_signal=StopSignal(),
            ),
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
