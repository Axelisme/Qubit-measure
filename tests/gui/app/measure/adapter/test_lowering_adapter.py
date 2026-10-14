"""Measure adapter integration for shared finished-cfg lowering ports."""

from __future__ import annotations

import pytest
from zcu_tools.cfg_model import module_spec_from_raw
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.app.measure.specs import make_pulse_spec
from zcu_tools.gui.app.measure.state import State
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
from zcu_tools.gui.cfg.resource import CfgEdit, CfgResource, CfgStatus
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument

from tests.gui.app.measure._cfg_fakes import PublishedHost, parameter_capture
from tests.gui.app.measure._components import QubitSchema
from tests.gui.app.measure.adapter._cfg_fakes import CalibrationAdapter


@pytest.fixture
def device_name_schema() -> CfgSchema:
    return CfgSchema(
        CfgSectionSpec(
            fields={
                "flux_dev": ScalarSpec(
                    "Flux", str, choices_source="devices", required=True
                )
            }
        ),
        CfgSectionValue(fields={"flux_dev": EvalValue("$flux_dev")}),
    )


def test_device_name_resource_refresh_preview_and_accept_share_fixed_sources(
    device_name_schema: CfgSchema,
) -> None:
    base = parameter_capture()
    first = base.bind_aliases((AliasRow("flux_dev", "yoko1"),), {})
    state = State(SessionEnv(parameters=first, soc=None, soccfg=None))
    bindings = MeasureCfgBindings(PublishedHost(state))
    current = first

    def resolution():
        return bindings.snapshot((), captured_values={}, parameters=current)

    resource = CfgResource(lambda: device_name_schema, resolution=resolution)
    try:
        observed = resource.observe()
        assert observed.status is CfgStatus.VALID
        assert resource.accept().values == {"flux_dev": "yoko1"}
        current = base.bind_aliases((AliasRow("flux_dev", "yoko2"),), {})
        preview = resource.preview(resolution=resolution())
        assert preview.tree.children["flux_dev"].value == EvalValue(
            "$flux_dev", resolved="yoko2"
        )
        assert resource.observe() == observed
        assert resource.accept(resolution=resolution()).values == {"flux_dev": "yoko2"}
        changed = resource.edit((CfgEdit(("flux_dev",), EvalValue("$flux_dev")),))
        assert changed.status is CfgStatus.VALID
        current = base.bind_aliases((AliasRow("flux_dev", "yoko3"),), {})
        resource.refresh()
        assert resource.accept().values == {"flux_dev": "yoko3"}
        assert resource.snapshot_inputs().value.fields["flux_dev"] == EvalValue(
            "$flux_dev"
        )
        current = base.bind_aliases((AliasRow("flux_dev", ""),), {})
        invalid = resource.refresh()
        assert invalid.status is CfgStatus.INVALID
        assert any(
            "flux_dev" in diagnostic.message for diagnostic in invalid.diagnostics
        )
    finally:
        resource.revoke()


@pytest.mark.parametrize("parameters", [None, parameter_capture().bind_aliases((), {})])
def test_name_lowering_never_uses_old_resolved_when_context_is_missing(
    parameters,
) -> None:
    schema = CfgSchema(
        CfgSectionSpec(fields={"flux_dev": ScalarSpec("Flux", str)}),
        CfgSectionValue(fields={"flux_dev": EvalValue("$flux_dev", resolved="yoko1")}),
    )
    with pytest.raises(RuntimeError, match="flux_dev"):
        schema_to_raw_dict(schema, parameters)
    captured = parameter_capture().bind_aliases((AliasRow("flux_dev", "yoko2"),), {})
    assert schema_to_raw_dict(schema, captured) == {"flux_dev": "yoko2"}


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
    from tests.gui.app.measure._cfg_fakes import parameter_capture
    from tests.gui.app.measure._components import QubitSchema, ResonatorSchema

    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(kind="resonator", freq=1.0),
            "Q1": QubitSchema(kind="qubit/transmon"),
        }
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "frequency": ScalarSpec("Frequency", float),
                "sweep": SweepSpec("Sweep"),
            }
        ),
        value=CfgSectionValue(
            fields={
                "frequency": EvalValue("R1.freq + 1"),
                "sweep": SweepValue(EvalValue("R1.freq"), 2.0, 5),
            }
        ),
    )

    raw = schema_to_raw_dict(schema, parameters)

    assert raw["frequency"] == 2.0
    sweep = raw["sweep"]
    assert isinstance(sweep, dict)
    assert sweep == {
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
        value=CfgSectionValue(
            fields={"drive": ReferenceValue("Q1.pulses.drive", snapshot)}
        ),
    )
    ml = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
    ).replaced_module("Q1.pulses.drive", module_spec_from_raw(_PULSE))
    ml = ml.deleted("Q1.pulses.drive", kind="module")

    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(
            schema, parameter_capture({"Q1": QubitSchema(kind="qubit/transmon")}, ml)
        )
    assert str(exc_info.value) == "Unknown module reference: 'Q1.pulses.drive'"

    ml = ml.replaced_module(
        "Q1.pulses.drive", module_spec_from_raw({**_PULSE, "gain": 0.9})
    )
    raw = schema_to_raw_dict(
        schema, parameter_capture({"Q1": QubitSchema(kind="qubit/transmon")}, ml)
    )
    assert raw["drive"]["gain"] == 0.25  # type: ignore[index]


def _calibration_schema(mode: str) -> tuple[CalibrationAdapter, SessionEnv, CfgSchema]:
    adapter = CalibrationAdapter()
    ctx = SessionEnv(parameters=parameter_capture(), soc=None, soccfg=None)
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
        schema_to_raw_dict(schema, ctx.parameters)


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
        schema_to_raw_dict(_unknown_reference_schema(disabled=False), None)

    assert str(exc_info.value) == (
        "Config field 'asset' uses unsupported reference kind 'unknown/asset'; "
        "allowed kinds: module, waveform"
    )


def test_measure_rejects_unknown_disabled_reference_kind_exactly() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        schema_to_raw_dict(_unknown_reference_schema(disabled=True), None)

    assert str(exc_info.value) == (
        "Config field 'asset' uses unsupported reference kind 'unknown/asset'; "
        "allowed kinds: module, waveform"
    )
