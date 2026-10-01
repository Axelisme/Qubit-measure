from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import zcu_tools.gui.app.measure.cfg_binding as binding_module
from zcu_tools.experiment.cfg_editing import ProgramShape, UnknownProgramShapeError
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.state import (
    DEVICE_SET_VERSION_KEY,
    DeviceState,
    DeviceStatus,
    State,
)
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgPreconditionError,
    CfgPreconditionReason,
    CfgResource,
    CfgRevision,
    SourceRevision,
)
from zcu_tools.resources.context import MetaDict, ModuleLibrary


def _bindings(ml: ModuleLibrary | None = None) -> tuple[MeasureCfgBindings, MagicMock]:
    host = MagicMock()
    host.get_current_md.return_value = MetaDict()
    host.get_current_ml.return_value = ml or ModuleLibrary()
    host.list_device_names.return_value = ["flux"]
    host.arb_waveforms.list_data_keys.return_value = ["asset"]
    return MeasureCfgBindings(host), host


def _pulse() -> dict[str, object]:
    return {
        "type": "pulse",
        "waveform": {"style": "const", "length": 0.1},
        "ch": 0,
        "freq": 5000.0,
        "gain": 0.2,
        "phase": 0.0,
        "pre_delay": 0.0,
        "post_delay": 0.0,
    }


def test_measure_snapshot_detaches_metadata_options_and_captures() -> None:
    bindings, host = _bindings()
    md = host.get_current_md.return_value
    md.update(offset=2.0)
    captures = {"device.flux.value": 0.2}
    basis = (
        SourceRevision("context", CfgRevision(3)),
        SourceRevision("device:flux", CfgRevision(4)),
    )
    frozen = bindings.snapshot(basis, captured_values=captures)
    md.update(offset=9.0)
    captures["device.flux.value"] = 0.7
    host.list_device_names.return_value.append("later")
    host.arb_waveforms.list_data_keys.return_value.clear()

    assert frozen.source_basis == basis
    assert frozen.evaluate_expression("offset * 2") == 4.0
    assert frozen.read_capture("offset") == 2.0
    assert frozen.read_capture("device.flux.value") == 0.2
    assert frozen.provide_options("devices") == ("flux",)
    assert frozen.provide_options("arb_waveforms") == ("asset",)
    host.read_value_source.assert_not_called()
    newer = bindings.snapshot(basis, captured_values=captures)
    assert newer.evaluate_expression("offset * 2") == 18.0
    assert newer.read_capture("device.flux.value") == 0.7


def test_source_snapshot_tracks_context_and_device_set_aba_without_live_reads() -> None:
    bindings, host = _bindings()
    state = State(MagicMock())
    captures = {"device.flux.value": 0.2}
    first = bindings.snapshot_from_state(state, captured_values=captures)
    assert first.source_basis == (
        SourceRevision("context", CfgRevision(0)),
        SourceRevision(DEVICE_SET_VERSION_KEY, CfgRevision(0)),
    )

    state.version.bump("context")
    device = DeviceState(
        name="flux",
        type_name="YOKOGS200",
        address="addr",
        status=DeviceStatus.CONNECTED,
        remember=False,
        info=None,
    )
    state.put_device(device)
    second = bindings.snapshot_from_state(state, captured_values=captures)
    assert second.source_basis == (
        SourceRevision("context", CfgRevision(1)),
        SourceRevision(DEVICE_SET_VERSION_KEY, CfgRevision(1)),
        SourceRevision("device:flux", CfgRevision(1)),
    )
    captures["device.flux.value"] = 0.7
    assert second.read_capture("device.flux.value") == 0.2
    state.remove_device("flux")
    state.put_device(device)
    recreated = bindings.snapshot_from_state(state, captured_values=captures)
    assert recreated.source_basis[-1] == second.source_basis[-1]
    assert recreated.source_basis != second.source_basis
    assert recreated.read_capture("device.flux.value") == 0.7
    host.read_value_source.assert_not_called()


def test_measure_snapshot_missing_capture_is_precondition_failure() -> None:
    bindings, host = _bindings()
    frozen = bindings.snapshot((), captured_values={})
    for name in ("missing", "device.flux.value"):
        with pytest.raises(CfgPreconditionError) as caught:
            frozen.read_capture(name)
        assert caught.value.reason is CfgPreconditionReason.CAPTURE_UNAVAILABLE
    host.read_value_source.assert_not_called()
    with pytest.raises(RuntimeError, match="Unsupported measure cfg option source"):
        frozen.provide_options("unknown")


def test_measure_snapshot_catalog_isolated_from_live_and_returned_values() -> None:
    ml = ModuleLibrary()
    ml.modules["drive"] = cast(Any, _pulse())
    bindings, _ = _bindings(ml)
    frozen = bindings.snapshot((), captured_values={})
    ml.modules.clear()
    assert frozen.references.keys("module", frozenset({"Pulse"})) == ("drive",)
    first = frozen.references.resolve("module", "drive")
    assert first is not None and first.value is not None
    original = first.value.fields["freq"]
    assert original == DirectValue(5000.0)
    first.value.fields["freq"] = DirectValue(1.0)
    again = frozen.references.resolve("module", "drive")
    assert again is not None and again.value is not None
    assert again.value.fields["freq"] == original
    assert bindings.resolve("module", "drive") is None


def test_measure_resource_preserves_capture_and_refreshes_dynamic_metadata() -> None:
    bindings, host = _bindings()
    md = host.get_current_md.return_value
    md.update(offset=2.0)
    captures = {"device.flux.value": 0.2}
    schema = CfgSchema(
        CfgSectionSpec(fields={"value": ScalarSpec("Value", float)}),
        CfgSectionValue({"value": DirectValue(0.0)}),
    )
    resource = CfgResource(
        lambda: schema,
        resolution=lambda: bindings.snapshot((), captured_values=captures),
        make_range=lambda start, stop, *, expts: (start, stop, expts),
    )
    changed = resource.edit(
        resource.observe().ref.revision,
        (CfgEdit(("value",), EvalValue("$device.flux.value + offset")),),
    )
    assert resource.accept(changed.ref.revision).values["value"] == pytest.approx(2.2)
    captures["device.flux.value"] = 0.7
    md.update(offset=5.0)
    refreshed = resource.refresh(changed.ref.revision)
    assert resource.accept(refreshed.ref.revision).values["value"] == pytest.approx(5.2)
    host.read_value_source.assert_not_called()


def test_measure_option_provider_owns_device_and_arb_catalogs() -> None:
    bindings, _ = _bindings()

    assert bindings.provide_options("devices") == ["flux"]
    assert bindings.provide_options("arb_waveforms") == ["asset"]
    with pytest.raises(RuntimeError, match="Unsupported measure cfg option source"):
        bindings.provide_options("unknown")


def test_measure_catalog_filters_by_shape_and_materializes_resolution() -> None:
    ml = ModuleLibrary()
    ml.modules["drive"] = cast(Any, _pulse())
    ml.modules["direct"] = cast(
        Any,
        {
            "type": "readout/direct",
            "ro_ch": 0,
            "ro_freq": 6000.0,
            "ro_length": 1.0,
            "trig_offset": 0.1,
        },
    )
    bindings, _ = _bindings(ml)

    assert bindings.keys("module", frozenset({"Pulse"})) == ("drive",)
    resolved = bindings.resolve("module", "drive")
    assert resolved is not None
    assert resolved.label == "Pulse"
    assert resolved.value is not None
    assert bindings.resolve("module", "missing") is None


def test_measure_catalog_corrupt_entry_fast_fails_during_enumeration() -> None:
    ml = ModuleLibrary()
    ml.modules["corrupt"] = cast(Any, {"type": "not-a-module"})
    bindings, _ = _bindings(ml)

    with pytest.raises(UnknownProgramShapeError, match="Unknown module program shape"):
        bindings.keys("module", frozenset({"Pulse"}))


def test_measure_keys_inspects_shapes_without_converter_or_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TypedModule:
        def __init__(self, discriminator: str) -> None:
            self.type = discriminator

        def to_dict(self):
            raise AssertionError("keys must not normalize typed cfg")

    ml = ModuleLibrary()
    ml.modules["drive"] = cast(Any, TypedModule("pulse"))
    ml.modules["readout"] = cast(Any, TypedModule("readout/direct"))
    bindings, _ = _bindings(ml)
    shape_lookup = MagicMock(side_effect=binding_module.program_shape_for_input)
    converter = MagicMock(side_effect=binding_module.module_cfg_to_value)
    monkeypatch.setattr(binding_module, "program_shape_for_input", shape_lookup)
    monkeypatch.setattr(binding_module, "module_cfg_to_value", converter)
    monkeypatch.setattr(
        ProgramShape,
        "make_spec",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("keys must not construct specs")
        ),
    )

    assert bindings.keys("module", frozenset({"Pulse", "Direct Readout"})) == (
        "drive",
        "readout",
    )
    assert shape_lookup.call_count == 2
    converter.assert_not_called()


def test_measure_resolve_calls_app_converter_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ml = ModuleLibrary()
    ml.modules["drive"] = cast(Any, _pulse())
    bindings, _ = _bindings(ml)
    shape_lookup = MagicMock(side_effect=binding_module.program_shape_for_input)
    converter = MagicMock(side_effect=binding_module.module_cfg_to_value)
    monkeypatch.setattr(binding_module, "program_shape_for_input", shape_lookup)
    monkeypatch.setattr(binding_module, "module_cfg_to_value", converter)

    assert bindings.resolve("module", "missing") is None
    shape_lookup.assert_not_called()
    converter.assert_not_called()
    assert bindings.resolve("module", "drive") is not None
    shape_lookup.assert_not_called()
    assert converter.call_count == 1


def test_measure_resolve_materializes_missing_waveform_style_as_const(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ml = ModuleLibrary()
    ml.waveforms["legacy"] = cast(Any, {})
    bindings, _ = _bindings(ml)
    shape_lookup = MagicMock(side_effect=binding_module.program_shape_for_input)
    converter = MagicMock(side_effect=binding_module.waveform_cfg_to_value)
    monkeypatch.setattr(binding_module, "program_shape_for_input", shape_lookup)
    monkeypatch.setattr(binding_module, "waveform_cfg_to_value", converter)

    resolved = bindings.resolve("waveform", "legacy")

    assert resolved is not None
    assert resolved.label == "Const"
    assert resolved.value is not None
    shape_lookup.assert_not_called()
    assert converter.call_count == 1
