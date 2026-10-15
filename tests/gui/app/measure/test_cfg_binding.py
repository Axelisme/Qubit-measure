from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ModuleCfg, module_spec_from_raw
from zcu_tools.experiment.cfg_editing import UnknownProgramShapeError
from zcu_tools.format_version import YamlMap
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
    SourceRevision,
)
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import RoleResolutionError

from tests.gui.app.measure._cfg_fakes import PublishedHost, parameter_capture
from tests.gui.app.measure._components import QubitSchema, ResonatorSchema


def _bindings(
    library: LibrarySnapshot | None = None,
) -> tuple[MeasureCfgBindings, MagicMock]:
    bindings, host = _published_bindings()
    host.read_context.return_value = parameter_capture(
        {
            "Q1": QubitSchema(kind="qubit/transmon"),
            "R1": ResonatorSchema(kind="resonator"),
        },
        library,
    )
    return bindings, host


def _published_bindings() -> tuple[MeasureCfgBindings, MagicMock]:
    parameters = parameter_capture()
    state = State(SessionEnv(parameters=parameters, soc=None, soccfg=None))
    host = MagicMock(spec=PublishedHost, wraps=PublishedHost(state))
    host.read_context.return_value = parameters
    host.list_device_names.return_value = ["flux"]
    host.arb_waveforms.list_data_keys.return_value = ["asset"]
    return MeasureCfgBindings(host), host


class _InspectionLibrary(LibrarySnapshot):
    """Supply entry facts while rejecting materialization during shape listing."""

    def __init__(
        self, document: ModuleLibraryDocument, *, corrupt: YamlMap | None = None
    ) -> None:
        super().__init__(document)
        self.corrupt = corrupt

    def entry_value(self, path: str, *, kind: Literal["module", "waveform"]) -> YamlMap:
        if self.corrupt is not None:
            return self.corrupt
        return super().entry_value(path, kind=kind)

    def get_module(self, path: str) -> ModuleCfg:
        raise AssertionError("keys must not materialize modules")


def _pulse() -> YamlMap:
    return {
        "type": "pulse",
        "waveform": {"style": "const", "length": 0.1},
        "ch": 0,
        "nqz": 1,
        "freq": 5000.0,
        "gain": 0.2,
        "phase": 0.0,
        "pre_delay": 0.0,
        "post_delay": 0.0,
    }


def test_measure_snapshot_detaches_metadata_options_and_captures() -> None:
    bindings, host = _bindings()
    parameters = parameter_capture(offset=2.0)
    captures = {"device.flux.value": 0.2}
    basis = (
        SourceRevision("context", 3),
        SourceRevision("device:flux", 4),
    )
    frozen = bindings.snapshot(basis, captured_values=captures, parameters=parameters)
    parameters = parameter_capture(offset=9.0)
    captures["device.flux.value"] = 0.7
    host.list_device_names.return_value.append("later")
    host.arb_waveforms.list_data_keys.return_value.clear()

    assert frozen.source_basis == basis
    assert frozen.evaluate_expression("general.ext.offset * 2") == 4.0
    assert frozen.read_capture("general.ext.offset") == 2.0
    assert frozen.read_capture("device.flux.value") == 0.2
    assert frozen.provide_options("devices") == ("flux",)
    assert frozen.provide_options("arb_waveforms") == ("asset",)
    host.read_value_source.assert_not_called()
    newer = bindings.snapshot(basis, captured_values=captures, parameters=parameters)
    assert newer.evaluate_expression("general.ext.offset * 2") == 18.0
    assert newer.read_capture("device.flux.value") == 0.7


def test_source_snapshot_tracks_context_and_device_set_aba_without_live_reads() -> None:
    bindings, host = _published_bindings()
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    captures = {"device.flux.value": 0.2}
    first = bindings.snapshot_from_state(
        state, captured_values=captures, parameters=host.read_context()
    )
    assert first.source_basis == (
        SourceRevision("context", 0),
        SourceRevision(DEVICE_SET_VERSION_KEY, 0),
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
    second = bindings.snapshot_from_state(
        state, captured_values=captures, parameters=host.read_context()
    )
    assert second.source_basis == (
        SourceRevision("context", 1),
        SourceRevision(DEVICE_SET_VERSION_KEY, 1),
        SourceRevision("device:flux", 1),
    )
    captures["device.flux.value"] = 0.7
    assert second.read_capture("device.flux.value") == 0.2
    state.remove_device("flux")
    state.put_device(device)
    recreated = bindings.snapshot_from_state(
        state, captured_values=captures, parameters=host.read_context()
    )
    assert recreated.source_basis[-1] == second.source_basis[-1]
    assert recreated.source_basis != second.source_basis
    assert recreated.read_capture("device.flux.value") == 0.7
    host.read_value_source.assert_not_called()


def test_measure_snapshot_missing_capture_is_precondition_failure() -> None:
    bindings, host = _published_bindings()
    frozen = bindings.snapshot((), captured_values={}, parameters=host.read_context())
    with pytest.raises(CfgPreconditionError) as caught:
        frozen.read_capture("general.ext.missing")
    assert caught.value.reason is CfgPreconditionReason.CAPTURE_UNAVAILABLE
    assert str(caught.value) == "Capture source 'general.ext.missing' is unavailable"
    with pytest.raises(RoleResolutionError) as device_error:
        frozen.read_capture("device.flux.value")
    assert device_error.value.reason == "Unknown path prefix 'device'"
    assert device_error.value.reason in str(device_error.value)
    host.read_value_source.assert_not_called()
    with pytest.raises(RuntimeError, match="Unsupported measure cfg option source"):
        frozen.provide_options("unknown")


def test_measure_snapshot_catalog_isolated_from_live_and_returned_values() -> None:
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
    ).replaced_module("Q1.pulses.drive", module_spec_from_raw(_pulse()))
    bindings, host = _bindings(library)
    frozen = bindings.snapshot((), captured_values={}, parameters=host.read_context())
    host.read_context.return_value = parameter_capture(
        {"Q1": QubitSchema(kind="qubit/transmon")},
        library.deleted("Q1.pulses.drive", kind="module"),
    )
    assert frozen.references.keys("module", frozenset({"Pulse"})) == (
        "Q1.pulses.drive",
    )
    first = frozen.references.resolve("module", "Q1.pulses.drive")
    assert first is not None and first.value is not None
    original = first.value.fields["freq"]
    assert original == DirectValue(5000.0)
    first.value.fields["freq"] = DirectValue(1.0)
    again = frozen.references.resolve("module", "Q1.pulses.drive")
    assert again is not None and again.value is not None
    assert again.value.fields["freq"] == original
    assert bindings.resolve("module", "Q1.pulses.drive") is None


def test_measure_resource_preserves_capture_and_refreshes_dynamic_metadata() -> None:
    bindings, host = _bindings()
    parameters = parameter_capture(offset=2.0)
    captures = {"device.flux.value": 0.2}
    schema = CfgSchema(
        CfgSectionSpec(fields={"value": ScalarSpec("Value", float)}),
        CfgSectionValue({"value": DirectValue(0.0)}),
    )
    resource = CfgResource(
        lambda: schema,
        # This generic resource has caller-owned sources, not Run metadata.
        resolution=lambda: bindings.snapshot(
            (), captured_values=captures, parameters=parameters
        ),
    )
    resource.edit(
        (CfgEdit(("value",), EvalValue("@{device.flux.value} + general.ext.offset")),),
    )
    assert resource.accept().values["value"] == pytest.approx(2.2)
    captures["device.flux.value"] = 0.7
    parameters = parameter_capture(offset=5.0)
    resource.refresh()
    assert resource.accept().values["value"] == pytest.approx(5.2)
    host.read_value_source.assert_not_called()


def test_measure_option_provider_owns_device_and_arb_catalogs() -> None:
    bindings, _ = _published_bindings()

    assert bindings.provide_options("devices") == ["flux"]
    assert bindings.provide_options("arb_waveforms") == ["asset"]
    with pytest.raises(RuntimeError, match="Unsupported measure cfg option source"):
        bindings.provide_options("unknown")


def test_measure_catalog_filters_by_shape_and_materializes_resolution() -> None:
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
    ).replaced_module("Q1.pulses.drive", module_spec_from_raw(_pulse()))
    library = library.replaced_module(
        "R1.readout.probe",
        module_spec_from_raw(
            {
                "ro_ch": 0,
                "ro_length": 1.0,
                "trig_offset": 0.1,
                "type": "readout/pulse",
                "freq": 6000.0,
                "gen_ch": 0,
                "nqz": 1,
                "gain": 0.1,
                "waveform": {"style": "const", "length": 1.0},
            }
        ),
    )
    bindings, _ = _bindings(library)

    assert bindings.keys("module", frozenset({"Pulse"})) == ("Q1.pulses.drive",)
    resolved = bindings.resolve("module", "Q1.pulses.drive")
    assert resolved is not None
    assert resolved.label == "Pulse"
    assert resolved.value is not None
    assert bindings.resolve("module", "Q1.pulses.missing") is None


def test_measure_catalog_corrupt_entry_fast_fails_during_enumeration() -> None:
    document = (
        LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
        )
        .replaced_module("Q1.pulses.corrupt", module_spec_from_raw(_pulse()))
        .document()
    )
    library = _InspectionLibrary(document, corrupt={"type": "not-a-module"})
    bindings, _ = _bindings(library)

    with pytest.raises(UnknownProgramShapeError, match="Unknown module program shape"):
        bindings.keys("module", frozenset({"Pulse"}))


def test_measure_keys_inspects_shapes_without_converter_or_normalization() -> None:
    document = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
    ).replaced_module("Q1.pulses.drive", module_spec_from_raw(_pulse()))
    document = document.replaced_module(
        "R1.readout.probe",
        module_spec_from_raw(
            {
                "ro_ch": 0,
                "ro_length": 1.0,
                "trig_offset": 0.1,
                "type": "readout/pulse",
                "freq": 6000.0,
                "gen_ch": 0,
                "nqz": 1,
                "gain": 0.1,
                "waveform": {"style": "const", "length": 1.0},
            }
        ),
    )
    bindings, _ = _bindings(_InspectionLibrary(document.document()))

    assert bindings.keys("module", frozenset({"Pulse", "Pulse Readout"})) == (
        "Q1.pulses.drive",
        "R1.readout.probe",
    )


def test_measure_resolve_calls_app_converter_exactly_once() -> None:
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
    ).replaced_module("Q1.pulses.drive", module_spec_from_raw(_pulse()))
    bindings, _ = _bindings(library)

    assert bindings.resolve("module", "Q1.pulses.missing") is None
    resolved = bindings.resolve("module", "Q1.pulses.drive")
    assert resolved is not None and resolved.value is not None
    assert resolved.label == "Pulse"
    raw = library.get_module("Q1.pulses.drive").to_dict()
    assert resolved.value.fields["freq"] == DirectValue(raw["freq"])
    assert resolved.value.fields["gain"] == DirectValue(raw["gain"])
    assert resolved.value.fields["ch"] == DirectValue(raw["ch"])
