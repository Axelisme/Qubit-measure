"""Captured module defaults through definition instantiation and lowering."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    DirectReadoutCfg,
    PulseReadoutSpec,
    PulseSpec,
)
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.measure.cfg_binding import (
    MeasureCfgBindingHost,
    MeasureCfgBindings,
)
from zcu_tools.gui.cfg import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
    read_value_path,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgPreconditionError,
    CfgResource,
    CfgStatus,
)
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    EntryDraft,
)
from zcu_tools.gui.session.types import ParameterContext, RoleBinding, SessionEnv
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry import ComponentSchema, RoleSpec, component_registry
from zcu_tools.resources.entry.schema import PointDocument

from zcu_lab.v2._support.measure.schema_builder import MeasureCfgBuilder
from zcu_lab.v2._support.measure.seeds import (
    literal,
    module_seed,
    path_seed,
    scaled_path_seed,
)
from zcu_lab.v2._support.measure.spec_helpers import (
    make_pulse_module_spec,
    make_readout_module_spec,
)


class ModuleOwner(ComponentSchema):
    module: dict[str, str]
    calibration: float | str | None = None


@pytest.fixture(autouse=True)
def owner_kind() -> Generator[None]:
    before = component_registry.kinds()
    component_registry.register("seed/module_owner", ModuleOwner)
    try:
        yield
    finally:
        component_registry.unregister("seed/module_owner")
        assert component_registry.kinds() == before


def _environment(
    *,
    slot: str | None = "Q1.pulses.cal",
    broken_waveform: bool = False,
    faults: tuple[ContextSourceFault, ...] = (),
    calibration: float | str | None = None,
    calibration_present: bool = False,
    bound: bool = True,
) -> SessionEnv:
    point = PointDocument.model_validate(
        {
            "format": "zcu.parameter-container",
            "format_version": "1.0",
            "general": {"created_at": "2026-10-08T00:00:00Z"},
            "components": {
                "Q1": {
                    "kind": "seed/module_owner",
                    "module": {} if slot is None else {"x180": slot},
                    **({"calibration": calibration} if calibration_present else {}),
                }
            },
        }
    )
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="1.0",
            components={
                "Q1": ComponentLibrary(
                    pulses={
                        "cal": PulseSpec(
                            waveform=(
                                "Q1.waveforms.missing"
                                if broken_waveform
                                else ConstWaveformCfg(length=2.0)
                            ),
                            ch=1,
                            nqz=2,
                            freq=4300.0,
                            gain=0.3,
                        )
                    }
                )
            },
        )
    )
    parameters = ParameterContext(
        point=point,
        library=library,
        target=EntryDraft(entry_id="entry"),
        roles=RoleBinding(
            "Q1",
            {"qubit": RoleSpec("seed/module_owner")},
            {"qubit": "Q1"} if bound else {},
        ),
        source_faults=faults,
    )
    return SessionEnv(parameters=parameters, soc=None, soccfg=None)


@pytest.mark.parametrize("scaled", [False, True], ids=["path", "scaled-path"])
@pytest.mark.parametrize(
    ("present", "calibration"),
    [(False, None), (True, None), (True, 6.0)],
    ids=["absent", "explicit-null", "numeric"],
)
def test_optional_scalar_seed_lowers_calibration_or_explicit_fallback(
    scaled: bool, present: bool, calibration: float | None
) -> None:
    env = _environment(calibration=calibration, calibration_present=present)
    seed = (
        scaled_path_seed("qubit.calibration", factor=2.0, fallback_value=8.0)
        if scaled
        else path_seed("qubit.calibration", fallback=8.0)
    )
    definition = MeasureCfgBuilder().float("value", label="Value", default=seed).build()
    schema = definition.instantiate(env)
    assert env.parameters is not None
    raw = schema_to_raw_dict(schema, env.parameters)
    expected = (
        calibration * (2.0 if scaled else 1.0) if calibration is not None else 8.0
    )
    assert raw["value"] == expected


@pytest.mark.parametrize("scaled", [False, True], ids=["path", "scaled-path"])
def test_optional_scalar_seed_does_not_hide_non_null_invalid_calibration(
    scaled: bool,
) -> None:
    env = _environment(calibration="bad-calibration", calibration_present=True)
    seed = (
        scaled_path_seed("qubit.calibration", factor=2.0, fallback_value=8.0)
        if scaled
        else path_seed("qubit.calibration", fallback=8.0)
    )
    definition = MeasureCfgBuilder().float("value", label="Value", default=seed).build()
    schema = definition.instantiate(env)
    assert env.parameters is not None
    with pytest.raises(RuntimeError, match="is not numeric"):
        schema_to_raw_dict(schema, env.parameters)


@pytest.mark.parametrize("scaled", [False, True], ids=["path", "scaled-path"])
def test_optional_scalar_seed_does_not_hide_unavailable_source(
    scaled: bool, tmp_path: Path
) -> None:
    env = _environment(
        calibration=None,
        calibration_present=True,
        faults=(
            ContextSourceFault("point", tmp_path / "point.yaml", "OSError", "gone"),
        ),
    )
    seed = (
        scaled_path_seed("qubit.calibration", factor=2.0, fallback_value=8.0)
        if scaled
        else path_seed("qubit.calibration", fallback=8.0)
    )
    with pytest.raises(ContextSourceUnavailableError):
        seed.resolve(env)


@pytest.mark.parametrize("scaled", [False, True], ids=["path", "scaled-path"])
@pytest.mark.parametrize("calibration", [6.0, None], ids=["calibrated", "null"])
def test_unbound_scalar_defaults_publish_invalid_then_refresh_or_reset(
    scaled: bool, calibration: float | None
) -> None:
    environment = _environment(bound=False)
    seed = (
        scaled_path_seed("qubit.calibration", factor=2.0, fallback_value=8.0)
        if scaled
        else path_seed("qubit.calibration", fallback=8.0)
    )
    expression = "2.0 * qubit.calibration" if scaled else "qubit.calibration"
    definition = (
        MeasureCfgBuilder()
        .float("value", label="Value", default=seed)
        .float("neighbor", label="Neighbor", default=3.0)
        .build()
    )
    host = MagicMock(spec=MeasureCfgBindingHost)
    host.list_device_names.return_value = []
    host.arb_waveforms.list_data_keys.return_value = []
    bindings = MeasureCfgBindings(host)
    resource = CfgResource(
        defaults=lambda: definition.instantiate(environment),
        resolution=lambda: bindings.snapshot(
            (), captured_values={}, parameters=environment.parameters
        ),
    )
    initial = resource.observe()
    assert initial.status is CfgStatus.INVALID
    assert any(
        diagnostic.path == ("value",) and "qubit" in diagnostic.message
        for diagnostic in initial.diagnostics
    )
    assert read_value_path(resource.snapshot_inputs().value, "value") == EvalValue(
        expression
    )
    assert read_value_path(resource.snapshot_inputs().value, "neighbor") == DirectValue(
        3.0
    )
    with pytest.raises(CfgPreconditionError):
        resource.accept(initial.ref.revision)

    environment = _environment(calibration=calibration, calibration_present=True)
    refreshed = resource.refresh(initial.ref.revision)
    assert refreshed.ref.cfg_id == initial.ref.cfg_id
    assert refreshed.ref.revision > initial.ref.revision
    assert read_value_path(resource.snapshot_inputs().value, "value") == EvalValue(
        expression
    )
    if calibration is not None:
        assert refreshed.status is CfgStatus.VALID
        accepted = resource.accept(refreshed.ref.revision)
        assert accepted.values["value"] == (12.0 if scaled else 6.0)
        assert accepted.values["neighbor"] == 3.0
    else:
        assert refreshed.status is CfgStatus.INVALID
        with pytest.raises(CfgPreconditionError):
            resource.accept(refreshed.ref.revision)
        reset = resource.reset(refreshed.ref.revision)
        assert reset.status is CfgStatus.VALID
        assert resource.accept(reset.ref.revision).values["value"] == 8.0


def test_unbound_pulse_default_preserves_empty_link_and_refreshes_calibration() -> None:
    environment = _environment(bound=False)
    definition = (
        MeasureCfgBuilder()
        .field(
            "modules.drive",
            spec=make_pulse_module_spec(),
            default=module_seed("qubit.module.x180", fallback=literal(_fallback())),
        )
        .float("neighbor", label="Neighbor", default=3.0)
        .build()
    )
    host = MagicMock(spec=MeasureCfgBindingHost)
    host.list_device_names.return_value = []
    host.arb_waveforms.list_data_keys.return_value = []
    bindings = MeasureCfgBindings(host)
    resource = CfgResource(
        defaults=lambda: definition.instantiate(environment),
        resolution=lambda: bindings.snapshot(
            (), captured_values={}, parameters=environment.parameters
        ),
    )
    initial = resource.observe()
    assert initial.status is CfgStatus.INVALID
    value = read_value_path(resource.snapshot_inputs().value, "modules.drive")
    assert isinstance(value, ReferenceValue)
    assert value.chosen_key == "qubit.module.x180"
    assert value.value == CfgSectionValue()
    assert any(
        diagnostic.path == ("modules", "drive")
        and "qubit.module.x180" in diagnostic.message
        for diagnostic in initial.diagnostics
    )
    with pytest.raises(CfgPreconditionError):
        resource.accept(initial.ref.revision)
    edited = resource.edit(initial.ref.revision, (CfgEdit(("neighbor",), 4.0),))
    assert edited.status is CfgStatus.INVALID
    environment = _environment()
    refreshed = resource.refresh(edited.ref.revision)
    assert refreshed.status is CfgStatus.VALID
    assert refreshed.ref.cfg_id == initial.ref.cfg_id
    assert refreshed.ref.revision > initial.ref.revision
    raw = resource.accept(refreshed.ref.revision).values
    assert raw["neighbor"] == 4.0
    modules = raw["modules"]
    assert isinstance(modules, dict)
    drive = modules["drive"]
    assert isinstance(drive, dict)
    assert drive["gain"] == 0.3
    assert drive["freq"] == 4300.0


def _readout_environment(
    readout: DirectReadoutCfg | PulseReadoutSpec, *, bound: bool
) -> SessionEnv:
    """Capture one readout slot and its concrete library family without I/O."""
    base = _environment()
    assert base.parameters is not None
    point = base.parameters.document()
    owner = point.components["Q1"]
    assert isinstance(owner, ModuleOwner)
    owner.module["readout"] = "Q1.readout.cal"
    return SessionEnv(
        parameters=ParameterContext(
            point=point,
            library=LibrarySnapshot(
                ModuleLibraryDocument(
                    format="zcu.module-library",
                    format_version="1.0",
                    components={"Q1": ComponentLibrary(readout={"cal": readout})},
                )
            ),
            target=EntryDraft(entry_id="entry"),
            roles=RoleBinding(
                "Q1",
                {"readout": RoleSpec("seed/module_owner")},
                {"readout": "Q1"} if bound else {},
            ),
        ),
        soc=None,
        soccfg=None,
    )


@pytest.mark.parametrize("pulse", [False, True], ids=["direct", "pulse"])
def test_unbound_readout_default_takes_catalog_shape_after_binding(pulse: bool) -> None:
    direct = DirectReadoutCfg(ro_ch=0, ro_freq=200.0, ro_length=2.0)
    readout = (
        PulseReadoutSpec(
            ro_cfg=direct,
            pulse_cfg=PulseSpec(
                waveform=ConstWaveformCfg(length=2.0),
                ch=1,
                nqz=2,
                freq=200.0,
                gain=0.3,
            ),
        )
        if pulse
        else direct
    )
    environment = _readout_environment(readout, bound=False)
    definition = (
        MeasureCfgBuilder()
        .field(
            "modules.readout",
            spec=make_readout_module_spec(),
            default=module_seed("readout.module.readout", fallback=None),
        )
        .build()
    )
    host = MagicMock(spec=MeasureCfgBindingHost)
    host.list_device_names.return_value = []
    host.arb_waveforms.list_data_keys.return_value = []
    bindings = MeasureCfgBindings(host)
    resource = CfgResource(
        defaults=lambda: definition.instantiate(environment),
        resolution=lambda: bindings.snapshot(
            (), captured_values={}, parameters=environment.parameters
        ),
    )
    initial = resource.observe()
    assert initial.status is CfgStatus.INVALID
    value = read_value_path(resource.snapshot_inputs().value, "modules.readout")
    assert isinstance(value, ReferenceValue)
    assert value.value == CfgSectionValue()
    assert initial.tree.children["modules"].children["readout"].children == {}
    assert any(
        diagnostic.path == ("modules", "readout")
        and "readout.module.readout" in diagnostic.message
        for diagnostic in initial.diagnostics
    )
    with pytest.raises(CfgPreconditionError):
        resource.accept(initial.ref.revision)

    environment = _readout_environment(readout, bound=True)
    refreshed = resource.refresh(initial.ref.revision)
    assert refreshed.status is CfgStatus.VALID
    raw = resource.accept(refreshed.ref.revision).values["modules"]
    assert isinstance(raw, dict)
    selected = raw["readout"]
    assert isinstance(selected, dict)
    assert selected["type"] == ("readout/pulse" if pulse else "readout/direct")
    if pulse:
        pulse_cfg = selected["pulse_cfg"]
        assert isinstance(pulse_cfg, dict)
        assert pulse_cfg["gain"] == 0.3
    else:
        assert selected["ro_freq"] == 200.0


def test_unbound_optional_module_is_invalid_until_explicitly_disabled() -> None:
    environment = _environment(bound=False)
    definition = (
        MeasureCfgBuilder()
        .field(
            "modules.drive",
            spec=make_pulse_module_spec(optional=True),
            default=module_seed("qubit.module.x180", fallback=None),
        )
        .build()
    )
    host = MagicMock(spec=MeasureCfgBindingHost)
    host.list_device_names.return_value = []
    host.arb_waveforms.list_data_keys.return_value = []
    bindings = MeasureCfgBindings(host)
    resource = CfgResource(
        defaults=lambda: definition.instantiate(environment),
        resolution=lambda: bindings.snapshot(
            (), captured_values={}, parameters=environment.parameters
        ),
    )
    initial = resource.observe()
    assert initial.status is CfgStatus.INVALID
    assert isinstance(
        read_value_path(resource.snapshot_inputs().value, "modules.drive"),
        ReferenceValue,
    )
    disabled = resource.edit(
        initial.ref.revision, (CfgEdit(("modules", "drive", "__ref"), None),)
    )
    assert disabled.status is CfgStatus.VALID
    assert read_value_path(resource.snapshot_inputs().value, "modules.drive") is None
    environment = _environment()
    refreshed = resource.refresh(disabled.ref.revision)
    assert refreshed.status is CfgStatus.VALID
    assert read_value_path(resource.snapshot_inputs().value, "modules.drive") is None
    assert resource.accept(refreshed.ref.revision).values == {"modules": {}}


@pytest.mark.parametrize("slot", [None, "Q1.pulses.missing"], ids=["unset", "dangling"])
def test_unbound_module_keeps_missing_link_until_reset_uses_bound_fallback(
    slot: str | None,
) -> None:
    environment = _environment(bound=False)
    definition = (
        MeasureCfgBuilder()
        .field(
            "modules.drive",
            spec=make_pulse_module_spec(),
            default=module_seed("qubit.module.x180", fallback=literal(_fallback())),
        )
        .build()
    )
    host = MagicMock(spec=MeasureCfgBindingHost)
    host.list_device_names.return_value = []
    host.arb_waveforms.list_data_keys.return_value = []
    bindings = MeasureCfgBindings(host)
    resource = CfgResource(
        defaults=lambda: definition.instantiate(environment),
        resolution=lambda: bindings.snapshot(
            (), captured_values={}, parameters=environment.parameters
        ),
    )
    initial = resource.observe()
    assert initial.status is CfgStatus.INVALID
    environment = _environment(slot=slot)
    refreshed = resource.refresh(initial.ref.revision)
    assert refreshed.status is CfgStatus.INVALID
    value = read_value_path(resource.snapshot_inputs().value, "modules.drive")
    assert isinstance(value, ReferenceValue)
    assert value.chosen_key == "qubit.module.x180"
    assert value.value == CfgSectionValue()
    with pytest.raises(CfgPreconditionError):
        resource.accept(refreshed.ref.revision)
    reset = resource.reset(refreshed.ref.revision)
    assert reset.status is CfgStatus.VALID
    modules = resource.accept(reset.ref.revision).values["modules"]
    assert isinstance(modules, dict)
    drive = modules["drive"]
    assert isinstance(drive, dict)
    assert drive["gain"] == 0.1


def _fallback() -> ReferenceValue:
    value = make_default_value(make_pulse_module_spec().allowed[0])
    value.with_field("gain", 0.1)
    return ReferenceValue(make_custom_reference_key("Pulse"), value)


@pytest.mark.parametrize("path", ["qubit.module.x180", "Q1.pulses.cal"])
def test_module_seed_mounts_fixed_capture_and_lowers_calibration(path: str) -> None:
    environment = _environment()
    definition = (
        MeasureCfgBuilder()
        .field(
            "modules.drive",
            spec=make_pulse_module_spec(),
            default=module_seed(path, fallback=literal(_fallback())),
        )
        .build()
    )
    schema = definition.instantiate(environment)
    reference = read_value_path(schema.value, "modules.drive")
    assert isinstance(reference, ReferenceValue)
    assert reference.chosen_key == path
    lowered = schema_to_raw_dict(schema, environment.parameters)
    modules = lowered["modules"]
    assert isinstance(modules, dict)
    drive = modules["drive"]
    assert isinstance(drive, dict)
    assert drive["gain"] == 0.3
    assert drive["freq"] == 4300.0
    assert drive["ch"] == 1
    assert drive["waveform"] == {"style": "const", "length": 2.0}


@pytest.mark.parametrize("slot", [None, "Q1.pulses.missing"])
def test_missing_slot_uses_explicit_fallback_or_disables(slot: str | None) -> None:
    environment = _environment(slot=slot)
    result = module_seed("qubit.module.x180", fallback=literal(_fallback())).resolve(
        environment
    )
    assert isinstance(result, ReferenceValue)
    assert read_value_path(result.value, "gain") == DirectValue(0.1)
    assert module_seed("qubit.module.x180", fallback=None).resolve(environment) is None


def test_missing_direct_root_and_context_use_explicit_fallback() -> None:
    for environment in [
        _environment(),
        SessionEnv(parameters=None, soc=None, soccfg=None),
    ]:
        result = module_seed(
            "Q1.pulses.missing", fallback=literal(_fallback())
        ).resolve(environment)
        assert isinstance(result, ReferenceValue)
        assert read_value_path(result.value, "gain") == DirectValue(0.1)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("", "module seed path must not be empty"),
        ("Q1.waveforms.missing", "does not match partition"),
        ("qubit.bad.bad.bad", "Expected a library root"),
    ],
)
def test_invalid_module_path_does_not_become_fallback(path: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        module_seed(path, fallback=literal(_fallback())).resolve(_environment())


def test_unavailable_capture_does_not_become_fallback(tmp_path: Path) -> None:
    environment = _environment(
        faults=(
            ContextSourceFault(
                "point", tmp_path / "point.yaml", "ValueError", "broken"
            ),
        )
    )
    with pytest.raises(ContextSourceUnavailableError):
        module_seed("qubit.module.x180", fallback=literal(_fallback())).resolve(
            environment
        )


@pytest.mark.parametrize("slot", [None, "Q1.pulses.missing"])
def test_scaled_module_calibration_absence_uses_explicit_fallback(
    slot: str | None,
) -> None:
    seed = scaled_path_seed(
        "qubit.module.x180.waveform.length", factor=4.0, fallback_value=0.3
    )
    assert seed.resolve(_environment(slot=slot)) == 0.3


def test_scaled_module_calibration_preserves_capture_expression() -> None:
    value = scaled_path_seed(
        "qubit.module.x180.waveform.length", factor=4.0, fallback_value=0.3
    ).resolve(_environment())
    assert isinstance(value, EvalValue)
    assert value.expr == "4.0 * qubit.module.x180.waveform.length"


def test_scaled_module_calibration_does_not_hide_materialization_errors() -> None:
    seed = scaled_path_seed(
        "qubit.module.x180.waveform.length", factor=4.0, fallback_value=0.3
    )
    with pytest.raises((ValueError, KeyError)):
        seed.resolve(_environment(broken_waveform=True))


def test_scaled_module_calibration_does_not_hide_source_faults(tmp_path: Path) -> None:
    seed = scaled_path_seed(
        "qubit.module.x180.waveform.length", factor=4.0, fallback_value=0.3
    )
    with pytest.raises(ContextSourceUnavailableError):
        seed.resolve(
            _environment(
                faults=(
                    ContextSourceFault(
                        "point", tmp_path / "point.yaml", "ValueError", "broken"
                    ),
                )
            )
        )


def test_nested_materialization_failure_does_not_become_fallback() -> None:
    with pytest.raises((ValueError, KeyError)):
        module_seed("qubit.module.x180", fallback=literal(_fallback())).resolve(
            _environment(broken_waveform=True)
        )
