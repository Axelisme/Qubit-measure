"""Registry contract for role-aware adapter defaults and captured lowering."""

from collections.abc import Generator
from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    is_custom_reference_key,
    read_value_path,
)
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    EntryDraft,
)
from zcu_tools.gui.session.types import ParameterContext, SessionEnv
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import RoleResolutionError, RoleSpec, component_registry
from zcu_tools.resources.entry.schema import PointDocument

from zcu_lab.components import QubitSchema, ResonatorSchema
from zcu_lab.definitions import register_all


def _registry() -> Registry:
    registry = Registry()
    register_all(registry)
    return registry


def _role_adapters() -> list[str]:
    registry = _registry()
    return [
        name
        for name in registry.list_names()
        if set(registry.create(name).parameter_roles()) == {"qubit", "resonator"}
    ]


@pytest.fixture(autouse=True)
def component_kinds() -> Generator[None]:
    before = component_registry.kinds()
    added: list[str] = []
    for kind, model in [
        ("qubit/transmon", QubitSchema),
        ("resonator", ResonatorSchema),
    ]:
        if kind not in before:
            component_registry.register(kind, model)
            added.append(kind)
    existing_roles: dict[str, RoleSpec] = {}
    added_roles: list[str] = []
    for role, spec in [
        ("qubit", RoleSpec("qubit/*")),
        ("resonator", RoleSpec("resonator")),
    ]:
        try:
            existing_roles[role] = component_registry.roles.get(role)
        except ValueError:
            component_registry.roles.register(role, spec)
            added_roles.append(role)
    try:
        yield
    finally:
        for role in reversed(added_roles):
            component_registry.roles.unregister(role)
        for role, spec in existing_roles.items():
            assert component_registry.roles.get(role) == spec
        for kind in reversed(added):
            component_registry.unregister(kind)
        assert component_registry.kinds() == before


def _source(*, calibrated: bool, null_calibrations: bool = False) -> ParameterContext:
    components: dict[str, object] = {}
    libraries: dict[str, object] = {}
    for index in [1, 2]:
        qubit, resonator = f"Q{index}", f"R{index}"
        pulse = {
            "waveform": {"style": "const", "length": 0.05 * index},
            "ch": index,
            "nqz": 2,
            "freq": 4200.0 + 200 * index,
            "gain": 0.2 * index,
        }
        components[qubit] = {
            "kind": "qubit/transmon",
            "resonator": resonator,
            "freq": None if null_calibrations else pulse["freq"],
            "kappa": 10.0 * index,
            "t1": None if null_calibrations else 12.0 * index,
            "wiring": {"L01": {"ch": index}},
            "ext": {"best_ro_freq": 6000.0 + 100 * index},
            "module": (
                {
                    "x180": f"{qubit}.pulses.cal",
                    "x90": f"{qubit}.pulses.half",
                    "readout": f"{resonator}.readout.cal",
                }
                if calibrated
                else {}
            ),
        }
        components[resonator] = {
            "kind": "resonator",
            "freq": None if null_calibrations else 6000.0 + 100 * index,
            "kappa": 20.0 * index,
            "wiring": {"ch": index, "ro_ch": index},
            "ext": {
                "default_probe_length": None if null_calibrations else 2.0 * index,
                "time_of_fly": None if null_calibrations else 0.2 * index,
            },
        }
        libraries[qubit] = {
            "pulses": {"cal": pulse, "half": {**pulse, "gain": 0.1 * index}}
        }
        libraries[resonator] = {
            "readout": {
                "cal": {
                    "type": "readout/pulse",
                    "pulse_cfg": {**pulse, "freq": 6000.0 + 100 * index},
                    "ro_cfg": {
                        "ro_ch": index,
                        "ro_length": 1.0,
                        "trig_offset": 0.5,
                    },
                }
            }
        }
    return ParameterContext(
        point=PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-08T00:00:00Z"},
                "components": components,
            }
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument.model_validate(
                {
                    "format": "zcu.module-library",
                    "format_version": "1.0",
                    "components": libraries,
                }
            )
        ),
        target=EntryDraft(entry_id="contract-entry"),
        roles=None,
    )


@pytest.mark.parametrize("name", _role_adapters())
@pytest.mark.parametrize("calibrated", [False, True], ids=["inline", "library"])
def test_registered_role_adapter_lowers_one_template_on_fixed_captures(
    name: str, calibrated: bool
) -> None:
    adapter = _registry().create(name)
    assert isinstance(adapter, BaseAdapter)
    source = _source(calibrated=calibrated)
    first = source.bind_roles(adapter.parameter_roles(), focus="Q1", explicit={})
    second = source.bind_roles(adapter.parameter_roles(), focus="Q2", explicit={})
    definition = type(adapter).cfg_definition()
    schema = definition.instantiate(SessionEnv(parameters=first, soc=None, soccfg=None))
    second_schema = definition.instantiate(
        SessionEnv(parameters=second, soc=None, soccfg=None)
    )
    raw_first = schema_to_raw_dict(schema, first)
    raw_second = schema_to_raw_dict(second_schema, second)
    assert definition.spec == schema.spec == second_schema.spec
    module_spec = definition.spec.fields["modules"]
    assert isinstance(module_spec, CfgSectionSpec)
    pulses: list[tuple[str, ReferenceSpec]] = []
    for key, spec in module_spec.fields.items():
        if isinstance(spec, ReferenceSpec):
            discriminator = spec.allowed[0].fields.get("type")
            if (
                isinstance(discriminator, LiteralSpec)
                and discriminator.value == "pulse"
            ):
                pulses.append((key, spec))
    for capture, cfg, raw in [
        (first, schema, raw_first),
        (second, second_schema, raw_second),
    ]:
        readout = read_value_path(cfg.value, "modules.readout")
        assert isinstance(readout, ReferenceValue)
        references = [readout]
        modules = raw["modules"]
        assert isinstance(modules, dict)
        for pulse_key, pulse_spec in pulses:
            pulse = read_value_path(cfg.value, f"modules.{pulse_key}")
            if pulse is None:
                assert pulse_spec.optional
            else:
                assert isinstance(pulse, ReferenceValue)
                references.append(pulse)
                lowered_pulse = modules[pulse_key]
                assert isinstance(lowered_pulse, dict)
                assert (
                    lowered_pulse["ch"] == capture.lookup("qubit.wiring.L01.ch").value
                )
        for ref in references:
            if not is_custom_reference_key(ref.chosen_key):
                assert ref.chosen_key.startswith("qubit.module.") and (
                    capture.library_key(ref.chosen_key)
                    == capture.lookup(ref.chosen_key).value
                )
        lowered_readout = modules["readout"]
        assert isinstance(lowered_readout, dict)
        readout_pulse, readout_ro = (
            lowered_readout["pulse_cfg"],
            lowered_readout["ro_cfg"],
        )
        assert isinstance(readout_pulse, dict) and isinstance(readout_ro, dict)
        assert readout_pulse["ch"] == capture.lookup("resonator.wiring.ch").value
        assert readout_ro["ro_ch"] == capture.lookup("resonator.wiring.ro_ch").value
    # Later target selection does not alter the original source or the template.
    source.bind_roles(adapter.parameter_roles(), focus="Q2", explicit={})
    assert schema_to_raw_dict(schema, first) == raw_first
    assert (
        definition.instantiate(SessionEnv(parameters=first, soc=None, soccfg=None))
        == schema
    )


@pytest.mark.parametrize("name", _role_adapters())
@pytest.mark.parametrize("null_calibrations", [False, True])
def test_registered_readout_frequency_sweep_uses_captured_probe_timing(
    name: str, null_calibrations: bool
) -> None:
    adapter = _registry().create(name)
    assert isinstance(adapter, BaseAdapter)
    modules_spec = type(adapter).cfg_definition().spec.fields["modules"]
    assert isinstance(modules_spec, CfgSectionSpec)
    for spec in modules_spec.fields.values():
        if isinstance(spec, ReferenceSpec):
            discriminator = spec.allowed[0].fields.get("type")
            if (
                isinstance(discriminator, LiteralSpec)
                and discriminator.value == "pulse"
            ):
                pytest.skip("adapter also calibrates a qubit-drive pulse")
    readout_spec = modules_spec.fields["readout"]
    assert isinstance(readout_spec, ReferenceSpec)
    pulse_spec = readout_spec.allowed[0].fields.get("pulse_cfg")
    if not isinstance(pulse_spec, CfgSectionSpec) or not isinstance(
        pulse_spec.fields["freq"], LiteralSpec
    ):
        pytest.skip("adapter does not own an inline readout-frequency sweep")
    capture = _source(calibrated=True, null_calibrations=null_calibrations).bind_roles(
        adapter.parameter_roles(), focus="Q2", explicit={}
    )
    schema = adapter.make_default_cfg(
        SessionEnv(parameters=capture, soc=None, soccfg=None)
    )
    raw = schema_to_raw_dict(schema, capture)
    modules = raw["modules"]
    assert isinstance(modules, dict)
    readout = modules["readout"]
    assert isinstance(readout, dict)
    pulse, acquisition = readout["pulse_cfg"], readout["ro_cfg"]
    assert isinstance(pulse, dict) and isinstance(acquisition, dict)
    assert pulse["freq"] == acquisition["ro_freq"] == 0.0
    assert acquisition["ro_length"] == pytest.approx(0.9 if null_calibrations else 3.9)
    assert acquisition["trig_offset"] == pytest.approx(
        0.55 if null_calibrations else 0.45
    )


@pytest.mark.parametrize("name", _role_adapters())
def test_registered_role_adapter_accepts_uncalibrated_null_scalar_values(
    name: str,
) -> None:
    adapter = _registry().create(name)
    capture = _source(calibrated=False, null_calibrations=True).bind_roles(
        adapter.parameter_roles(), focus="Q2", explicit={}
    )
    schema = adapter.make_default_cfg(
        SessionEnv(parameters=capture, soc=None, soccfg=None)
    )
    raw = schema_to_raw_dict(schema, capture)
    assert isinstance(raw["relax_delay"], (int, float))
    assert isinstance(raw["modules"], dict)


@pytest.mark.parametrize("name", _role_adapters())
def test_registered_role_adapter_does_not_hide_missing_roles_or_source_faults(
    name: str,
) -> None:
    adapter = _registry().create(name)
    source = _source(calibrated=True)
    with pytest.raises(RoleResolutionError):
        source.bind_roles(adapter.parameter_roles(), focus=None, explicit={})
    capture = source.bind_roles(adapter.parameter_roles(), focus="Q1", explicit={})
    unavailable = ParameterContext(
        point=capture.document(),
        library=capture.library,
        target=capture.target,
        roles=capture.roles,
        source_faults=(
            ContextSourceFault(
                document="point",
                path=Path("/contract/point.yaml"),
                error_type="ValueError",
                message="broken source",
            ),
        ),
    )
    with pytest.raises(ContextSourceUnavailableError):
        adapter.make_default_cfg(
            SessionEnv(parameters=unavailable, soc=None, soccfg=None)
        )
