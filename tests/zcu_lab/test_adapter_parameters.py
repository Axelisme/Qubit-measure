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
                {"x180": f"{qubit}.pulses.cal", "readout": f"{resonator}.readout.cal"}
                if calibrated
                else {}
            ),
        }
        components[resonator] = {
            "kind": "resonator",
            "freq": None if null_calibrations else 6000.0 + 100 * index,
            "kappa": 20.0 * index,
            "wiring": {"ch": index, "ro_ch": index},
        }
        libraries[qubit] = {"pulses": {"cal": pulse}}
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
    assert len(pulses) == 1
    pulse_key, pulse_spec = pulses[0]
    inline_probe = isinstance(pulse_spec.allowed[0].fields["freq"], LiteralSpec)
    for capture, cfg, raw in [
        (first, schema, raw_first),
        (second, second_schema, raw_second),
    ]:
        pulse = read_value_path(cfg.value, f"modules.{pulse_key}")
        readout = read_value_path(cfg.value, "modules.readout")
        assert isinstance(pulse, ReferenceValue)
        assert isinstance(readout, ReferenceValue)
        assert is_custom_reference_key(pulse.chosen_key) == (
            not calibrated or inline_probe
        )
        assert is_custom_reference_key(readout.chosen_key) == (not calibrated)
        for ref, slot in [(pulse, "x180"), (readout, "readout")]:
            if not is_custom_reference_key(ref.chosen_key):
                assert ref.chosen_key == f"qubit.module.{slot}"
                assert (
                    capture.library_key(ref.chosen_key)
                    == capture.lookup(f"qubit.module.{slot}").value
                )
        modules = raw["modules"]
        assert isinstance(modules, dict)
        lowered_pulse, lowered_readout = modules[pulse_key], modules["readout"]
        assert isinstance(lowered_pulse, dict)
        assert isinstance(lowered_readout, dict)
        readout_pulse, readout_ro = (
            lowered_readout["pulse_cfg"],
            lowered_readout["ro_cfg"],
        )
        assert isinstance(readout_pulse, dict) and isinstance(readout_ro, dict)
        assert lowered_pulse["ch"] == capture.lookup("qubit.wiring.L01.ch").value
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
