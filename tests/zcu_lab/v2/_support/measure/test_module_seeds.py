"""Captured module defaults through definition instantiation and lowering."""

from collections.abc import Generator
from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import (
    DirectValue,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
    read_value_path,
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
from zcu_lab.v2._support.measure.seeds import literal, module_seed
from zcu_lab.v2._support.measure.spec_helpers import make_pulse_module_spec


class ModuleOwner(ComponentSchema):
    module: dict[str, str]


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
            "Q1", {"qubit": RoleSpec("seed/module_owner")}, {"qubit": "Q1"}
        ),
        source_faults=faults,
    )
    return SessionEnv(parameters=parameters, soc=None, soccfg=None)


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


def test_nested_materialization_failure_does_not_become_fallback() -> None:
    with pytest.raises((ValueError, KeyError)):
        module_seed("qubit.module.x180", fallback=literal(_fallback())).resolve(
            _environment(broken_waveform=True)
        )
