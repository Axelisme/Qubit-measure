"""Context-free measure cfg definition, seed, and module policy contracts."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from zcu_tools.cfg_model import CfgInputMap, module_spec_from_raw
from zcu_tools.gui.app.measure import seeds
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.role_factories import RoleFactorySpec
from zcu_tools.gui.app.measure.schema_builder import MeasureCfgBuilder, ModuleInit
from zcu_tools.gui.app.measure.seeds import (
    SweepDefault,
    custom,
    literal,
    module_seed,
    path_seed,
    scaled_path_seed,
    value_source,
)
from zcu_tools.gui.app.measure.spec_helpers import (
    make_bath_reset_module_spec,
    make_pulse_module_spec,
    make_pulse_readout_module_spec,
    make_pulse_reset_module_spec,
)
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceValue,
    SweepValue,
    make_custom_reference_key,
    make_default_value,
    read_value_path,
)
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.value_lookup import (
    EmptyValueLookup,
    MissingValue,
    ValueKey,
    ValueRegistry,
)
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import ComponentSchema

from tests.gui.app.measure._cfg_fakes import parameter_capture


class QubitSchema(ComponentSchema):
    t1: float | None = None
    module: dict[str, str]


class ResonatorSchema(ComponentSchema):
    ext: dict[str, float | None]


def _pulse_shape() -> CfgSectionSpec:
    return make_pulse_module_spec().allowed[0]


def _readout_shape() -> CfgSectionSpec:
    return make_pulse_readout_module_spec().allowed[0]


def _reset_shape() -> CfgSectionSpec:
    return make_pulse_reset_module_spec().allowed[0]


def _bath_shape() -> CfgSectionSpec:
    return make_bath_reset_module_spec().allowed[0]


def _blank(shape: CfgSectionSpec) -> ReferenceValue:
    return ReferenceValue(
        make_custom_reference_key(shape.label), make_default_value(shape)
    )


def _readout_ref(ctx: SessionEnv, *, optional: bool = False) -> ReferenceValue | None:
    return module_seed(
        "$qubit.module.readout",
        fallback=None if optional else literal(_blank(_readout_shape())),
    ).resolve(ctx)


ROLE_FACTORIES = {
    "pi_pulse": RoleFactorySpec(
        "module", _pulse_shape, lambda _ctx: _blank(_pulse_shape())
    ),
    "readout": RoleFactorySpec(
        "module", _readout_shape, lambda _ctx: _blank(_readout_shape()), _readout_ref
    ),
    "reset": RoleFactorySpec(
        "module",
        _reset_shape,
        lambda _ctx: _blank(_reset_shape()),
        lambda _ctx, **_kwargs: None,
    ),
    "bath_reset": RoleFactorySpec(
        "module", _bath_shape, lambda _ctx: _blank(_bath_shape())
    ),
}


def _ctx(
    *,
    md_values: dict[str, float] | None = None,
    ml: LibrarySnapshot | None = None,
    values: ValueRegistry | None = None,
) -> SessionEnv:
    point_values = md_values or {}
    readout = "R1.readout.readout_rf" if ml is not None else None
    parameters = parameter_capture(
        {
            "Q1": QubitSchema(
                kind="test/builder-qubit",
                t1=point_values.get("t1"),
                module={} if readout is None else {"readout": readout},
            ),
            "R1": ResonatorSchema(
                kind="test/builder-resonator",
                ext={"default_probe_length": point_values.get("res_probe_len")},
            ),
        },
        ml,
    ).bind_aliases((AliasRow("qubit", "Q1"), AliasRow("resonator", "R1")), {})
    return SessionEnv(
        parameters=parameters,
        soc=None,
        soccfg=None,
        values=values if values is not None else EmptyValueLookup(),
    )


def test_alias_name_default_is_direct_and_fixed_per_instance() -> None:
    definition = (
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
        .device(
            "flux_dev", label="Flux Device", default=seeds.alias_name_seed("flux_dev")
        )
        .build()
    )
    parameters = parameter_capture()
    first_ctx = SessionEnv(
        parameters=parameters.bind_aliases((AliasRow("flux_dev", "yoko1"),), {}),
        soc=None,
        soccfg=None,
    )
    first = definition.instantiate(first_ctx)
    second_ctx = SessionEnv(
        parameters=parameters.bind_aliases((AliasRow("flux_dev", "yoko2"),), {}),
        soc=None,
        soccfg=None,
    )
    second = definition.instantiate(second_ctx)
    assert read_value_path(first.value, "dev.flux_dev") == DirectValue("yoko1")
    assert read_value_path(second.value, "dev.flux_dev") == DirectValue("yoko2")


def test_unbound_alias_name_default_retains_direct_invalid_input() -> None:
    definition = (
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
        .device(
            "flux_dev", label="Flux Device", default=seeds.alias_name_seed("flux_dev")
        )
        .build()
    )
    ctx = SessionEnv(
        parameters=parameter_capture().bind_aliases((AliasRow("flux_dev", ""),), {}),
        soc=None,
        soccfg=None,
    )
    schema = definition.instantiate(ctx)
    value = read_value_path(schema.value, "dev.flux_dev")
    assert isinstance(value, DirectValue)
    assert value.value is None
    assert value.raw == "$flux_dev"
    assert value.error is not None and "flux_dev" in value.error


def _readout_raw(*, gain: float = 0.2) -> CfgInputMap:
    return {
        "type": "readout/pulse",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": 1.0},
            "ch": 1,
            "nqz": 2,
            "freq": 6100.0,
            "gain": gain,
        },
        "ro_cfg": {
            "ro_ch": 2,
            "ro_freq": 6100.0,
            "ro_length": 1.0,
            "trig_offset": 0.5,
        },
    }


def _readout_gain(node: ReferenceValue) -> DirectValue:
    pulse = node.value.fields["pulse_cfg"]
    assert isinstance(pulse, CfgSectionValue)
    gain = pulse.fields["gain"]
    assert isinstance(gain, DirectValue)
    return gain


def test_definition_shape_is_context_free_and_instances_are_isolated() -> None:
    definition = (
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
        .pulse("pi_pulse", role_id="pi_pulse")
        .relax_delay(scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=100.0))
        .sweep(
            "length",
            label="Delay (us)",
            default=SweepDefault(
                start=0.0,
                stop=scaled_path_seed("$qubit.t1", factor=5.0, fallback_value=500.0),
                expts=101,
            ),
        )
        .reps(1000)
        .rounds(100)
        .build()
    )
    static_spec = definition.spec

    empty = definition.instantiate(_ctx())
    calibrated = definition.instantiate(_ctx(md_values={"t1": 12.0}))

    assert empty.spec == static_spec == calibrated.spec
    assert empty.value.fields["relax_delay"] == EvalValue("5.0 * $qubit.t1")
    assert calibrated.value.fields["relax_delay"] == EvalValue("5.0 * $qubit.t1")
    sweep = calibrated.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    length = sweep.fields["length"]
    assert isinstance(length, SweepValue)
    assert length.stop == EvalValue("5.0 * $qubit.t1")

    empty.spec.fields.clear()
    assert definition.spec == static_spec


def test_literal_sweep_is_snapshotted_at_authoring_and_per_instance() -> None:
    original = SweepValue(0.0, 1.0, expts=11)
    definition = (
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
        .sweep("freq", label="Freq", default=original)
        .reps(1)
        .rounds(1)
        .build()
    )

    original.stop = 9.0
    original.expts = 2
    first = definition.instantiate(_ctx())
    second = definition.instantiate(_ctx())
    first_section = first.value.fields["sweep"]
    second_section = second.value.fields["sweep"]
    assert isinstance(first_section, CfgSectionValue)
    assert isinstance(second_section, CfgSectionValue)
    first_sweep = first_section.fields["freq"]
    second_sweep = second_section.fields["freq"]
    assert isinstance(first_sweep, SweepValue)
    assert isinstance(second_sweep, SweepValue)
    assert first_sweep.stop == second_sweep.stop == 1.0
    assert first_sweep.expts == second_sweep.expts == 11

    first_sweep.stop = 7.0
    assert second_sweep.stop == 1.0


def test_module_init_modes_are_literal_and_smart_optional_means_adopt_or_none() -> None:
    ml = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
    ).replaced_module("R1.readout.readout_rf", module_spec_from_raw(_readout_raw()))
    definition = (
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
        .readout("smart", role_id="readout")
        .readout("inline", role_id="readout", init=ModuleInit.INLINE)
        .readout(
            "disabled",
            role_id="readout",
            optional=True,
            init=ModuleInit.DISABLED,
        )
        .reset("reset", optional=True)
        .reps(1)
        .rounds(1)
        .build()
    )

    schema = definition.instantiate(_ctx(ml=ml))
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    smart = modules.fields["smart"]
    inline = modules.fields["inline"]
    assert (
        isinstance(smart, ReferenceValue)
        and smart.chosen_key == "$qubit.module.readout"
    )
    assert isinstance(inline, ReferenceValue)
    assert inline.chosen_key == "<Custom:Pulse Readout>"
    assert modules.fields["disabled"] is None
    assert modules.fields["reset"] is None


@pytest.mark.parametrize(
    ("build_bad", "path", "role_id", "expected", "actual"),
    [
        (
            lambda: MeasureCfgBuilder(role_factories=ROLE_FACTORIES).pulse(
                "bad", role_id="readout"
            ),
            "modules.bad",
            "readout",
            "type='pulse'",
            "type='readout/pulse'",
        ),
        (
            lambda: MeasureCfgBuilder(role_factories=ROLE_FACTORIES).reset(
                "tested_reset",
                shape="pulse",
                role_id="bath_reset",
            ),
            "modules.tested_reset",
            "bath_reset",
            "type='reset/pulse'",
            "type='reset/bath'",
        ),
    ],
)
def test_module_role_shape_mismatch_fails_at_authoring(
    build_bad: Callable[[], object],
    path: str,
    role_id: str,
    expected: str,
    actual: str,
) -> None:
    with pytest.raises(TypeError) as exc_info:
        build_bad()

    message = str(exc_info.value)
    assert path in message
    assert role_id in message
    assert expected in message
    assert actual in message


def _exception_notes(exc: BaseException) -> str:
    return "\n".join(getattr(exc, "__notes__", ()))


def test_custom_seed_failure_identifies_cfg_path_and_seed() -> None:
    def fail(_ctx: SessionEnv) -> float:
        raise ValueError("broken custom resolver")

    definition = (
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
        .float(
            "probe.limit",
            label="Limit",
            default=custom(fail, description="probe limit policy"),
        )
        .build()
    )

    with pytest.raises(ValueError, match="broken custom resolver") as exc_info:
        definition.instantiate(_ctx())

    notes = _exception_notes(exc_info.value)
    assert "probe.limit" in notes
    assert "probe limit policy" in notes


def test_missing_value_source_identifies_cfg_path_and_seed() -> None:
    definition = (
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
        .device(
            "flux_dev",
            label="Flux Device",
            default=value_source(
                "device.flux.name",
                target_type=str,
            ),
        )
        .build()
    )

    with pytest.raises(MissingValue) as exc_info:
        definition.instantiate(_ctx())

    notes = _exception_notes(exc_info.value)
    assert "dev.flux_dev" in notes
    assert "value source:device.flux.name" in notes


def test_module_override_failure_identifies_module_path_and_seed() -> None:
    def fail(_ctx: SessionEnv) -> float:
        raise RuntimeError("broken module override")

    definition = (
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
        .readout(
            overrides={
                "ro_cfg.ro_length": custom(
                    fail,
                    description="readout length policy",
                )
            }
        )
        .build()
    )

    with pytest.raises(RuntimeError, match="broken module override") as exc_info:
        definition.instantiate(_ctx())

    notes = _exception_notes(exc_info.value)
    assert "modules.readout" in notes
    assert "module role:readout" in notes
    assert "modules.readout.ro_cfg.ro_length" in notes
    assert "readout length policy" in notes


def test_blank_overrides_do_not_pollute_library_but_overrides_always_apply() -> None:
    ml = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
    ).replaced_module(
        "R1.readout.readout_rf", module_spec_from_raw(_readout_raw(gain=0.2))
    )
    definition = (
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
        .readout(
            blank_overrides={"pulse_cfg.gain": 0.05},
            overrides={"ro_cfg.ro_length": 1.5},
        )
        .reps(1)
        .rounds(1)
        .build()
    )

    linked = definition.instantiate(_ctx(ml=ml))
    blank = definition.instantiate(_ctx())

    linked_modules = linked.value.fields["modules"]
    blank_modules = blank.value.fields["modules"]
    assert isinstance(linked_modules, CfgSectionValue)
    assert isinstance(blank_modules, CfgSectionValue)
    linked_readout = linked_modules.fields["readout"]
    blank_readout = blank_modules.fields["readout"]
    assert isinstance(linked_readout, ReferenceValue)
    assert isinstance(blank_readout, ReferenceValue)
    assert _readout_gain(linked_readout) == DirectValue(0.2)
    assert _readout_gain(blank_readout) == DirectValue(0.05)
    for readout in (linked_readout, blank_readout):
        ro = readout.value.fields["ro_cfg"]
        assert isinstance(ro, CfgSectionValue)
        assert ro.fields["ro_length"] == DirectValue(1.5)


def test_md_override_and_locks_are_local_to_one_module_declaration() -> None:
    definition = (
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
        .readout(
            pulse_only=True,
            init=ModuleInit.INLINE,
            locked={"pulse_cfg.freq": 0.0, "ro_cfg.ro_freq": 0.0},
            overrides={
                "ro_cfg.ro_length": path_seed(
                    "$resonator.ext.default_probe_length",
                    expr="$resonator.ext.default_probe_length - 0.1",
                    fallback=0.9,
                )
            },
        )
        .reps(1)
        .rounds(1)
        .build()
    )

    schema = definition.instantiate(_ctx(md_values={"res_probe_len": 1.2}))
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    readout = modules.fields["readout"]
    assert isinstance(readout, ReferenceValue)
    pulse = readout.value.fields["pulse_cfg"]
    ro = readout.value.fields["ro_cfg"]
    assert isinstance(pulse, CfgSectionValue)
    assert isinstance(ro, CfgSectionValue)
    assert pulse.fields["freq"] == DirectValue(0.0)
    assert ro.fields["ro_freq"] == DirectValue(0.0)
    assert ro.fields["ro_length"] == EvalValue(
        "$resonator.ext.default_probe_length - 0.1"
    )

    with pytest.raises(ValueError, match="cannot also be overridden"):
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES).readout(
            locked={"pulse_cfg.freq": 0.0},
            overrides={"pulse_cfg.freq": 1.0},
        )


def test_value_source_seed_resolves_once_per_fresh_instance() -> None:
    calls = 0
    registry = ValueRegistry()

    def read_device() -> str:
        nonlocal calls
        calls += 1
        return "flux"

    registry.register(ValueKey("device.flux.name", str), read_device, owner="test")
    ctx = _ctx(values=registry)
    definition = (
        MeasureCfgBuilder(role_factories=ROLE_FACTORIES)
        .device_from_value_source(
            "flux_dev",
            label="Flux Device",
            source_key="device.flux.name",
            fallback="flux_yoko",
        )
        .reps(1)
        .rounds(1)
        .build()
    )

    schema = definition.instantiate(ctx)
    assert calls == 1
    dev = schema.value.fields["dev"]
    assert isinstance(dev, CfgSectionValue)
    assert dev.fields["flux_dev"] == DirectValue("flux")
    assert calls == 1
