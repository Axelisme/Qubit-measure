"""Persistent module Specs generate editable forms that convert back to Specs."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import JsonValue, create_model
from zcu_tools.cfg_model import (
    BathResetSpec,
    ConfigBase,
    ConstWaveformCfg,
    Expr,
    FlatTopWaveformSpec,
    ModuleSpec,
    MuxReadoutSpec,
    PulseReadoutSpec,
    PulseResetSpec,
    PulseSpec,
    Slot,
    Template,
    TemplateDef,
    TwoPulseResetSpec,
    WaveformSpec,
    get_field_meta,
)
from zcu_tools.cfg_model import PulseCfg as MaterializedPulseCfg
from zcu_tools.gui import form_adapter
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    FormWritePath,
)
from zcu_tools.gui.form_adapter import ModuleInline
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    SavedPoint,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.program.v2.modules import PulseCfg as NativePulseCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

UNSUPPORTED_PULSE_FIELDS = ("desc",)

PULSE_LEAVES: dict[FormWritePath, JsonValue] = {
    ("ch",): 2,
    ("nqz",): 1,
    ("freq",): 100.0,
    ("gain",): 0.25,
}


class SyntheticCfg(ConfigBase):
    """An ordinary cfg is not a persistent module or waveform Spec."""

    value: float = 1.0


def result_spec[S: ConfigBase](
    form: form_adapter.Form[S], state: FormState[FormModel], env: FormEnv
) -> S:
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    return cfg


@pytest.mark.parametrize(
    "spec_type",
    [ModuleSpec, WaveformSpec, MaterializedPulseCfg, NativePulseCfg, SyntheticCfg],
    ids=["module-union", "waveform-union", "materialized", "native", "synthetic"],
)
def test_module_form_rejects_non_persistent_spec_roots(
    spec_type: type[ConfigBase],
) -> None:
    with pytest.raises(TypeError):
        form_adapter.module_form(spec_type)


def waveform_env(length: float = 0.1) -> FormEnv:
    """A captured waveform is detached from the source library document."""
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        waveforms={"flat": ConstWaveformCfg(length=length)}
                    )
                },
            )
        )
    )


def expect_object(value: JsonValue) -> dict[str, JsonValue]:
    """Read a schema or input object through the public JSON boundary."""
    assert isinstance(value, dict)
    return value


def test_captured_waveform_key_is_a_unique_valid_option_and_stays_a_key() -> None:
    form = form_adapter.module_form(PulseSpec)
    env = waveform_env()
    spec = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.25, waveform="Q.waveforms.flat")
    state = FormState(form.input_from_spec(spec, env))
    current = state.input.data["waveform"]

    options = form.model.options(env).get(("waveform",), ())
    matching = [option for option in options if option.matches(current)]
    assert [option.label for option in matching] == ["Q.waveforms.flat"]
    assert matching[0].value == current == {"kind": "ref", "name": "Q.waveforms.flat"}
    target = state.describe(("waveform",), env).target
    assert target is not None and target.options is not None
    assert target.selected is not None
    assert target.options[target.selected].value == current
    assert result_spec(form, state, env).waveform == "Q.waveforms.flat"
    assert state.input.data["waveform"] == current


def test_waveform_options_keep_all_inline_kinds_with_branch_fields() -> None:
    form = form_adapter.module_form(PulseSpec)
    env = waveform_env()
    options = form.model.options(env).get(("waveform",), ())
    assert [option.label for option in options] == [
        "Q.waveforms.flat",
        "Inline: const",
        "Inline: cosine",
        "Inline: gauss",
        "Inline: drag",
        "Inline: arb",
        "Inline: flat_top",
    ]
    state = FormState(form.default_input(env))
    for option, names in (
        (options[1], {"length"}),
        (options[-1], {"length", "raise_waveform"}),
    ):
        state.select(("waveform",), option.value, env)
        children = state.describe(("waveform",), env).children
        assert {field.path[-1] for field in children} == names


def test_waveform_key_command_copies_once_and_disconnects_from_later_sources() -> None:
    form = form_adapter.module_form(PulseSpec)
    env = waveform_env()
    spec = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.25, waveform="Q.waveforms.flat")
    state = FormState(form.input_from_spec(spec, env))
    key = next(
        option
        for option in form.model.options(env)[("waveform",)]
        if option.matches(state.input.data["waveform"])
    )
    assert [(command.name, command.label) for command in key.commands] == [
        ("to_inline", "Convert to inline")
    ]
    notifications: list[str] = []
    state.subscribe(lambda: notifications.append("changed"))

    state.run_command(("waveform",), "to_inline", env)

    assert state.input.data["waveform"] == {
        "kind": "inline",
        "module": {"style": "const", "length": 0.1},
    }
    assert notifications == ["changed"]
    assert env.library is not None
    assert env.library.waveform_spec("Q.waveforms.flat").model_dump() == {
        "style": "const",
        "length": 0.1,
    }
    cfg = result_spec(form, state, waveform_env(0.9))
    assert isinstance(cfg.waveform, ConstWaveformCfg) and cfg.waveform.length == 0.1
    with pytest.raises(ValueError, match="Unavailable field command"):
        state.run_command(("waveform",), "to_inline", env)
    assert notifications == ["changed"]


class UnreadableWaveformLibrary(LibrarySnapshot):
    """A public source fault leaves commands without a usable captured Spec."""

    def waveform_spec(self, path: str) -> WaveformSpec:
        raise OSError(f"Cannot read captured waveform {path}")


def test_waveform_command_source_failure_keeps_input_and_emits_nothing() -> None:
    form = form_adapter.module_form(PulseSpec)
    env = waveform_env()
    spec = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.25, waveform="Q.waveforms.flat")
    state = FormState(form.input_from_spec(spec, env))
    key = next(
        option
        for option in form.model.options(env)[("waveform",)]
        if option.matches(state.input.data["waveform"])
    )
    assert [command.name for command in key.commands] == ["to_inline"]
    assert env.library is not None
    broken = FormEnv(library=UnreadableWaveformLibrary(env.library.document()))
    notifications: list[str] = []
    state.subscribe(lambda: notifications.append("changed"))
    before = state.input.data

    with pytest.raises(OSError, match="Cannot read captured waveform"):
        state.run_command(("waveform",), "to_inline", broken)
    with pytest.raises(OSError, match="Cannot read captured waveform"):
        state.run_command(("waveform",), "to_inline", broken)

    assert state.input.data == before
    assert notifications == []


@pytest.mark.parametrize(
    ("spec_type", "raw", "field", "expected"),
    [
        (
            PulseSpec,
            {
                "ch": 0,
                "nqz": 1,
                "freq": 100.0,
                "gain": 0.25,
                "waveform": "$unknown.waveform",
            },
            "waveform",
            {"kind": "ref", "name": "$unknown.waveform"},
        ),
        (
            PulseResetSpec,
            {"pulse_cfg": "unknown.pulse"},
            "pulse_cfg",
            {"kind": "ref", "name": "unknown.pulse"},
        ),
        (
            TwoPulseResetSpec,
            {"pulse1_cfg": "one", "pulse2_cfg": "two"},
            "pulse1_cfg",
            {"kind": "ref", "name": "one"},
        ),
        (
            TwoPulseResetSpec,
            {"pulse1_cfg": "one", "pulse2_cfg": "two"},
            "pulse2_cfg",
            {"kind": "ref", "name": "two"},
        ),
        (
            BathResetSpec,
            {"cavity_tone_cfg": "cavity", "qubit_tone_cfg": "qubit", "pi2_cfg": "pi2"},
            "cavity_tone_cfg",
            {"kind": "ref", "name": "cavity"},
        ),
        (
            BathResetSpec,
            {"cavity_tone_cfg": "cavity", "qubit_tone_cfg": "qubit", "pi2_cfg": "pi2"},
            "qubit_tone_cfg",
            {"kind": "ref", "name": "qubit"},
        ),
        (
            BathResetSpec,
            {"cavity_tone_cfg": "cavity", "qubit_tone_cfg": "qubit", "pi2_cfg": "pi2"},
            "pi2_cfg",
            {"kind": "ref", "name": "pi2"},
        ),
        (
            FlatTopWaveformSpec,
            {"length": 1.0, "raise_waveform": "ramp"},
            "raise_waveform",
            {"kind": "ref", "name": "ramp"},
        ),
        (
            PulseSpec,
            {
                "ch": 0,
                "nqz": 1,
                "freq": 100.0,
                "gain": 0.25,
                "waveform": {
                    "style": "flat_top",
                    "length": 1.0,
                    "raise_waveform": "ramp",
                },
            },
            "waveform",
            {
                "kind": "inline",
                "module": {
                    "style": "flat_top",
                    "length": 1.0,
                    "raise_waveform": {"kind": "ref", "name": "ramp"},
                },
            },
        ),
        (
            PulseResetSpec,
            {
                "pulse_cfg": {
                    "ch": 0,
                    "nqz": 1,
                    "freq": 100.0,
                    "gain": 0.25,
                    "waveform": "flat",
                }
            },
            "pulse_cfg",
            None,
        ),
    ],
    ids=[
        "waveform-key",
        "pulse-key",
        "pulse1-key",
        "pulse2-key",
        "cavity-key",
        "qubit-key",
        "pi2-key",
        "ramp-key",
        "nested-ramp",
        "nested-pulse",
    ],
)
def test_tagged_spec_projection_round_trips_without_sources(
    spec_type: type[ConfigBase],
    raw: dict[str, JsonValue],
    field: str,
    expected: dict[str, JsonValue] | None,
) -> None:
    spec = spec_type.model_validate(raw)
    form = form_adapter.module_form(spec_type)
    env = FormEnv()
    state = FormState(form.input_from_spec(spec, env))
    child = state.input.data[field]
    if expected is None:
        child = expect_object(child)
        assert child.get("kind") == "inline"
        assert expect_object(child["module"])["waveform"] == {
            "kind": "ref",
            "name": "flat",
        }
    else:
        assert child == expected
    assert result_spec(form, state, env).model_dump() == spec.model_dump()


def test_tagged_waveform_name_is_key_only_in_description_result_defaults_and_restore() -> (
    None
):
    form = form_adapter.module_form(PulseSpec)
    env = FormEnv()
    spec = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.25, waveform="$missing.value")
    state = FormState(form.input_from_spec(spec, env))
    assert state.input.data["waveform"] == {"kind": "ref", "name": "$missing.value"}
    assert result_spec(form, state, env).waveform == "$missing.value"
    state.write({("waveform",): {"kind": "ref", "name": {"expr": "1 / 0"}}}, env)
    assert expect_object(state.input.data["waveform"])["name"] == {"expr": "1 / 0"}
    errors = state.result(env)
    assert isinstance(errors, FormErrors)
    assert errors.messages == {("waveform", "name"): "Input should be a valid string"}
    restored = FormState(form.input_from_spec(spec, env))
    assert (
        restored.restore_input(FormInput(restored.input.model, state.input.data), env)
        == ()
    )
    assert restored.input.data == state.input.data
    assert isinstance(restored.result(env), FormErrors)


@pytest.mark.parametrize("missing", ["gen_ch", "ro_ch"])
def test_flat_readout_requires_explicit_channels(missing: str) -> None:
    form = form_adapter.module_form(PulseReadoutSpec)
    env = FormEnv()
    data = form.input_from_spec(writeback_proposal(), env).data
    data.pop(missing)
    state = FormState(FormInput(form.model, data))

    errors = state.result(env)

    assert isinstance(errors, FormErrors)
    assert errors.messages == {(missing,): "Field required"}


def test_flat_readout_edits_generator_and_adc_without_expanding_waveform_key() -> None:
    form = form_adapter.module_form(PulseReadoutSpec)
    env = FormEnv()
    source = writeback_proposal()
    state = FormState(form.input_from_spec(source, env))
    state.write(
        {
            ("gen_ch",): 2,
            ("ro_ch",): 4,
            ("freq",): {"expr": "500 + 25"},
            ("gain",): 0.3,
            ("ro_length",): 1.5,
            ("trig_offset",): 0.2,
            ("waveform",): {"kind": "ref", "name": "Q.waveforms.ramp"},
        },
        env,
    )

    cfg = result_spec(form, state, env)

    assert (cfg.gen_ch, cfg.ro_ch, cfg.freq, cfg.gain) == (2, 4, 525.0, 0.3)
    assert (cfg.ro_length, cfg.trig_offset) == (1.5, 0.2)
    assert cfg.waveform == "Q.waveforms.ramp"
    assert cfg.desc is None
    assert source == writeback_proposal()


def test_spec_root_tag_is_hidden_and_remains_in_input() -> None:
    form = form_adapter.module_form(PulseSpec)
    env = FormEnv()
    tag = FormState(form.default_input(env)).describe(("type",), env).target
    assert tag is not None and tag.hidden and tag.fixed_value == "pulse"
    assert form.default_input(env).data["type"] == "pulse"


def test_spec_form_evaluates_input_and_keeps_child_library_key() -> None:
    form = form_adapter.module_form(PulseSpec)
    env = FormEnv()
    state = FormState(form.default_input(env))
    batch: dict[FormWritePath, JsonValue] = {
        **PULSE_LEAVES,
        ("gain",): {"expr": "0.5 / 2"},
        ("waveform",): {"kind": "ref", "name": "Q.flat"},
    }
    state.write(batch, env)

    cfg = result_spec(form, state, env)

    assert type(cfg) is PulseSpec
    assert cfg.gain == 0.25
    assert cfg.waveform == "Q.flat"


@pytest.mark.parametrize("name", UNSUPPORTED_PULSE_FIELDS)
def test_unsupported_pulse_fields_cannot_be_written_and_use_spec_defaults(
    name: str,
) -> None:
    form = form_adapter.module_form(PulseSpec)
    env = FormEnv()
    state = FormState(form.default_input(env))

    with pytest.raises(ValueError, match="Unknown.*field path"):
        state.write({(name,): 1}, env)
    batch: dict[FormWritePath, JsonValue] = {
        **PULSE_LEAVES,
        ("waveform",): {"kind": "ref", "name": "Q.flat"},
    }
    state.write(batch, env)
    cfg = result_spec(form, state, env)

    assert getattr(cfg, name) == PulseSpec.model_fields[name].default


def library_entry() -> PulseSpec:
    """A stored entry sets unsupported fields that the Form cannot edit."""
    return PulseSpec.model_validate(
        {
            "ch": 2,
            "nqz": 1,
            "freq": 100.0,
            "gain": 0.25,
            "desc": "stored entry",
            "waveform": "Q.flat",
        }
    )


def writeback_proposal() -> PulseReadoutSpec:
    """A proposal keeps its waveform key and optional description."""
    return PulseReadoutSpec(
        desc="proposal",
        gen_ch=0,
        nqz=2,
        freq=6000.0,
        gain=0.1,
        waveform="Q.square",
        ro_ch=3,
        ro_length=1.0,
    )


@pytest.mark.parametrize(
    ("spec", "unsupported"),
    [
        (library_entry(), {("desc",)}),
        (
            writeback_proposal(),
            {("desc",)},
        ),
    ],
    ids=["library-entry", "writeback-proposal"],
)
def test_existing_spec_round_trips_except_unsupported_values(
    spec: PulseSpec | PulseReadoutSpec, unsupported: set[tuple[str, ...]]
) -> None:
    form = form_adapter.module_form(type(spec))
    env = FormEnv()
    original = spec.model_dump(mode="json")

    state = FormState(form.input_from_spec(spec, env))
    cfg = result_spec(form, state, env)

    assert spec.model_dump(mode="json") == original
    assert not any(path[-1] == "mixer_freq" for path in leaves(state.input.data))
    changed = {
        path
        for path in leaves(original) | leaves(cfg.model_dump(mode="json"))
        if lookup(original, path) != lookup(cfg.model_dump(mode="json"), path)
    }
    assert changed == unsupported


def leaves(data: Mapping[str, JsonValue]) -> set[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = set()
    for name, value in data.items():
        if isinstance(value, dict):
            paths |= {(name, *child) for child in leaves(value)}
        else:
            paths.add((name,))
    return paths


def lookup(data: JsonValue, path: tuple[str, ...]) -> JsonValue:
    for part in path:
        data = data.get(part) if isinstance(data, dict) else None
    return data


class DerivedPulse(PulseSpec):
    """A subclass instance is not the Form's concrete persistent Spec."""


def test_input_from_spec_rejects_another_spec_type_or_core_form() -> None:
    env = FormEnv()
    derived = DerivedPulse.model_validate(library_entry().model_dump(mode="json"))

    with pytest.raises(TypeError):
        form_adapter.module_form(PulseSpec).input_from_spec(derived, env)
    with pytest.raises(TypeError):
        form_adapter.Form(SyntheticCfg, {}).input_from_spec(SyntheticCfg(), env)


class HostInline(ModuleInline):
    """A host offers its own channel choices for the nested generated Spec."""

    @classmethod
    def choices(
        cls, env: FormEnv
    ) -> Mapping[tuple[str | int, ...], tuple[JsonValue, ...]]:
        del env
        return {("module", "ch"): (2, 3)}


def test_generated_model_nests_in_module_inline_with_host_choices() -> None:
    form = form_adapter.module_form(PulseSpec)
    host = create_model("PulseInline", __base__=HostInline, module=(form.model, ...))
    env = FormEnv()
    state = FormState(host.default_input(env))
    batch: dict[FormWritePath, JsonValue] = {
        ("module", *path): value for path, value in PULSE_LEAVES.items()
    }
    batch |= {("module", "ch"): 7, ("module", "waveform"): {"kind": "ref", "name": "w"}}

    state.write(batch, env)
    rejected = state.result(env)
    state.write({("module", "ch"): 3}, env)
    accepted = state.result(env)

    assert isinstance(rejected, FormErrors)
    assert set(rejected.messages) == {("module", "ch")}
    assert isinstance(accepted, HostInline) and isinstance(accepted.module, FormModel)
    cfg = form.to_cfg(accepted.module, env)
    assert isinstance(cfg, PulseSpec) and cfg.ch == 3 and cfg.waveform == "w"


@pytest.mark.parametrize("spec_type", [PulseSpec, PulseReadoutSpec])
def test_spec_field_meta_reaches_generated_description(
    spec_type: type[PulseSpec] | type[PulseReadoutSpec],
) -> None:
    form = form_adapter.module_form(spec_type)
    env = FormEnv()
    fields = {
        field.path[-1]: field
        for field in FormState(form.default_input(env)).describe((), env).children
    }
    observed: dict[str, tuple[JsonValue, ...]] = {}
    expected: dict[str, tuple[JsonValue, ...]] = {}
    for name, info in spec_type.model_fields.items():
        meta = get_field_meta(info)
        field = fields.get(name)
        if meta is None or field is None:
            continue
        expected[name] = (meta.unit, meta.decimals, meta.group)
        observed[name] = (field.unit, field.decimals, field.group)
    assert expected and observed == expected


@pytest.fixture
def creation_templates() -> dict[str, TemplateDef]:
    """Synthetic creation values cover module and nested waveform roots."""
    return {
        "pulse": TemplateDef(
            PulseSpec,
            "Synthetic pulse",
            {
                "ch": 2,
                "nqz": 1,
                "freq": Expr("general.ext.seed"),
                "gain": 0.25,
                "waveform.style": "const",
                "waveform.length": Expr("2 * .2"),
            },
        ),
        "flat": TemplateDef(
            FlatTopWaveformSpec,
            "Synthetic flat top",
            {
                "length": Expr("2 * .2"),
                "raise_waveform.style": "cosine",
                "raise_waveform.length": 0.1,
            },
        ),
        "reset": TemplateDef(
            PulseResetSpec,
            "Synthetic reset",
            {"pulse_cfg": Slot("Q.pulses.probe")},
        ),
    }


def creation_env(
    seed: float = 120.0, *, faults: tuple[ContextSourceFault, ...] = ()
) -> FormEnv:
    """Capture detached synthetic point values and one persistent pulse key."""
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="2.0",
            components={"Q": ComponentLibrary(pulses={"probe": library_entry()})},
        )
    )
    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="2.0",
            general=PointGeneral(created_at="2026-10-13T00:00:00Z", ext={"seed": seed}),
        ),
        library=library,
        target=SavedPoint(entry_id="synthetic", point="synthetic"),
        roles=None,
        source_faults=faults,
    )
    return FormEnv(parameters=parameters, library=library)


@pytest.mark.parametrize(
    ("spec_type", "template_id", "expected"),
    [
        (
            PulseSpec,
            "pulse",
            {
                "type": "pulse",
                "ch": 2,
                "nqz": 1,
                "freq": 120.0,
                "gain": 0.25,
                "phase": 0.0,
                "pre_delay": 0.0,
                "post_delay": 0.0,
                "waveform": {"style": "const", "length": 0.4},
            },
        ),
        (
            FlatTopWaveformSpec,
            "flat",
            {
                "style": "flat_top",
                "length": 0.4,
                "raise_waveform": {"style": "cosine", "length": 0.1},
            },
        ),
    ],
    ids=["module", "waveform"],
)
def test_template_input_is_fresh_and_round_trips_to_persistent_spec(
    creation_templates: dict[str, TemplateDef],
    spec_type: type[PulseSpec] | type[FlatTopWaveformSpec],
    template_id: str,
    expected: dict[str, JsonValue],
) -> None:
    form = form_adapter.module_form(spec_type)
    env = creation_env()
    seeds_before = dict(creation_templates[template_id].seeds)
    point_before = env.parameters.document() if env.parameters is not None else None
    library_before = library_entry().model_dump()
    first = form.template_input(Template(template_id), creation_templates, env)
    second = form.template_input(Template(template_id), creation_templates, env)
    state = FormState(first)

    cfg = result_spec(form, state, env)
    assert type(cfg) is spec_type
    assert cfg.model_dump(mode="json", exclude_none=True) == expected
    assert first is not second and first.data == second.data
    state.write({("length" if template_id == "flat" else "gain",): 0.8}, env)
    assert second.data == first.data
    assert dict(creation_templates[template_id].seeds) == seeds_before
    assert env.parameters is not None and env.parameters.document() == point_before
    assert env.library is not None
    assert env.library.module_spec("Q.pulses.probe").model_dump() == library_before


@pytest.mark.parametrize("expression", ["1 / 0", "'not a number'", "$missing.value"])
def test_template_input_failed_expr_requires_input_instead_of_static_default(
    creation_templates: dict[str, TemplateDef], expression: str
) -> None:
    form = form_adapter.module_form(PulseSpec)
    env = creation_env()
    template = Template("pulse", set={"phase": Expr(expression)})
    input = form.template_input(template, creation_templates, env)
    state = FormState(input)

    assert "phase" not in input.data
    errors = state.result(env)
    assert isinstance(errors, FormErrors)
    assert errors.messages == {("phase",): "Field required"}
    assert "phase" not in input.model.default_input(env).data
    state.write({("phase",): 15.0}, env)
    assert result_spec(form, state, env).phase == 15.0
    assert form.model.default_input(env).data["phase"] == 0.0


@pytest.mark.parametrize(
    ("spec_type", "template_id", "child", "authored_kind", "other_kind", "length"),
    [
        (PulseSpec, "pulse", "waveform", "const", "cosine", 0.4),
        (FlatTopWaveformSpec, "flat", "raise_waveform", "cosine", "gauss", 0.1),
    ],
    ids=["pulse", "flat-top"],
)
def test_template_input_nested_kind_reselection_uses_authored_defaults(
    creation_templates: dict[str, TemplateDef],
    spec_type: type[PulseSpec] | type[FlatTopWaveformSpec],
    template_id: str,
    child: str,
    authored_kind: str,
    other_kind: str,
    length: float,
) -> None:
    form = form_adapter.module_form(spec_type)
    env = creation_env()
    state = FormState(
        form.template_input(Template(template_id), creation_templates, env)
    )
    target = state.describe((child,), env).target
    assert target is not None and target.options is not None
    options = target.options

    def choose(kind: str) -> None:
        value = next(item.value for item in options if item.label == f"Inline: {kind}")
        state.select((child,), value, env)

    assert (
        expect_object(expect_object(state.input.data[child])["module"])["length"]
        == length
    )
    state.write({(child, "module", "length"): 1.7}, env)
    choose(other_kind)
    assert expect_object(state.input.data[child]) == {
        "kind": "inline",
        "module": {"style": other_kind, "length": 1.7},
    }
    choose(authored_kind)
    assert (
        expect_object(expect_object(state.input.data[child])["module"]).get("length")
        == 1.7
    )


def test_template_input_models_isolate_locks_defaults_and_recomputation(
    creation_templates: dict[str, TemplateDef],
) -> None:
    form = form_adapter.module_form(PulseSpec)
    original_model = form.model
    env = creation_env()
    first_template = Template("pulse", lock={"gain": 0.6})
    second_template = Template("pulse", set={"phase": 7.0}, lock={"ch": 3})
    first = form.template_input(first_template, creation_templates, env)
    second = form.template_input(second_template, creation_templates, env)
    first_state = FormState(first)
    second_state = FormState(second)

    assert issubclass(first.model, original_model)
    assert issubclass(second.model, original_model)
    assert form.model is original_model
    first_state.write({("gain",): 0.7}, env)
    errors = first_state.result(env)
    assert isinstance(errors, FormErrors) and set(errors.messages) == {("gain",)}
    first_state.write({("gain",): 0.6}, env)
    first_cfg = result_spec(form, first_state, env)
    second_cfg = result_spec(form, second_state, env)
    assert type(first_cfg) is PulseSpec and first_cfg.gain == 0.6
    assert type(second_cfg) is PulseSpec and second_cfg.gain == 0.25
    assert second_cfg.phase == 7.0 and second_cfg.ch == 3
    second_state.write({("gain",): 0.4, ("ch",): 2}, env)
    errors = second_state.result(env)
    assert isinstance(errors, FormErrors) and set(errors.messages) == {("ch",)}
    refreshed = first.model.default_input(creation_env(125.0))
    assert refreshed.data["freq"] == 125.0 and refreshed.data["gain"] == 0.6
    assert first.data["freq"] == 120.0
    assert form.default_input(env).data == original_model.default_input(env).data
    assert dict(first_template.lock) == {"gain": 0.6}
    assert dict(second_template.set) == {"phase": 7.0}


def test_template_input_slot_keeps_bound_key(
    creation_templates: dict[str, TemplateDef],
) -> None:
    form = form_adapter.module_form(PulseResetSpec)
    env = creation_env()
    input = form.template_input(Template("reset"), creation_templates, env)
    cfg = result_spec(form, FormState(input), env)
    assert input.data["pulse_cfg"] == {"kind": "ref", "name": "Q.pulses.probe"}
    assert cfg.pulse_cfg == "Q.pulses.probe"


@pytest.mark.parametrize(
    ("template", "message"),
    [
        (Template("unknown"), "unknown"),
        (Template("flat"), "flat.*incompatible"),
        (Template("pulse", set={"desc": "unsupported"}), "pulse.*desc"),
        (Template("pulse", set={"waveform.bad": 1}), "pulse.*waveform.bad"),
    ],
    ids=["unknown-id", "wrong-family", "hidden-field", "unknown-path"],
)
def test_template_input_rejects_invalid_templates_at_public_entry(
    creation_templates: dict[str, TemplateDef], template: Template, message: str
) -> None:
    form = form_adapter.module_form(PulseSpec)
    with pytest.raises(ValueError, match=message):
        form.template_input(template, creation_templates, creation_env())


def test_template_input_rejects_core_cfg_form(
    creation_templates: dict[str, TemplateDef],
) -> None:
    form = form_adapter.Form(SyntheticCfg, {})
    with pytest.raises(TypeError, match="persistent Spec"):
        form.template_input(Template("pulse"), creation_templates, creation_env())


def test_template_input_propagates_unavailable_captured_source(
    creation_templates: dict[str, TemplateDef],
) -> None:
    fault = ContextSourceFault(
        "point", Path("/unavailable/point.yaml"), "OSError", "unreadable"
    )
    env = creation_env(faults=(fault,))
    form = form_adapter.module_form(PulseSpec)
    with pytest.raises(ContextSourceUnavailableError):
        form.template_input(Template("pulse"), creation_templates, env)


def test_mux_module_form_edits_tone_labels_as_a_whole_json_leaf() -> None:
    form = form_adapter.module_form(MuxReadoutSpec)
    env = FormEnv()
    source = MuxReadoutSpec.model_validate(
        {
            "gen_ch": 9,
            "nqz": 2,
            "mixer_freq": 6000.0,
            "length": 2.0,
            "ro_length": 1.5,
            "tones": [
                {"freq": 6020.0, "gain": 0.1, "ro_ch": 7},
                {"freq": 5980.0, "gain": 0.2, "ro_ch": 4},
            ],
        }
    )
    state = FormState(form.input_from_spec(source, env))
    tones: list[JsonValue] = [
        {"freq": 6020.0, "gain": 0.1, "ro_ch": 7, "label": "Upper"},
        {"freq": 5980.0, "gain": 0.2, "ro_ch": 4, "label": "Lower"},
    ]
    state.write({("tones",): tones}, env)
    spec = result_spec(form, state, env)
    assert [tone.label for tone in spec.tones] == ["Upper", "Lower"]
    assert [tone.ro_ch for tone in spec.tones] == [7, 4]
    restored = FormState(form.input_from_spec(spec, env))
    assert result_spec(form, restored, env) == spec
    assert [tone.label for tone in source.tones] == ["", ""]
