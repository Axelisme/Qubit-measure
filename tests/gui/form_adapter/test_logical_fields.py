"""Logical edits and captured Spec observations through the public Form seam."""

from collections.abc import Mapping
from typing import Self

import pytest
from pydantic import Field, JsonValue
from zcu_tools.cfg_model import (
    ArbWaveformCfg,
    ConfigBase,
    ConstWaveformCfg,
    PulseSpec,
    Template,
    TemplateDef,
)
from zcu_tools.gui import form_adapter as adapter
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPathError,
    FormState,
    FormValue,
)
from zcu_tools.program.v2.modules import PulseCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class RealLeaf(ConfigBase):
    value: int = 2


class RealSection(ConfigBase):
    module: RealLeaf


class LogicalCfg(ConfigBase):
    drive: PulseCfg | None = None
    real: RealSection
    reps: int = Field(default=1, gt=0)
    samples: list[int] = Field(default_factory=list)


def captured() -> FormEnv:
    pulse = PulseSpec(
        ch=1, nqz=1, freq=4100, gain=0.3, waveform="Q.waveforms.wave", desc="stored"
    )
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        pulses={"drive": pulse},
                        waveforms={"wave": ConstWaveformCfg(length=0.08)},
                    )
                },
            )
        )
    )


def object_at(
    data: Mapping[str, JsonValue], path: tuple[str, ...]
) -> dict[str, JsonValue]:
    """Narrow a public authored input subtree for its behavioral assertions."""
    value: JsonValue = dict(data)
    for name in path:
        assert isinstance(value, dict), path
        value = value[name]
    assert isinstance(value, dict), path
    return value


def ordinary_object(value: FormValue) -> dict[str, FormValue]:
    """Narrow an observed ordinary subtree, not a product implementation detail."""
    assert isinstance(value, dict)
    return value


def keyed(form: adapter.Form[LogicalCfg], env: FormEnv) -> FormInput[FormModel]:
    state = FormState(form.default_input(env))
    state.write({("drive",): {"kind": "ref", "name": "Q.pulses.drive"}}, env)
    return state.input


def test_ref_read_projects_nested_capture_without_authored_input_or_mutation() -> None:
    form = adapter.Form(LogicalCfg, {})
    env = captured()
    input = keyed(form, env)
    before = input.to_json()
    library = env.library.document().model_dump() if env.library is not None else None
    observed = form.read_logical_field(input, ("drive", "waveform", "length"), env)
    assert observed.value == 0.08 and observed.value_status == "available"
    assert observed.readonly and not observed.input_present and observed.input is None
    assert observed.selector is None and observed.errors is None
    assert observed.source == adapter.LogicalSource(
        ("drive", "waveform"), "Q.waveforms.wave", "waveform", ("length",)
    )
    assert observed.description.path == ("drive", "waveform", "length")
    assert (
        observed.description.target is not None
        and not observed.description.target.fixed
    )
    assert input.to_json() == before
    assert env.library is not None and env.library.document().model_dump() == library
    with pytest.raises(adapter.LogicalEditError) as failure:
        form.prepare_logical_edits(
            input, [adapter.LogicalEdit(("drive", "freq"), 5)], env
        )
    assert (
        failure.value.reason_code == "ref_ancestor"
        and failure.value.selector_path == ("drive",)
    )


def test_copy_is_explicit_preserves_child_key_and_batch_keeps_expressions() -> None:
    form = adapter.Form(LogicalCfg, {})
    env = captured()
    input = keyed(form, env)
    candidate = form.prepare_inline_copy(input, ("drive",), env)
    assert candidate.changed and input.data["drive"] == {
        "kind": "ref",
        "name": "Q.pulses.drive",
    }
    assert object_at(candidate.input.data, ("drive", "module"))["waveform"] == {
        "kind": "ref",
        "name": "Q.waveforms.wave",
    }
    assert not form.prepare_inline_copy(candidate.input, ("drive",), env).changed
    child = form.prepare_inline_copy(candidate.input, ("drive", "waveform"), env)
    edits = [
        adapter.LogicalEdit(("drive", "waveform", "length"), {"expr": "1 / 4"}),
        adapter.LogicalEdit(("real", "module", "value"), 9),
    ]
    result = form.prepare_logical_edits(child.input, edits, env)
    assert object_at(result.input.data, ("drive", "module", "waveform", "module"))[
        "length"
    ] == {"expr": "1 / 4"}
    leaf = form.read_logical_field(result.input, ("drive", "waveform", "length"), env)
    assert leaf.input_present and leaf.input == {"expr": "1 / 4"} and leaf.value == 0.25
    assert not leaf.readonly and leaf.source is None
    assert (
        form.read_logical_field(result.input, ("real", "module", "value"), env).value
        == 9
    )
    assert isinstance(
        FormState(
            form.prepare_logical_edits(
                result.input, [adapter.LogicalEdit(("reps",), -1)], env
            ).input
        ).result(env),
        FormErrors,
    )


@pytest.mark.parametrize(
    "paths,code,index",
    [
        ([("reps",), ("drive", "freq")], "ref_ancestor", 1),
        ([("reps",), ("unknown",)], "form_path", 1),
        ([("reps",), ("reps",)], "form_structure", 1),
        ([("drive",)], "module_selector", 0),
        ([("samples", "0")], "form_path", 0),
    ],
)
def test_batch_rejection_has_original_index_and_no_success_prefix(
    paths, code, index
) -> None:
    form = adapter.Form(LogicalCfg, {})
    env = captured()
    input = keyed(form, env)
    before = input.to_json()
    with pytest.raises(adapter.LogicalEditError) as failure:
        form.prepare_logical_edits(
            input, [adapter.LogicalEdit(path, 4) for path in paths], env
        )
    assert failure.value.reason_code == code and failure.value.edit_index == index
    assert input.to_json() == before


def test_selector_metadata_selection_disable_and_unavailable_source_are_distinct() -> (
    None
):
    form = adapter.Form(LogicalCfg, {})
    env = captured()
    input = keyed(form, env)
    observed = form.read_logical_field(input, ("drive",), env)
    assert observed.selector == adapter.LogicalSelector(
        "key", "Q.pulses.drive", ("Q.pulses.drive",)
    )
    assert not observed.readonly and observed.source == adapter.LogicalSource(
        ("drive",), "Q.pulses.drive", "module", ()
    )
    assert "freq" in [field.name for field in observed.description.children]
    assert all(
        field.path == ("drive", field.name) for field in observed.description.children
    )
    assert ordinary_object(observed.value)["waveform"] == "Q.waveforms.wave"
    unavailable = form.read_logical_field(input, ("drive",), FormEnv())
    assert (
        unavailable.selector is not None
        and unavailable.selector.mode == "key"
        and unavailable.value_status == "unavailable"
    )
    assert (
        unavailable.source is None
        and unavailable.errors is not None
        and not unavailable.description.children
    )
    with pytest.raises(adapter.LogicalSourceError) as failure:
        form.read_logical_field(input, ("drive", "freq"), FormEnv())
    assert failure.value.path == ("drive", "freq") and failure.value.selector_path == (
        "drive",
    )
    with pytest.raises(adapter.LogicalEditError) as rejected:
        form.prepare_library_selection(input, ("drive",), "Q.waveforms.wave", env)
    assert rejected.value.reason_code == "library_key"
    disabled = form.prepare_module_disable(input, ("drive",), env)
    assert disabled.input.data["drive"] is None
    with pytest.raises(FormPathError):
        form.read_logical_field(disabled.input, ("drive", "freq"), env)


def test_spec_template_locks_and_handwritten_generated_fields_use_same_mapper() -> None:
    field = adapter.ModuleInputField(PulseSpec)
    from pydantic import create_model

    extras = create_model(
        "Extras", __base__=FormModel, reference=(field.annotation, ...)
    )
    form = adapter.Form(LogicalCfg, {}, extras=extras, lower=lambda extras, cfg: cfg)
    env = captured()
    input = FormInput(
        form.model, {"extras": {"reference": {"kind": "ref", "name": "Q.pulses.drive"}}}
    )
    assert (
        form.read_logical_field(input, ("extras", "reference", "freq"), env).value
        == 4100
    )
    copied = form.prepare_inline_copy(input, ("extras", "reference"), env)
    assert object_at(copied.input.data, ("extras", "reference"))["kind"] == "inline"
    spec = adapter.module_form(PulseSpec)
    template = spec.template_input(
        Template("pulse", lock={"freq": 42.0}),
        {
            "pulse": TemplateDef(
                PulseSpec,
                "Synthetic pulse",
                {
                    "ch": 1,
                    "nqz": 1,
                    "freq": 10.0,
                    "gain": 0.3,
                    "waveform.style": "const",
                    "waveform.length": 0.1,
                },
            )
        },
        env,
    )
    locked = spec.read_logical_field(template, ("freq",), env)
    assert (
        locked.description.target is not None
        and locked.description.target.fixed
        and locked.value == 42.0
    )
    assert (
        spec.prepare_logical_edits(
            template, [adapter.LogicalEdit(("gain",), 0.5)], env
        ).input.model
        is template.model
    )


def test_logical_errors_hide_only_generated_wrappers_and_preserve_indexes() -> None:
    form = adapter.Form(LogicalCfg, {})
    env = captured()
    input = form.prepare_inline_copy(keyed(form, env), ("drive",), env).input
    errors = FormErrors(
        {
            ("drive", "module", "freq"): "frequency",
            ("drive", "module", "waveform", "name"): "key",
            ("real", "module", "value"): "real",
            ("samples", 2): "item",
        }
    )
    result = form.logical_errors(input, errors, env)
    assert result is not None and result.messages == {
        ("drive", "freq"): "frequency",
        ("drive", "waveform"): "key",
        ("real", "module", "value"): "real",
        ("samples", 2): "item",
    }


def test_root_observation_contains_logical_subtrees_and_scalar_failure_is_unavailable() -> (
    None
):
    form = adapter.Form(LogicalCfg, {})
    env = captured()
    input = form.prepare_inline_copy(keyed(form, env), ("drive",), env).input
    root = form.read_logical_field(input, (), env)
    assert isinstance(root.value, dict)
    drive = root.value["drive"]
    assert isinstance(drive, dict) and drive["freq"] == 4100
    assert drive["waveform"] == "Q.waveforms.wave"
    invalid = FormState(input)
    invalid.write({("reps",): -1}, env)
    unrelated = form.read_logical_field(invalid.input, ("drive", "freq"), env)
    assert unrelated.value == 4100 and unrelated.errors is None
    leaf = form.read_logical_field(invalid.input, ("reps",), env)
    assert (
        leaf.value_status == "unavailable"
        and leaf.value is None
        and leaf.errors is not None
    )


def test_parent_model_errors_are_relevant_but_unrelated_sibling_errors_are_not() -> (
    None
):
    from pydantic import model_validator

    class RootError(FormModel):
        first: int = 1
        second: int = 2

        @model_validator(mode="after")
        def reject_pair(self) -> Self:
            if self.first == self.second:
                raise ValueError("Pair must differ")
            return self

    form = adapter.Form(LogicalCfg, {}, extras=RootError, lower=lambda extras, cfg: cfg)
    input = FormInput(form.model, {"extras": {"first": 2, "second": 2}})
    observation = form.read_logical_field(input, ("extras", "first"), FormEnv())
    assert observation.errors is not None and ("extras",) in observation.errors.messages


def test_ref_source_with_missing_child_key_does_not_block_scalar_or_fallback() -> None:
    form = adapter.Form(LogicalCfg, {})
    env = captured()
    assert env.library is not None
    missing = env.library.replaced_waveform(
        "Q.waveforms.wave", ArbWaveformCfg(data="not-an-asset")
    )
    env = FormEnv(library=missing)
    input = keyed(form, env)
    scalar = form.read_logical_field(input, ("drive", "freq"), env)
    assert scalar.value == 4100 and scalar.errors is None
    empty = FormEnv(library=missing.deleted("Q.waveforms.wave", kind="waveform"))
    assert form.read_logical_field(input, ("drive", "freq"), empty).value == 4100
    with pytest.raises(adapter.LogicalSourceError) as failure:
        form.read_logical_field(input, ("drive", "waveform", "length"), empty)
    assert failure.value.selector_path == ("drive", "waveform")


def test_root_nested_sections_and_captured_inline_subtrees_never_expose_wrappers() -> (
    None
):
    from zcu_tools.cfg_model import FlatTopWaveformSpec

    class Container(ConfigBase):
        modules: LogicalCfg

    form = adapter.Form(Container, {})
    env = captured()
    assert env.library is not None
    pulse = PulseSpec(
        ch=1,
        nqz=1,
        freq=12,
        gain=0.2,
        waveform=FlatTopWaveformSpec(length=2, raise_waveform="Q.waveforms.edge"),
    )
    env = FormEnv(library=env.library.replaced_module("Q.pulses.drive", pulse))
    input = FormInput(
        form.model, {"modules": {"drive": {"kind": "ref", "name": "Q.pulses.drive"}}}
    )
    copied = form.prepare_inline_copy(input, ("modules", "drive"), env).input
    root = ordinary_object(form.read_logical_field(copied, (), env).value)
    drive = ordinary_object(ordinary_object(root["modules"])["drive"])
    assert drive.get("freq") == 12
    assert "module" not in drive
    source = form.read_logical_field(input, ("modules", "drive"), env)
    waveform = ordinary_object(ordinary_object(source.value)["waveform"])
    assert waveform["raise_waveform"] == "Q.waveforms.edge"
    assert source.errors is None


def test_source_admission_reports_consuming_edit_not_later_batch_item() -> None:
    from zcu_tools.gui.form import field_error

    class Consumer(FormModel):
        name: str = "ok"

        @classmethod
        def check_sources(cls, data: Mapping[str, JsonValue], env: FormEnv) -> None:
            del env
            if data.get("name") == "missing":
                raise field_error(("name",), "Missing synthetic source")

    form = adapter.Form(LogicalCfg, {}, extras=Consumer, lower=lambda extras, cfg: cfg)
    input = form.default_input(FormEnv())
    with pytest.raises(adapter.LogicalEditError) as failure:
        form.prepare_logical_edits(
            input,
            [
                adapter.LogicalEdit(("extras", "name"), "missing"),
                adapter.LogicalEdit(("reps",), 12),
            ],
            FormEnv(),
        )
    assert failure.value.reason_code == "form_source"
    assert failure.value.edit_index == 0 and failure.value.path == ("extras", "name")
    assert input.data["reps"] == 1


def test_missing_null_and_failed_scalar_values_have_distinct_statuses() -> None:
    class Values(FormModel):
        missing: int
        explicit: int | None = None
        invalid: int = 1

    form = adapter.Form(LogicalCfg, {}, extras=Values, lower=lambda extras, cfg: cfg)
    input = FormInput(
        form.model, {"extras": {"explicit": None, "invalid": {"expr": "1 / 0"}}}
    )
    missing = form.read_logical_field(input, ("extras", "missing"), FormEnv())
    assert missing.value_status == "missing" and not missing.input_present
    explicit = form.read_logical_field(input, ("extras", "explicit"), FormEnv())
    assert (
        explicit.value_status == "available"
        and explicit.input_present
        and explicit.value is None
    )
    failed = form.read_logical_field(input, ("extras", "invalid"), FormEnv())
    assert (
        failed.value_status == "unavailable"
        and failed.input_present
        and failed.value is None
    )


def test_locked_selector_rejects_library_and_preserves_inline_noop() -> None:
    form = adapter.Form(LogicalCfg, {"drive": Template("pulse", lock={"freq": 7.0})})
    form.bind_templates(
        {
            "pulse": TemplateDef(
                PulseSpec,
                "Locked pulse",
                {
                    "ch": 1,
                    "nqz": 1,
                    "freq": 1.0,
                    "gain": 0.2,
                    "waveform.style": "const",
                    "waveform.length": 0.1,
                },
            )
        },
        adapter="synthetic",
    )
    env = captured()
    input = form.default_input(env)
    before = input.to_json()
    observed = form.read_logical_field(input, ("drive",), env)
    assert (
        observed.description.target is not None
        and observed.description.target.selection_fixed
    )
    with pytest.raises(adapter.LogicalEditError) as failure:
        form.prepare_library_selection(input, ("drive",), "Q.pulses.drive", env)
    assert failure.value.reason_code == "module_selector"
    assert not form.prepare_inline_copy(input, ("drive",), env).changed
    assert input.to_json() == before


def test_ordinary_union_partial_write_keeps_core_defaults_and_initial_active_structure() -> (
    None
):
    from typing import Annotated, Literal

    class Flat(FormModel):
        kind: Literal["flat"] = "flat"
        length: float = 1.0

    class Ramp(FormModel):
        kind: Literal["ramp"] = "ramp"
        start: float = 0.0
        stop: float = 1.0

    class Extras(FormModel):
        shape: Annotated[Flat | Ramp, Field(discriminator="kind")]

    form = adapter.Form(LogicalCfg, {}, extras=Extras, lower=lambda extras, cfg: cfg)
    input = FormInput(
        form.model, {"extras": {"shape": {"kind": "flat", "length": 9.0}}}
    )
    env = FormEnv()
    candidate = form.prepare_logical_edits(
        input,
        [adapter.LogicalEdit(("extras", "shape"), {"kind": "ramp", "start": 3.0})],
        env,
    )
    assert object_at(candidate.input.data, ("extras", "shape")) == {
        "kind": "ramp",
        "start": 3.0,
        "stop": 1.0,
    }
    with pytest.raises(adapter.LogicalEditError) as failure:
        form.prepare_logical_edits(
            input,
            [
                adapter.LogicalEdit(("reps",), 8),
                adapter.LogicalEdit(("extras", "shape", "start"), 3),
            ],
            env,
        )
    assert failure.value.reason_code == "form_path" and failure.value.edit_index == 1


def test_ref_selector_observation_keeps_affecting_ancestor_failure() -> None:
    from typing import Self

    from pydantic import create_model, model_validator

    class Parent(FormModel):
        @model_validator(mode="after")
        def fail(self) -> Self:
            raise ValueError("Ancestor failure")

    field = adapter.ModuleInputField(PulseSpec)
    extras = create_model(
        "InvalidParent", __base__=Parent, reference=(field.annotation, ...)
    )
    form = adapter.Form(LogicalCfg, {}, extras=extras, lower=lambda extras, cfg: cfg)
    input = FormInput(
        form.model, {"extras": {"reference": {"kind": "ref", "name": "Q.pulses.drive"}}}
    )
    observed = form.read_logical_field(input, ("extras", "reference"), captured())
    assert observed.errors is not None and ("extras",) in observed.errors.messages
