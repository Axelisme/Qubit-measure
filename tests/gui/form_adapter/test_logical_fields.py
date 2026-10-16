"""Logical edits and captured Spec observations through the public Form seam."""

import pytest
from pydantic import Field, JsonValue
from zcu_tools.cfg_model import ConfigBase, PulseSpec, Template, TemplateDef
from zcu_tools.gui import form_adapter as adapter
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPathError,
    FormState,
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
                        waveforms={"wave": {"style": "const", "length": 0.08}},
                    )
                },
            )
        )
    )


def keyed(form: adapter.Form, env: FormEnv) -> FormInput[FormModel]:
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
    assert candidate.input.data["drive"]["module"]["waveform"] == {
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
    assert result.input.data["drive"]["module"]["waveform"]["module"]["length"] == {
        "expr": "1 / 4"
    }
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
    assert observed.value["waveform"] == "Q.waveforms.wave"
    unavailable = form.read_logical_field(input, ("drive",), FormEnv())
    assert (
        unavailable.selector.mode == "key" and unavailable.value_status == "unavailable"
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
    assert copied.input.data["extras"]["reference"]["kind"] == "inline"
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
                    "waveform": {"style": "const", "length": 0.1},
                },
            )
        },
        env,
    )
    locked = spec.read_logical_field(template, ("freq",), env)
    assert locked.description.target.fixed and locked.value == 42.0
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
