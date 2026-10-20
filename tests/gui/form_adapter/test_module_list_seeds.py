"""Typed model-list seeds enter modules through the public authoring boundary."""

import pytest
from zcu_tools.cfg_model import (
    ConfigBase,
    Expr,
    MuxReadoutCfg,
    MuxReadoutSpec,
    Slot,
    Template,
    TemplateDef,
    TemplateSeed,
)
from zcu_tools.gui.form import FormEnv, FormErrors, FormInputRejected, FormState
from zcu_tools.gui.form_adapter import FixedElements, Form
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class ModuleOwner(ConfigBase):
    readout: MuxReadoutCfg


def module_env() -> FormEnv:
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "R": ComponentLibrary(
                        readout={
                            "default": MuxReadoutSpec.model_validate(
                                {
                                    "gen_ch": 0,
                                    "nqz": 1,
                                    "mixer_freq": 6000.0,
                                    "length": 1.0,
                                    "ro_length": 0.5,
                                    "tones": [
                                        {"freq": 6000.0, "gain": 0.1, "ro_ch": 0}
                                    ],
                                }
                            )
                        }
                    )
                },
            )
        )
    )


def module_form(
    node: Template | Slot, *, fixed_elements: tuple[FixedElements, ...] = ()
) -> Form[ModuleOwner]:
    form = Form(ModuleOwner, {"readout": node}, fixed_elements=fixed_elements)
    form.bind_templates(
        {
            "blank": TemplateDef(
                MuxReadoutSpec,
                "Readout",
                {
                    "gen_ch": 0,
                    "nqz": 1,
                    "mixer_freq": 6000.0,
                    "length": 1.0,
                    "ro_length": 0.5,
                },
            )
        },
        adapter="synthetic",
    )
    return form


@pytest.mark.parametrize("source", ["template", "slot"])
def test_module_list_seeds_preserve_typed_fields_and_expression_defaults(source):
    seeds: list[dict[str, TemplateSeed]] = [
        {"freq": Expr("6000 + 2"), "gain": 0.1, "ro_ch": 4, "label": "A"},
        {"freq": 6010.0, "gain": 0.2, "ro_ch": 7, "label": "B"},
    ]
    node = (
        Template("blank", set={"tones": seeds})
        if source == "template"
        else Slot("R.readout.default", copy=True, set={"tones": seeds})
    )
    env = module_env()
    try:
        form = module_form(node)
        state = FormState(form.default_input(env))
    except (TypeError, ValueError) as error:
        pytest.fail(
            f"Typed model-list initial values must build editable input: {error}"
        )
    path = ("readout", "module", "tones")
    description = state.describe(path, env)
    assert description.target is not None and description.target.kind == "elements"
    assert [item.path for item in description.children] == [(*path, 0), (*path, 1)]
    values = state.preview(env).values
    assert values[(*path, 0, "freq")] == 6002.0
    assert values[(*path, 1, "freq")] == 6010.0
    assert seeds[0]["freq"] == Expr("6000 + 2")
    state.write({(*path, 1, "gain"): 2.0}, env)
    errors = state.result(env)
    assert isinstance(errors, FormErrors) and (*path, 1, "gain") in errors.messages
    state.write({(*path, 1, "gain"): 0.3, (*path, 1, "label"): "Edited"}, env)
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    assert [
        (tone.freq, tone.gain, tone.label, tone.ro_ch) for tone in cfg.readout.tones
    ] == [
        (6002.0, 0.1, "A", 4),
        (6010.0, 0.3, "Edited", 7),
    ]
    assert env.library is not None
    assert len(env.library.module_spec("R.readout.default").model_dump()["tones"]) == 1


def test_nested_fixed_slots_allow_value_edits_and_module_branch_switches():
    path = ("readout", "module", "tones")
    try:
        declaration = FixedElements(path)
    except TypeError as error:
        pytest.fail(f"Fixed slots must not require names or participants: {error}")
    form = module_form(
        Template(
            "blank",
            set={
                "tones": [
                    {"freq": 6000.0, "gain": 0.1, "ro_ch": 0, "label": "Same"},
                    {"freq": 6010.0, "gain": 0.2, "ro_ch": 1, "label": "Same"},
                ]
            },
        ),
        fixed_elements=(declaration,),
    )
    env = module_env()
    try:
        state = FormState(form.default_input(env))
    except ValueError as error:
        pytest.fail(f"Fixed slots must locate the nested active inline list: {error}")
    state.write(
        {
            (*path, 0, "freq"): 6002.0,
            (*path, 0, "gain"): 0.3,
            (*path, 0, "ro_ch"): 4,
            (*path, 0, "label"): "Edited",
        },
        env,
    )
    values = state.preview(env).values
    assert [
        values[(*path, 0, field)] for field in ("freq", "gain", "ro_ch", "label")
    ] == [
        6002.0,
        0.3,
        4,
        "Edited",
    ]
    saved = state.input
    for replacement in ([], [{"freq": 6000.0, "gain": 0.1, "ro_ch": 0}], None):
        with pytest.raises(FormInputRejected):
            state.write({path: replacement, ("readout", "module", "length"): 5.0}, env)
        assert state.input.same_as(saved)
    state.select(("readout",), {"kind": "ref", "name": "R.readout.default"}, env)
    assert form.module_input_kind(state.input, ("readout",)) == "ref"
    state.run_command(("readout",), "to_inline", env)
    assert form.module_input_kind(state.input, ("readout",)) == "inline"
    assert len(form.read_elements(state.input, path, env).elements) == 1
    state.write({(*path, 0, "label"): "Copied and editable"}, env)
    with pytest.raises(FormInputRejected):
        state.write(
            {
                path: [
                    {"freq": 6000.0, "gain": 0.1, "ro_ch": 0},
                    {"freq": 6010.0, "gain": 0.2, "ro_ch": 1},
                ]
            },
            env,
        )
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    assert [tone.label for tone in cfg.readout.tones] == ["Copied and editable"]
