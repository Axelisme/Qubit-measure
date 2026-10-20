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
from zcu_tools.gui.form import FormEnv, FormErrors, FormState
from zcu_tools.gui.form_adapter import Form
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


def module_form(node: Template | Slot) -> Form[ModuleOwner]:
    form = Form(ModuleOwner, {"readout": node})
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
