"""Inline option fields reuse classes, not environment-dependent option data."""

from collections.abc import Callable

import pytest
from pydantic import BaseModel
from zcu_tools.cfg_model import ConfigBase, PulseSpec, ResetSpec
from zcu_tools.cfg_model import ResetCfg as ResetFamily
from zcu_tools.gui.form import FormEnv, Option
from zcu_tools.gui.form_adapter import Form, ModuleInputField, module_form


class ResetCfg(ConfigBase):
    reset: ResetFamily
    other: ResetFamily


def inline_fields(options: tuple[Option, ...]) -> dict[str, type[BaseModel]]:
    return {
        option.label: option.fields for option in options if option.fields is not None
    }


@pytest.mark.parametrize("owner", ["module_input", "core_form", "spec_form"])
def test_inline_fields_reuse_is_scoped_to_wrapper_and_kind(owner: str) -> None:
    read: Callable[[FormEnv], tuple[Option, ...]]
    other: Callable[[FormEnv], tuple[Option, ...]]
    if owner == "module_input":
        first = ModuleInputField(ResetSpec)
        second = ModuleInputField(ResetSpec)
        read, other = first.options, second.options
    elif owner == "core_form":
        form = Form(ResetCfg, {})
        read = lambda env: form.model.options(env)[("reset",)]
        other = lambda env: form.model.options(env)[("other",)]
    else:
        first_form = module_form(PulseSpec)
        second_form = module_form(PulseSpec)
        read = lambda env: first_form.model.options(env)[("waveform",)]
        other = lambda env: second_form.model.options(env)[("waveform",)]

    first_options = read(FormEnv())
    initial = inline_fields(first_options)
    assert len(initial) > 1
    assert len(set(initial.values())) == len(initial)  # Kinds must not share fields.
    isolated = inline_fields(other(FormEnv()))
    assert set(initial) == set(isolated)
    assert all(initial[label] is not isolated[label] for label in initial)
    repeated_options = read(FormEnv())
    assert [(option.label, option.value) for option in repeated_options] == [
        (option.label, option.value) for option in first_options
    ]
    repeated = inline_fields(repeated_options)
    assert all(repeated[label] is fields for label, fields in initial.items())
