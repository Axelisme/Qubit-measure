"""Key-only refs share input semantics without ModuleRef's source requirement."""

import pytest
from pydantic import JsonValue, create_model
from zcu_tools.gui import form


@pytest.mark.parametrize(
    "name", ["$missing.value", {"expr": "1 / 0"}], ids=["literal", "envelope"]
)
def test_key_ref_defaults_schema_result_and_restore_do_not_evaluate_names(
    name: JsonValue,
) -> None:
    assert hasattr(form, "KeyRef"), "Missing key-only reference input type"
    model = create_model("SeededKeyRef", __base__=form.KeyRef, name=(str, name))
    env = form.FormEnv()
    state = form.FormState(model.default_input(env))
    assert state.input.data == {"kind": "ref", "name": name}
    properties = model.input_schema(env)["properties"]
    assert isinstance(properties, dict)
    schema = properties["name"]
    assert isinstance(schema, dict) and schema.get("type") == "string"
    result = state.result(env)
    if isinstance(name, str):
        assert isinstance(result, form.KeyRef) and result.name == name
        restored = form.FormState(
            form.FormInput(model, {"kind": "ref", "name": "other"})
        )
        assert restored.write_compatible(state.input.data, env) == ()
        assert restored.input.data["name"] == name
    else:
        assert isinstance(result, form.FormErrors)
        assert result.messages == {("name",): "Input should be a valid string"}
        restored = form.FormState(
            form.FormInput(model, {"kind": "ref", "name": "other"})
        )
        assert restored.write_compatible(state.input.data, env) == (("name",),)
        assert restored.input.data["name"] == "other"
