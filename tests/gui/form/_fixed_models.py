"""Synthetic named elements shared by core, view and host contract scenarios."""

from typing import Annotated

from pydantic import Field, JsonValue, create_model
from zcu_tools.gui import form as api
from zcu_tools.gui.form import FormInput, FormModel


class Element(FormModel):
    """Experiment-owned identity spelling and a constrained editable value."""

    label: str
    gain: float = Field(gt=0)


def named_form() -> type[FormModel]:
    """Declare fixed identity without assuming the field is called name."""
    assert hasattr(api, "FixedList"), "Missing fixed model-list declaration"
    assert hasattr(api, "Participants"), "Missing independent participants declaration"
    return create_model(
        "NamedElements",
        __base__=FormModel,
        tones=(Annotated[list[Element], api.FixedList(name_field="label")], ...),
        participants=(
            Annotated[list[str], api.Participants(list_path=("tones",))],
            ...,
        ),
    )


def named_input(model: type[FormModel]) -> FormInput[FormModel]:
    """Two different captured initial values, with only the second participating."""
    data: dict[str, JsonValue] = {
        "tones": [{"label": "A", "gain": 1}, {"label": "B", "gain": 2}],
        "participants": ["B"],
    }
    return FormInput(model, data)
