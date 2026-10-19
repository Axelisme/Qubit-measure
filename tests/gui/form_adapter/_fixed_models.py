"""Synthetic declaration-owned fixed elements shared by views and hosts."""

from typing import Annotated

from pydantic import Field, JsonValue, create_model
from zcu_tools.gui.form import Display, FormInput, FormModel, InputPolicy
from zcu_tools.gui.form_adapter import FixedElements, FixedElementsForm


class Element(FormModel):
    """Owner chooses its identity spelling, literal input and presentation."""

    label: Annotated[str, Display(readonly=True), InputPolicy(expression=False)]
    gain: float = Field(gt=0)


def named_form() -> type[FixedElementsForm]:
    """Declare the owner's fixed list without assuming the field is called name."""
    model = create_model(
        "NamedElements",
        __base__=FixedElementsForm,
        tones=(Annotated[list[Element], Display(element_label="label")], ...),
        participants=(list[str], ...),
    )
    model.fixed_elements = (FixedElements(("tones",), "label", ("participants",)),)
    return model


def named_input(model: type[FormModel]) -> FormInput[FormModel]:
    """Two different captured values, with only the second participating."""
    data: dict[str, JsonValue] = {
        "tones": [{"label": "A", "gain": 1}, {"label": "B", "gain": 2}],
        "participants": ["B"],
    }
    return FormInput(model, data)
