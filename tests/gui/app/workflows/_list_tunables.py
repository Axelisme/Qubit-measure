"""Synthetic declaration whose Form and native tunables have different shapes."""

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.gui.form import Display, FormModel, InputPolicy
from zcu_tools.gui.form_adapter import FixedElements, FixedElementsForm


class Channel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    amplitude: float = Field(gt=0, le=10)


class NativeTunables(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channels: list[Channel]
    participants: list[str]


class Tone(FormModel):
    label: Annotated[str, Display(readonly=True), InputPolicy(expression=False)]
    gain: float = Field(gt=0)


class ListForm(FixedElementsForm):
    fixed_elements: ClassVar[tuple[FixedElements, ...]] = (
        FixedElements(("tones",), "label", ("participants",)),
    )
    tones: Annotated[list[Tone], Display(element_label="label")] = Field(
        default_factory=lambda: [Tone(label="A", gain=1), Tone(label="B", gain=2)]
    )
    participants: list[str] = Field(default_factory=lambda: ["B"])

    def to_cfg(self) -> NativeTunables:
        return NativeTunables(
            channels=[Channel(name=t.label, amplitude=t.gain) for t in self.tones],
            participants=self.participants,
        )
