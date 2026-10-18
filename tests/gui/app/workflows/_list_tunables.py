"""Synthetic declaration whose Form and native tunables have different shapes."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from zcu_tools.gui.form import FixedList, FormModel, Participants


class Channel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    amplitude: float = Field(gt=0, le=10)


class NativeTunables(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channels: list[Channel]
    participants: list[str]


class Tone(FormModel):
    label: str
    gain: float = Field(gt=0)


class ListForm(FormModel):
    tones: Annotated[list[Tone], FixedList(name_field="label")] = Field(
        default_factory=lambda: [Tone(label="A", gain=1), Tone(label="B", gain=2)]
    )
    participants: Annotated[list[str], Participants(list_path=("tones",))] = Field(
        default_factory=lambda: ["B"]
    )

    def to_cfg(self) -> NativeTunables:
        return NativeTunables(
            channels=[Channel(name=t.label, amplitude=t.gain) for t in self.tones],
            participants=self.participants,
        )
