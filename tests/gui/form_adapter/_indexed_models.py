"""Native synthetic model-list declarations shared by owner and frontend tests."""

from typing import Annotated

from pydantic import Field, JsonValue
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    Expr,
    FieldMeta,
    FormNode,
    PulseCfg,
    PulseSpec,
    Range,
    Slot,
)
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import FixedElements, Form
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class NativeTone(ConfigBase):
    name: str
    drive: PulseCfg
    sweep: SweepCfg
    phase: float = 0


class NativeGroup(ConfigBase):
    tones: list[NativeTone]
    participants: list[str]


def indexed_env() -> FormEnv:
    """Capture a pulse and its waveform without parameter or device I/O."""
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        pulses={
                            "drive": PulseSpec(
                                ch=0,
                                nqz=1,
                                freq=100,
                                gain=0.2,
                                waveform=ConstWaveformCfg(length=0.1),
                            )
                        }
                    )
                },
            )
        )
    )


def indexed_tree() -> dict[str, FormNode]:
    """Seed by original list position, never by participant order."""
    return {
        "tones": [
            {
                "name": "A",
                "drive": Slot("Q.pulses.drive"),
                "sweep": Range(start=Expr("1 + 1"), stop=4, expts=3),
            },
            {
                "name": "B",
                "drive": Slot("Q.pulses.drive"),
                "sweep": Range.between(10, 14, expts=3),
            },
        ],
        "participants": ["B", "A"],
    }


def indexed_form() -> Form[NativeGroup]:
    """Use only the public generator and declaration-owned fixed policy."""
    return Form(
        NativeGroup,
        indexed_tree(),
        fixed_elements=(FixedElements(("tones",), "name", ("participants",)),),
    )


def displayed_form() -> Form[NativeGroup]:
    """Native metadata must not require GUI types in the native cfg."""

    class DisplayedTone(NativeTone):
        name: Annotated[str, FieldMeta(readonly=True)]

    class DisplayedGroup(NativeGroup):
        tones: Annotated[list[DisplayedTone], FieldMeta(element_label="name")]

    return Form(
        DisplayedGroup,
        indexed_tree(),
        fixed_elements=(FixedElements(("tones",), "name", ("participants",)),),
    )


def native_defaults_form() -> Form[NativeGroup]:
    """Native default-factory values must be projected to editable inputs."""

    class DefaultGroup(NativeGroup):
        tones: list[NativeTone] = Field(
            default_factory=lambda: [
                NativeTone(
                    name="default",
                    drive=PulseCfg(
                        ch=0,
                        nqz=1,
                        freq=42,
                        gain=0.3,
                        waveform=ConstWaveformCfg(length=0.2),
                    ),
                    sweep=SweepCfg(start=3, stop=7, expts=3, step=2),
                )
            ]
        )
        participants: list[str] = Field(default_factory=lambda: ["default"])

    return Form(DefaultGroup, {})


def element(data: dict[str, JsonValue], index: int) -> dict[str, JsonValue]:
    """Narrow a public authored element for assertions."""
    tones = data["tones"]
    assert isinstance(tones, list)
    item = tones[index]
    assert isinstance(item, dict)
    return item
