from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Type

from qick.asm_v2 import QickParam

from ..base import MyProgramV2


class AbsWaveform(ABC):
    def __init__(self, name: str, waveform_cfg: Dict[str, Any]) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None: ...

    @classmethod
    def set_param(
        cls, waveform_cfg: Dict[str, Any], param_name: str, param_value: QickParam
    ) -> None:
        raise NotImplementedError(
            f"{cls.__name__} does not support set {param_name} params with {param_value}"
        )

    @abstractmethod
    def to_wav_kwargs(self) -> Dict[str, Any]: ...


waveform_support_styles = {}


def register_waveform(style: str) -> Callable[[Type[AbsWaveform]], Type[AbsWaveform]]:
    global waveform_support_styles

    if style in waveform_support_styles:
        raise ValueError(
            f"Waveform style {style} already registered by {waveform_support_styles[style].__name__}"
        )

    def decorator(cls: Type[AbsWaveform]) -> Type[AbsWaveform]:
        waveform_support_styles[style] = cls
        return cls

    return decorator


def make_waveform(name: str, waveform_cfg: Dict[str, Any]) -> AbsWaveform:
    style = waveform_cfg["style"]

    if style not in waveform_support_styles:
        raise ValueError(f"Unknown waveform style: {style}")

    return waveform_support_styles[style](name, waveform_cfg)


def set_waveform_param(
    waveform_cfg: Dict[str, Any], param_name: str, param_value: QickParam
) -> None:
    style = waveform_cfg["style"]
    if style not in waveform_support_styles:
        raise ValueError(f"Unknown waveform style: {style}")

    waveform_support_styles[style].set_param(waveform_cfg, param_name, param_value)


@register_waveform("const")
class ConstWaveform(AbsWaveform):
    @classmethod
    def set_param(
        cls, waveform_cfg: Dict[str, Any], param_name: str, param_value: QickParam
    ) -> None:
        if param_name == "on/off":
            waveform_cfg["length"] = (
                param_value * (waveform_cfg["length"] - 0.01) + 0.01
            )
        elif param_name == "length":
            waveform_cfg["length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    def to_wav_kwargs(self) -> Dict[str, Any]:
        return {
            "style": "const",
            "length": self.waveform_cfg["length"],
        }


@register_waveform("cosine")
class CosineWaveform(AbsWaveform):
    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        prog.add_cosine(ch, self.name, length=self.waveform_cfg["length"], **kwargs)

    def to_wav_kwargs(self) -> Dict[str, Any]:
        return {
            "style": "arb",
            "envelope": self.name,
        }


@register_waveform("gauss")
class GaussWaveform(AbsWaveform):
    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        prog.add_gauss(
            ch,
            self.name,
            sigma=self.waveform_cfg["sigma"],
            length=self.waveform_cfg["length"],
            **kwargs,
        )

    def to_wav_kwargs(self) -> Dict[str, Any]:
        return {
            "style": "arb",
            "envelope": self.name,
        }


@register_waveform("drag")
class DragWaveform(AbsWaveform):
    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        prog.add_DRAG(
            ch,
            self.name,
            sigma=self.waveform_cfg["sigma"],
            length=self.waveform_cfg["length"],
            delta=self.waveform_cfg["delta"],
            alpha=self.waveform_cfg["alpha"],
            **kwargs,
        )

    def to_wav_kwargs(self) -> Dict[str, Any]:
        return {
            "style": "arb",
            "envelope": self.name,
        }


@register_waveform("flat_top")
class FlatTopWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: Dict[str, Any]) -> None:
        super().__init__(name, waveform_cfg)

        raise_cfg = waveform_cfg["raise_waveform"]
        if raise_cfg["style"] == "flat_top":
            raise ValueError("Nested flat top pulses are not supported")

        if raise_cfg["style"] == "const":
            raise ValueError("Flat top with constant raise style is not supported")

        self.raise_waveform = make_waveform(name, raise_cfg)

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        kwargs.setdefault("even_length", True)

        self.raise_waveform.create(prog, ch, **kwargs)

    @classmethod
    def set_param(
        cls, waveform_cfg: Dict[str, Any], param_name: str, param_value: QickParam
    ) -> None:
        if param_name == "on/off":
            min_length = 0.01 + waveform_cfg["raise_waveform"]["length"]
            waveform_cfg["length"] = min_length + param_value * (
                waveform_cfg["length"] - min_length
            )
        elif param_name == "length":
            waveform_cfg["length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    def to_wav_kwargs(self) -> Dict[str, Any]:
        return {
            "style": "flat_top",
            "envelope": self.name,
            "length": self.waveform_cfg["length"]
            - self.waveform_cfg["raise_waveform"]["length"],
        }
