from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Dict, Literal, Type, Union

from qick.asm_v2 import QickParam
from typing_extensions import NotRequired, TypedDict

from ..base import MyProgramV2


class WaveformCfg(TypedDict):
    style: Literal["const", "cosine", "gauss", "drag", "flat_top", "padding"]
    length: Union[float, QickParam]

    sigma: NotRequired[float]  # guassian sigma
    delta: NotRequired[float]  # drag delta
    alpha: NotRequired[float]  # drag alpha
    raise_waveform: NotRequired[WaveformCfg]  # flat top

    # for padding pulse
    pre_length: NotRequired[Union[float, QickParam]]
    post_length: NotRequired[Union[float, QickParam]]
    pre_gain: NotRequired[Union[float, QickParam]]
    post_gain: NotRequired[Union[float, QickParam]]


class QickWaveformKwargs(TypedDict):
    style: Literal["const", "arb", "flat_top"]

    length: NotRequired[Union[float, QickParam]]
    envelope: NotRequired[str]


class AbsWaveform(ABC):
    def __init__(self, name: str, waveform_cfg: WaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None: ...

    @classmethod
    def set_param(
        cls,
        waveform_cfg: WaveformCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> WaveformCfg:
        raise NotImplementedError(
            f"{cls.__name__} does not support set {param_name} params with {param_value}"
        )

    @abstractmethod
    def to_wav_kwargs(self) -> QickWaveformKwargs: ...


class Waveform(AbsWaveform):
    SUPPORTED_STYLES: ClassVar[Dict[str, Type[AbsWaveform]]] = {}

    @classmethod
    def register_waveform(
        cls, style: str
    ) -> Callable[[Type[AbsWaveform]], Type[AbsWaveform]]:
        if style in cls.SUPPORTED_STYLES:
            raise ValueError(
                f"Waveform style {style} already registered by {cls.SUPPORTED_STYLES[style].__name__}"
            )

        def decorator(cls: Type[AbsWaveform]) -> Type[AbsWaveform]:
            Waveform.SUPPORTED_STYLES[style] = cls
            return cls

        return decorator

    @classmethod
    def get_waveform_cls(cls, style: str) -> Type[AbsWaveform]:
        if style not in cls.SUPPORTED_STYLES:
            raise ValueError(f"Unknown waveform style: {style}")
        return cls.SUPPORTED_STYLES[style]

    @classmethod
    def set_param(
        cls,
        waveform_cfg: WaveformCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> WaveformCfg:
        return cls.get_waveform_cls(waveform_cfg["style"]).set_param(
            waveform_cfg, param_name, param_value
        )

    def __init__(self, name: str, waveform_cfg: WaveformCfg) -> None:
        waveform_cls = Waveform.get_waveform_cls(waveform_cfg["style"])
        self.waveform = waveform_cls(name, waveform_cfg)

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        self.waveform.create(prog, ch, **kwargs)

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return self.waveform.to_wav_kwargs()

    @property
    def name(self) -> str:
        return self.waveform.name

    @property
    def waveform_cfg(self) -> WaveformCfg:
        return self.waveform.waveform_cfg


@Waveform.register_waveform("const")
class ConstWaveform(AbsWaveform):
    @classmethod
    def set_param(
        cls,
        waveform_cfg: WaveformCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> WaveformCfg:
        if param_name == "on/off":
            waveform_cfg["length"] = (
                param_value * (waveform_cfg["length"] - 0.01) + 0.01
            )
        elif param_name == "length":
            waveform_cfg["length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return waveform_cfg

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "const",
            "length": self.waveform_cfg["length"],
        }


@Waveform.register_waveform("cosine")
class CosineWaveform(AbsWaveform):
    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        prog.add_cosine(ch, self.name, length=self.waveform_cfg["length"], **kwargs)

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "arb",
            "envelope": self.name,
        }


@Waveform.register_waveform("gauss")
class GaussWaveform(AbsWaveform):
    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        prog.add_gauss(
            ch,
            self.name,
            sigma=self.waveform_cfg["sigma"],
            length=self.waveform_cfg["length"],
            **kwargs,
        )

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "arb",
            "envelope": self.name,
        }


@Waveform.register_waveform("drag")
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

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "arb",
            "envelope": self.name,
        }


@Waveform.register_waveform("flat_top")
class FlatTopWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: WaveformCfg) -> None:
        super().__init__(name, waveform_cfg)

        raise_cfg = waveform_cfg["raise_waveform"]
        if raise_cfg["style"] == "flat_top":
            raise ValueError("Nested flat top pulses are not supported")

        if raise_cfg["style"] == "const":
            raise ValueError("Flat top with constant raise style is not supported")

        self.raise_waveform = Waveform(name, raise_cfg)

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        kwargs.setdefault("even_length", True)

        self.raise_waveform.create(prog, ch, **kwargs)

    @classmethod
    def set_param(
        cls,
        waveform_cfg: WaveformCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> WaveformCfg:
        if param_name == "on/off":
            min_length = 0.01 + waveform_cfg["raise_waveform"]["length"]
            waveform_cfg["length"] = min_length + param_value * (
                waveform_cfg["length"] - min_length
            )
        elif param_name == "length":
            waveform_cfg["length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return waveform_cfg

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "flat_top",
            "envelope": self.name,
            "length": self.waveform_cfg["length"]
            - self.waveform_cfg["raise_waveform"]["length"],
        }
