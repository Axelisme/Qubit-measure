from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickParam
from typing_extensions import (
    Callable,
    ClassVar,
    Literal,
    NotRequired,
    Optional,
    Type,
    TypeAlias,
    TypedDict,
    Union,
    cast,
)

from ..base import MyProgramV2


class ConstWaveformCfg(TypedDict, closed=True):
    style: Literal["const"]
    length: Union[float, QickParam]


class CosineWaveformCfg(TypedDict, closed=True):
    style: Literal["cosine"]
    length: float


class GaussWaveformCfg(TypedDict, closed=True):
    style: Literal["gauss"]
    length: float
    sigma: float


class DragWaveformCfg(TypedDict, closed=True):
    style: Literal["drag"]
    length: float
    sigma: float
    delta: float
    alpha: float


class FlatTopWaveformCfg(TypedDict, closed=True):
    style: Literal["flat_top"]
    length: Union[float, QickParam]
    raise_waveform: Union[CosineWaveformCfg, GaussWaveformCfg, DragWaveformCfg]


class ArbWaveformCfg(TypedDict, closed=True):
    style: Literal["arb"]
    length: float
    data: str


WaveformCfg: TypeAlias = Union[
    ConstWaveformCfg,
    CosineWaveformCfg,
    GaussWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    ArbWaveformCfg,
]


class QickWaveformKwargs(TypedDict):
    style: Literal["const", "arb", "flat_top"]

    length: NotRequired[Union[float, QickParam]]
    envelope: NotRequired[str]


class AbsWaveform(ABC):
    def __init__(self, name: str, waveform_cfg: WaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None: ...

    @property
    def length(self) -> Union[float, QickParam]:
        return self.waveform_cfg["length"]

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
    SUPPORTED_STYLES: ClassVar[dict[str, Type[AbsWaveform]]] = {}

    @classmethod
    def register(cls, style: str) -> Callable[[Type[AbsWaveform]], Type[AbsWaveform]]:
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

    @property
    def length(self) -> Union[float, QickParam]:
        return self.waveform.length

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return self.waveform.to_wav_kwargs()

    @property
    def name(self) -> str:
        return self.waveform.name

    @property
    def waveform_cfg(self) -> WaveformCfg:
        return self.waveform.waveform_cfg


@Waveform.register("const")
class ConstWaveform(AbsWaveform):
    @classmethod
    def set_param(
        cls,
        waveform_cfg: WaveformCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> ConstWaveformCfg:
        wav_cfg = cast(ConstWaveformCfg, waveform_cfg)

        if param_name == "length":
            wav_cfg["length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return wav_cfg

    @property
    def length(self) -> Union[float, QickParam]:
        return self.waveform_cfg["length"]

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "const",
            "length": self.waveform_cfg["length"],
        }


@Waveform.register("cosine")
class CosineWaveform(AbsWaveform):
    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        prog.add_cosine(ch, self.name, length=self.waveform_cfg["length"], **kwargs)

    @classmethod
    def set_param(
        cls,
        waveform_cfg: WaveformCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> CosineWaveformCfg:
        wav_cfg = cast(CosineWaveformCfg, waveform_cfg)

        if param_name == "length":
            if isinstance(param_value, QickParam):
                raise ValueError("Cosine waveform length must be a float value")
            wav_cfg["length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return wav_cfg

    @property
    def length(self) -> float:
        return cast(CosineWaveformCfg, self.waveform_cfg)["length"]

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "arb",
            "envelope": self.name,
        }


@Waveform.register("gauss")
class GaussWaveform(AbsWaveform):
    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        wav_cfg = cast(GaussWaveformCfg, self.waveform_cfg)
        prog.add_gauss(
            ch,
            self.name,
            sigma=wav_cfg["sigma"],
            length=wav_cfg["length"],
            **kwargs,
        )

    @classmethod
    def set_param(
        cls,
        waveform_cfg: WaveformCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> GaussWaveformCfg:
        wav_cfg = cast(GaussWaveformCfg, waveform_cfg)

        if isinstance(param_value, QickParam):
            raise ValueError(f"Gauss waveform {param_name} must not be a QickParam")

        if param_name == "length":
            if "sigma" not in wav_cfg or "length" not in wav_cfg:
                raise ValueError(
                    "Set Gauss waveform length must provide reference length and sigma"
                )
            sigma_ratio = wav_cfg["sigma"] / wav_cfg["length"]
            wav_cfg["length"] = param_value
            wav_cfg["sigma"] = sigma_ratio * param_value
        elif param_name == "sigma":
            wav_cfg["sigma"] = param_value
        elif param_name == "only_length":
            wav_cfg["length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return wav_cfg

    @property
    def length(self) -> float:
        return cast(GaussWaveformCfg, self.waveform_cfg)["length"]

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "arb",
            "envelope": self.name,
        }


@Waveform.register("drag")
class DragWaveform(AbsWaveform):
    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        wav_cfg = cast(DragWaveformCfg, self.waveform_cfg)
        prog.add_DRAG(
            ch,
            self.name,
            sigma=wav_cfg["sigma"],
            length=wav_cfg["length"],
            delta=wav_cfg["delta"],
            alpha=wav_cfg["alpha"],
            **kwargs,
        )

    @classmethod
    def set_param(
        cls,
        waveform_cfg: WaveformCfg,
        param_name: str,
        param_value: Union[float, QickParam],
    ) -> DragWaveformCfg:
        wav_cfg = cast(DragWaveformCfg, waveform_cfg)

        if isinstance(param_value, QickParam):
            raise ValueError(f"Drag waveform {param_name} must not be a QickParam")

        if param_name == "length":
            if "sigma" not in wav_cfg or "length" not in wav_cfg:
                raise ValueError(
                    "Set Drag waveform length must provide reference length and sigma"
                )
            sigma_ratio = wav_cfg["sigma"] / wav_cfg["length"]
            wav_cfg["length"] = param_value
            wav_cfg["sigma"] = sigma_ratio * param_value
        elif param_name == "sigma":
            wav_cfg["sigma"] = param_value
        elif param_name == "delta":
            wav_cfg["delta"] = param_value
        elif param_name == "alpha":
            wav_cfg["alpha"] = param_value
        elif param_name == "only_length":
            wav_cfg["length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return wav_cfg

    @property
    def length(self) -> float:
        return cast(DragWaveformCfg, self.waveform_cfg)["length"]

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "arb",
            "envelope": self.name,
        }


@Waveform.register("flat_top")
class FlatTopWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: WaveformCfg) -> None:
        super().__init__(name, waveform_cfg)

        wav_cfg = cast(FlatTopWaveformCfg, self.waveform_cfg)

        raise_cfg = wav_cfg["raise_waveform"]
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
    ) -> FlatTopWaveformCfg:
        wav_cfg = cast(FlatTopWaveformCfg, waveform_cfg)

        if param_name == "length":
            wav_cfg["length"] = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        return wav_cfg

    @property
    def length(self) -> Union[float, QickParam]:
        return self.waveform_cfg["length"]

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        wav_cfg = cast(FlatTopWaveformCfg, self.waveform_cfg)
        return {
            "style": "flat_top",
            "envelope": self.name,
            "length": wav_cfg["length"] - wav_cfg["raise_waveform"]["length"],
        }


@Waveform.register("arb")
class ArbWaveform(AbsWaveform):
    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        idata, qdata = self.make_iqdata(ch, prog, **kwargs)
        prog.add_envelope(ch, self.name, idata=idata, qdata=qdata)

    def make_iqdata(
        self, ch: int, prog: MyProgramV2, even_length: bool = False
    ) -> tuple[NDArray, Optional[NDArray]]:
        # lazy import to avoid circular import
        from zcu_tools.meta_tool.arb_waveform import ArbWaveformDatabase

        wav_cfg = cast(ArbWaveformCfg, self.waveform_cfg)
        idata_raw, qdata_raw, time_raw = ArbWaveformDatabase.get(wav_cfg["data"])

        maxv = prog.soccfg.get_maxv(ch)
        samps_per_clk = prog.soccfg["gens"][ch]["samps_per_clk"]
        length = wav_cfg["length"]

        if even_length:
            n_clks = 2 * prog.us2cycles(gen_ch=ch, us=length / 2)
        else:
            n_clks = prog.us2cycles(gen_ch=ch, us=length)
        n_samples = int(n_clks * samps_per_clk)

        target_time = np.linspace(0, length, n_samples, endpoint=False)
        idata = np.interp(target_time, time_raw, idata_raw, left=0, right=0) * maxv

        qdata = None
        if qdata_raw is not None:
            qdata = np.interp(target_time, time_raw, qdata_raw, left=0, right=0) * maxv

        return idata, qdata

    @property
    def length(self) -> float:
        return cast(ArbWaveformCfg, self.waveform_cfg)["length"]

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "arb",
            "envelope": self.name,
        }
