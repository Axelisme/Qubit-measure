from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    NotRequired,
    Optional,
    Self,
    Type,
    TypeAlias,
    TypedDict,
    Union,
)

from ..base import MyProgramV2
from .base import ConfigBase

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary


class BaseWaveformCfg(ConfigBase):
    @classmethod
    def from_dict(cls, raw_cfg: dict[str, Any], ml: "ModuleLibrary") -> Self:
        return cls.model_validate(raw_cfg)


class ConstWaveformCfg(BaseWaveformCfg):
    style: Literal["const"] = "const"
    length: Union[float, QickParam]

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if name == "length":
            self.length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


class CosineWaveformCfg(BaseWaveformCfg):
    style: Literal["cosine"] = "cosine"
    length: float

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if name != "length":
            raise ValueError(f"Unknown parameter: {name}")
        if isinstance(value, QickParam):
            raise ValueError("Cosine waveform length must be a float value")
        self.length = value


class GaussWaveformCfg(BaseWaveformCfg):
    style: Literal["gauss"] = "gauss"
    length: float
    sigma: float

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if isinstance(value, QickParam):
            raise ValueError(f"Gauss waveform {name} must not be a QickParam")

        if name == "length":
            sigma_ratio = self.sigma / self.length
            self.length = value
            self.sigma = sigma_ratio * value
        elif name == "sigma":
            self.sigma = value
        elif name == "only_length":
            self.length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


class DragWaveformCfg(BaseWaveformCfg):
    style: Literal["drag"] = "drag"
    length: float
    sigma: float
    delta: float
    alpha: float

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if isinstance(value, QickParam):
            raise ValueError(f"Drag waveform {name} must not be a QickParam")

        if name == "length":
            sigma_ratio = self.sigma / self.length
            self.length = value
            self.sigma = sigma_ratio * value
        elif name == "sigma":
            self.sigma = value
        elif name == "delta":
            self.delta = value
        elif name == "alpha":
            self.alpha = value
        elif name == "only_length":
            self.length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


class ArbWaveformCfg(BaseWaveformCfg):
    style: Literal["arb"] = "arb"
    length: float
    data: str

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        raise ValueError("Arb waveform length and data cannot be changed")


RaiseWaveformCfg: TypeAlias = Union[
    CosineWaveformCfg, GaussWaveformCfg, DragWaveformCfg, ArbWaveformCfg
]


class FlatTopWaveformCfg(BaseWaveformCfg):
    style: Literal["flat_top"] = "flat_top"
    length: Union[float, QickParam]
    raise_waveform: RaiseWaveformCfg

    def set_param(self, param_name: str, param_value: Union[float, QickParam]) -> None:
        if param_name == "length":
            self.length = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")


WaveformCfg: TypeAlias = Union[
    ConstWaveformCfg,
    CosineWaveformCfg,
    GaussWaveformCfg,
    DragWaveformCfg,
    ArbWaveformCfg,
    FlatTopWaveformCfg,
]


class QickWaveformKwargs(TypedDict):
    style: Literal["const", "arb", "flat_top"]

    length: NotRequired[Union[float, QickParam]]
    envelope: NotRequired[str]


class AbsWaveform(ABC):
    def __init__(self, name: str, waveform_cfg: WaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> Union[float, QickParam]:
        return self.waveform_cfg.length

    @abstractmethod
    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None: ...

    @abstractmethod
    def to_wav_kwargs(self) -> QickWaveformKwargs: ...


class Waveform(AbsWaveform):
    _supported_waveforms: ClassVar[dict[str, type[AbsWaveform]]] = {}

    @classmethod
    def register_waveform(
        cls, style: str
    ) -> Callable[[Type[AbsWaveform]], Type[AbsWaveform]]:
        if style in cls._supported_waveforms:
            raise ValueError(
                f"Waveform style {style} already registered by {cls._supported_waveforms[style].__name__}"
            )

        def decorator(wav_cls: Type[AbsWaveform]) -> Type[AbsWaveform]:
            cls._supported_waveforms[style] = wav_cls
            return wav_cls

        return decorator

    def __init__(self, name: str, waveform_cfg: WaveformCfg) -> None:
        waveform_style = waveform_cfg.style
        if waveform_style not in self._supported_waveforms:
            raise ValueError(f"Unknown waveform style: {waveform_style}")
        self.waveform = self._supported_waveforms[waveform_style](name, waveform_cfg)

    @property
    def name(self) -> str:
        return self.waveform.name

    @property
    def length(self) -> Union[float, QickParam]:
        return self.waveform.length

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        self.waveform.create(prog, ch, **kwargs)

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return self.waveform.to_wav_kwargs()


@Waveform.register_waveform("const")
class ConstWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: ConstWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None: ...

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {"style": "const", "length": self.length}


@Waveform.register_waveform("cosine")
class CosineWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: CosineWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        return self.waveform_cfg.length

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        prog.add_cosine(ch, self.name, length=self.length, **kwargs)

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {"style": "arb", "envelope": self.name}


@Waveform.register_waveform("gauss")
class GaussWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: GaussWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        return self.waveform_cfg.length

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        wav_cfg = self.waveform_cfg
        prog.add_gauss(
            ch,
            self.name,
            sigma=wav_cfg.sigma,
            length=wav_cfg.length,
            **kwargs,
        )

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {"style": "arb", "envelope": self.name}


@Waveform.register_waveform("drag")
class DragWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: DragWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        return self.waveform_cfg.length

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        wav_cfg = self.waveform_cfg
        prog.add_DRAG(
            ch,
            self.name,
            sigma=wav_cfg.sigma,
            length=wav_cfg.length,
            delta=wav_cfg.delta,
            alpha=wav_cfg.alpha,
            **kwargs,
        )

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {"style": "arb", "envelope": self.name}


@Waveform.register_waveform("arb")
class ArbWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: ArbWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        return self.waveform_cfg.length

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        idata, qdata = self.make_iqdata(ch, prog, **kwargs)
        prog.add_envelope(ch, self.name, idata=idata, qdata=qdata)

    def make_iqdata(
        self, ch: int, prog: MyProgramV2, even_length: bool = False
    ) -> tuple[NDArray, Optional[NDArray]]:
        # lazy import to avoid circular import
        from zcu_tools.meta_tool.arb_waveform import ArbWaveformDatabase

        idata_raw, qdata_raw, time_raw = ArbWaveformDatabase.get(self.waveform_cfg.data)

        maxv = prog.soccfg.get_maxv(ch)
        samps_per_clk = prog.soccfg["gens"][ch]["samps_per_clk"]
        length = self.length

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

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {"style": "arb", "envelope": self.name}


@Waveform.register_waveform("flat_top")
class FlatTopWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: FlatTopWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

        self.raise_waveform = Waveform(name, waveform_cfg.raise_waveform)

    def create(self, prog: MyProgramV2, ch: int, **kwargs) -> None:
        kwargs.setdefault("even_length", True)
        self.raise_waveform.create(prog, ch, **kwargs)

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "flat_top",
            "envelope": self.name,
            "length": self.length - self.raise_waveform.length,
        }
