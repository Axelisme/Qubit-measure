from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, TypeAdapter
from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    Union,
    get_origin,
)

from zcu_tools.config import ConfigBase

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.program.v2.modular import ModularProgramV2


class WaveformCfg(ConfigBase):
    style: str
    length: Union[float, QickParam]

    def build(self, name: str) -> AbsWaveform:
        raise NotImplementedError(f"{type(self).__name__}.build is not implemented")

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support set_param")


class WaveformCfgFactory:
    _registry: ClassVar[dict[str, type[WaveformCfg]]] = {}
    _adapter: ClassVar[Optional[TypeAdapter]] = None

    @classmethod
    def register(cls, *waveform_cfgs: type[WaveformCfg]) -> None:
        for wc in waveform_cfgs:
            field = wc.model_fields.get("style")
            if field is None or get_origin(field.annotation) is not Literal:
                raise TypeError(
                    f"{wc.__name__} cannot be registered: missing Literal 'style' field"
                )
            style_value = field.default
            existing = cls._registry.get(style_value)
            if existing is not None and existing is not wc:
                raise ValueError(
                    f"Style discriminator {style_value!r} already registered to {existing.__name__}"
                )
            cls._registry[style_value] = wc
        cls._adapter = None

    @classmethod
    def _get_adapter(cls) -> TypeAdapter:
        if cls._adapter is None:
            if not cls._registry:
                raise RuntimeError("No WaveformCfg leaf subclasses registered")
            union = Annotated[
                Union[tuple(cls._registry.values())],
                Field(discriminator="style"),
            ]
            cls._adapter = TypeAdapter(union)
        return cls._adapter

    @classmethod
    def from_raw(cls, raw: Any, *, ml: Optional[ModuleLibrary] = None) -> WaveformCfg:
        ctx = {"ml": ml} if ml is not None else None
        return cls._get_adapter().validate_python(raw, context=ctx)


class ConstWaveformCfg(WaveformCfg):
    style: Literal["const"] = "const"
    length: Union[float, QickParam]

    def build(self, name: str) -> ConstWaveform:
        return ConstWaveform(name, self)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if name == "length":
            self.length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


class CosineWaveformCfg(WaveformCfg):
    style: Literal["cosine"] = "cosine"
    length: float

    def build(self, name: str) -> CosineWaveform:
        return CosineWaveform(name, self)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if name != "length":
            raise ValueError(f"Unknown parameter: {name}")
        if isinstance(value, QickParam):
            raise ValueError("Cosine waveform length must be a float value")
        self.length = value


class GaussWaveformCfg(WaveformCfg):
    style: Literal["gauss"] = "gauss"
    length: float
    sigma: float

    def build(self, name: str) -> GaussWaveform:
        return GaussWaveform(name, self)

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


class DragWaveformCfg(WaveformCfg):
    style: Literal["drag"] = "drag"
    length: float
    sigma: float
    delta: float
    alpha: float

    def build(self, name: str) -> DragWaveform:
        return DragWaveform(name, self)

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


class ArbWaveformCfg(WaveformCfg):
    style: Literal["arb"] = "arb"
    length: float
    data: str

    def build(self, name: str) -> ArbWaveform:
        return ArbWaveform(name, self)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        raise ValueError("Arb waveform length and data cannot be changed")


RaiseWaveformCfg: TypeAlias = Annotated[
    Union[CosineWaveformCfg, GaussWaveformCfg, DragWaveformCfg, ArbWaveformCfg],
    Field(discriminator="style"),
]


class FlatTopWaveformCfg(WaveformCfg):
    style: Literal["flat_top"] = "flat_top"
    length: Union[float, QickParam]
    raise_waveform: RaiseWaveformCfg

    def build(self, name: str) -> FlatTopWaveform:
        return FlatTopWaveform(name, self)

    def set_param(self, param_name: str, param_value: Union[float, QickParam]) -> None:
        if param_name == "length":
            self.length = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")


class QickWaveformKwargs(TypedDict):
    style: Literal["const", "arb", "flat_top"]

    length: NotRequired[Union[float, QickParam]]
    envelope: NotRequired[str]


class AbsWaveform(ABC):
    name: str
    waveform_cfg: WaveformCfg

    @property
    def length(self) -> Union[float, QickParam]:
        return self.waveform_cfg.length  # type: ignore[attr-defined]

    @abstractmethod
    def create(self, prog: ModularProgramV2, ch: int, **kwargs) -> None: ...

    @abstractmethod
    def to_wav_kwargs(self) -> QickWaveformKwargs: ...


class ConstWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: ConstWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    def create(self, prog: ModularProgramV2, ch: int, **kwargs) -> None: ...

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {"style": "const", "length": self.length}


class CosineWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: CosineWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        return self.waveform_cfg.length  # type: ignore[return-value]

    def create(self, prog: ModularProgramV2, ch: int, **kwargs) -> None:
        prog.add_cosine(ch, self.name, length=self.length, **kwargs)

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {"style": "arb", "envelope": self.name}


class GaussWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: GaussWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        return self.waveform_cfg.length  # type: ignore[return-value]

    def create(self, prog: ModularProgramV2, ch: int, **kwargs) -> None:
        wav_cfg = self.waveform_cfg
        assert isinstance(wav_cfg, GaussWaveformCfg)
        prog.add_gauss(
            ch,
            self.name,
            sigma=wav_cfg.sigma,
            length=wav_cfg.length,
            **kwargs,
        )

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {"style": "arb", "envelope": self.name}


class DragWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: DragWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        return self.waveform_cfg.length  # type: ignore[return-value]

    def create(self, prog: ModularProgramV2, ch: int, **kwargs) -> None:
        wav_cfg = self.waveform_cfg
        assert isinstance(wav_cfg, DragWaveformCfg)
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


class ArbWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: ArbWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        return self.waveform_cfg.length  # type: ignore[return-value]

    def create(self, prog: ModularProgramV2, ch: int, **kwargs) -> None:
        idata, qdata = self.make_iqdata(ch, prog, **kwargs)
        prog.add_envelope(ch, self.name, idata=idata, qdata=qdata)

    def make_iqdata(
        self, ch: int, prog: ModularProgramV2, even_length: bool = False
    ) -> tuple[NDArray, Optional[NDArray]]:
        # lazy import to avoid circular import
        from zcu_tools.meta_tool.arb_waveform import ArbWaveformDatabase

        cfg = self.waveform_cfg
        assert isinstance(cfg, ArbWaveformCfg)
        idata_raw, qdata_raw, time_raw = ArbWaveformDatabase.get(cfg.data)

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


class FlatTopWaveform(AbsWaveform):
    def __init__(self, name: str, waveform_cfg: FlatTopWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

        self.raise_waveform: AbsWaveform = waveform_cfg.raise_waveform.build(name)

    def create(self, prog: ModularProgramV2, ch: int, **kwargs) -> None:
        kwargs.setdefault("even_length", True)
        self.raise_waveform.create(prog, ch, **kwargs)

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {
            "style": "flat_top",
            "envelope": self.name,
            "length": self.length - self.raise_waveform.length,
        }
