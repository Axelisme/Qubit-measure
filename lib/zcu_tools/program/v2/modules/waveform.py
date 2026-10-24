from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NotRequired,
    TypeAlias,
    TypedDict,
)

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BeforeValidator,
    Field,
    TypeAdapter,
    ValidationInfo,
    field_validator,
)
from qick.asm_v2 import QickParam

from zcu_tools import cfg_model as core_cfgs
from zcu_tools.cfg_model import CfgValidationContext, ConfigBase
from zcu_tools.cfg_model import WaveformCfg as CoreWaveformCfg
from zcu_tools.core.waveform_format import decode_npz

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2
    from zcu_tools.resources.context import LibrarySnapshot


class AbsWaveformCfg(ConfigBase, ABC):
    style: str

    @abstractmethod
    def build(self, name: str) -> AbsWaveform:
        """Build the runtime waveform represented by this cfg."""

    @abstractmethod
    def set_param(self, name: str, value: float | QickParam) -> None:
        """Update a sweep-tunable parameter by name."""


class ConstWaveformCfg(AbsWaveformCfg):
    style: Literal["const"] = "const"
    length: float | QickParam

    def build(self, name: str) -> ConstWaveform:
        return ConstWaveform(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
        if name == "length":
            self.length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


class CosineWaveformCfg(AbsWaveformCfg):
    style: Literal["cosine"] = "cosine"
    length: float

    def build(self, name: str) -> CosineWaveform:
        return CosineWaveform(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
        if name != "length":
            raise ValueError(f"Unknown parameter: {name}")
        if isinstance(value, QickParam):
            raise ValueError("Cosine waveform length must be a float value")
        self.length = value


class GaussWaveformCfg(AbsWaveformCfg):
    style: Literal["gauss"] = "gauss"
    length: float
    sigma: float

    def build(self, name: str) -> GaussWaveform:
        return GaussWaveform(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
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


class DragWaveformCfg(AbsWaveformCfg):
    style: Literal["drag"] = "drag"
    length: float
    sigma: float
    delta: float
    alpha: float

    def build(self, name: str) -> DragWaveform:
        return DragWaveform(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
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


class ArbWaveformCfg(AbsWaveformCfg):
    """Native arbitrary envelope with an explicit local playback path.

    style is arb. data is the logical asset key, retained for diagnostics.
    path is the required host-absolute NPZ location, serialized with POSIX
    separators; it is the sole playback authority, not a repository key lookup.
    Relative/foreign paths raise ValueError during validation. Reads propagate
    file/format failures and read current bytes; cfg does not freeze samples.
    """

    style: Literal["arb"] = "arb"
    data: str
    path: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        path = Path(value)
        if not value or not path.is_absolute():
            raise ValueError("Arb playback path must be an absolute local path")
        return path.as_posix()

    @property
    def length(self) -> float:
        """Read path-derived time[-1] in µs; propagate I/O/format failures."""
        return float(decode_npz(Path(self.path).read_bytes()).samples.time[-1])

    def build(self, name: str) -> ArbWaveform:
        return ArbWaveform(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
        raise ValueError(
            f"Arb waveform length is asset-derived; data cannot be changed through set_param: {name}={value}"
        )


def resolve_waveform_ref(value: Any, info: ValidationInfo) -> Any:
    if isinstance(value, str):
        if info.context is None:
            raise ValueError("ModuleLibrary context not found")
        library: LibrarySnapshot = info.context["ml"]
        return library.get_waveform(value)
    return value


RaiseWaveformCfg: TypeAlias = Annotated[
    CosineWaveformCfg | GaussWaveformCfg | DragWaveformCfg | ArbWaveformCfg,
    BeforeValidator(resolve_waveform_ref),
    Field(discriminator="style"),
]


class FlatTopWaveformCfg(AbsWaveformCfg):
    style: Literal["flat_top"] = "flat_top"
    length: float | QickParam
    raise_waveform: RaiseWaveformCfg

    def build(self, name: str) -> FlatTopWaveform:
        return FlatTopWaveform(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
        if name == "length":
            self.length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


WaveformCfg: TypeAlias = Annotated[
    ConstWaveformCfg
    | CosineWaveformCfg
    | GaussWaveformCfg
    | DragWaveformCfg
    | ArbWaveformCfg
    | FlatTopWaveformCfg,
    BeforeValidator(resolve_waveform_ref),
    Field(discriminator="style"),
]


class WaveformCfgFactory:
    @staticmethod
    def from_core(
        core_cfg: CoreWaveformCfg,
        *,
        context: CfgValidationContext | None,
    ) -> WaveformCfg:
        """Convert a materialized core envelope to native cfg without mutation.

        context supplies capture-bound asset paths for arb, including a
        flat-top raise envelope. Arb without context/assets raises ValueError.
        Resolve keys through assets.path_for and propagate key/missing/I/O
        errors. Native cfg holds only key/path values, never the capability.
        Non-arb conversion needs no assets. Do not reinterpret formula recipes.
        """
        if isinstance(core_cfg, core_cfgs.ConstWaveformCfg):
            return ConstWaveformCfg.model_validate(core_cfg.to_dict())
        if isinstance(core_cfg, core_cfgs.FlatTopWaveformCfg):
            return FlatTopWaveformCfg(
                length=core_cfg.length,
                raise_waveform=_raise_from_core(
                    core_cfg.raise_waveform, context=context
                ),
            )
        return _raise_from_core(core_cfg, context=context)

    @staticmethod
    def from_raw(
        raw: str | dict[str, Any] | AbsWaveformCfg,
        *,
        ml: LibrarySnapshot | None = None,
    ) -> WaveformCfg:
        if isinstance(raw, str):
            if ml is None:
                raise ValueError(
                    f"Got waveform name '{raw}' but ModuleLibrary is not provided"
                )
            resolved_raw = ml.get_waveform(raw)
        else:
            resolved_raw = raw
        return TypeAdapter(WaveformCfg).validate_python(
            resolved_raw, context={"ml": ml}
        )


def _raise_from_core(
    core_cfg: core_cfgs.RaiseWaveformCfg,
    *,
    context: CfgValidationContext | None,
) -> RaiseWaveformCfg:
    """Convert a non-flat core envelope, resolving arb paths only at this seam."""
    if isinstance(core_cfg, core_cfgs.ArbWaveformCfg):
        if context is None or context.assets is None:
            raise ValueError("Arb waveform conversion requires context.assets")
        return ArbWaveformCfg(
            data=core_cfg.data,
            path=context.assets.path_for(core_cfg.data).as_posix(),
        )
    return TypeAdapter(RaiseWaveformCfg).validate_python(core_cfg.to_dict())


class QickWaveformKwargs(TypedDict):
    style: Literal["const", "arb", "flat_top"]

    length: NotRequired[float | QickParam]
    envelope: NotRequired[str]


class AbsWaveform(ABC):
    name: str
    waveform_cfg: WaveformCfg

    @property
    def length(self) -> float | QickParam:
        return self.waveform_cfg.length

    @abstractmethod
    def create(self, prog: ModularProgramV2, ch: int, **kwargs) -> None: ...

    @abstractmethod
    def to_wav_kwargs(self) -> QickWaveformKwargs: ...


class ConstWaveform(AbsWaveform):
    """Build a native envelope from settings declared as ``ConstWaveform.Cfg``."""

    Cfg: TypeAlias = ConstWaveformCfg

    def __init__(self, name: str, waveform_cfg: ConstWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    def create(self, prog: ModularProgramV2, ch: int, **kwargs) -> None: ...

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {"style": "const", "length": self.length}


class CosineWaveform(AbsWaveform):
    """Build a native envelope from settings declared as ``CosineWaveform.Cfg``."""

    Cfg: TypeAlias = CosineWaveformCfg

    def __init__(self, name: str, waveform_cfg: CosineWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        cfg = self.waveform_cfg
        assert isinstance(cfg, CosineWaveformCfg)
        return cfg.length

    def create(self, prog: ModularProgramV2, ch: int, **kwargs) -> None:
        prog.add_cosine(ch, self.name, length=self.length, **kwargs)

    def to_wav_kwargs(self) -> QickWaveformKwargs:
        return {"style": "arb", "envelope": self.name}


class GaussWaveform(AbsWaveform):
    """Build a native envelope from settings declared as ``GaussWaveform.Cfg``."""

    Cfg: TypeAlias = GaussWaveformCfg

    def __init__(self, name: str, waveform_cfg: GaussWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        cfg = self.waveform_cfg
        assert isinstance(cfg, GaussWaveformCfg)
        return cfg.length

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
    """Build a native envelope from settings declared as ``DragWaveform.Cfg``."""

    Cfg: TypeAlias = DragWaveformCfg

    def __init__(self, name: str, waveform_cfg: DragWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        cfg = self.waveform_cfg
        assert isinstance(cfg, DragWaveformCfg)
        return cfg.length

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
    """Build a native envelope from settings declared as ``ArbWaveform.Cfg``."""

    Cfg: TypeAlias = ArbWaveformCfg

    def __init__(self, name: str, waveform_cfg: ArbWaveformCfg) -> None:
        self.name = name
        self.waveform_cfg = waveform_cfg

    @property
    def length(self) -> float:
        cfg = self.waveform_cfg
        assert isinstance(cfg, ArbWaveformCfg)
        return cfg.length

    def create(self, prog: ModularProgramV2, ch: int, **kwargs) -> None:
        idata, qdata = self.make_iqdata(ch, prog, **kwargs)
        prog.add_envelope(ch, self.name, idata=idata, qdata=qdata)

    def make_iqdata(
        self, ch: int, prog: ModularProgramV2, *, even_length: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Read cfg.path samples and interpolate at the channel's timing grid.

        ch selects prog.soccfg generator scale and samples per clock. Scale
        normalized samples by get_maxv(ch) and return I/Q arrays; zero Q is
        None. even_length rounds to an even number of clocks for flat-top ramps.
        Use the full reference duration, never stretch/truncate by another cfg
        length. File/codec failures propagate; no repository lookup occurs.
        """
        cfg = self.waveform_cfg
        assert isinstance(cfg, ArbWaveformCfg)
        samples = decode_npz(Path(cfg.path).read_bytes()).samples
        idata_raw, qdata_raw, time_raw = samples.idata, samples.qdata, samples.time

        maxv = prog.soccfg.get_maxv(ch)
        samps_per_clk = prog.soccfg["gens"][ch]["samps_per_clk"]
        raw_end = float(time_raw[-1])
        length = raw_end

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
    """Build a native envelope from settings declared as ``FlatTopWaveform.Cfg``."""

    Cfg: TypeAlias = FlatTopWaveformCfg

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
