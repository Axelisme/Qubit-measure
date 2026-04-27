from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy

from pydantic import BeforeValidator, Field, ValidationInfo
from qick.asm_v2 import QickParam
from typing_extensions import (
    Annotated,
    Any,
    Literal,
    Optional,
    TypeAlias,
    Union,
    TYPE_CHECKING,
)

from .base import Module, ModuleCfg, get_ml_from_context
from .pulse import Pulse, PulseCfg
from .util import calc_max_length

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2


def _resolve_pulse_ref(value: Any, info: ValidationInfo) -> Any:
    if isinstance(value, str):
        ml = get_ml_from_context(info)
        if ml is None:
            raise ValueError(
                f"Cannot resolve pulse reference {value!r} without ModuleLibrary context"
            )
        return ml.get_module(value)
    return value


PulseOrRef: TypeAlias = Annotated[PulseCfg, BeforeValidator(_resolve_pulse_ref)]


class AbsResetCfg(ModuleCfg):
    @abstractmethod
    def build(self, name: str) -> AbsReset: ...


class NoneResetCfg(AbsResetCfg):
    type: Literal["reset/none"] = "reset/none"

    def build(self, name: str) -> NoneReset:
        return NoneReset(name, self)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        raise ValueError("NoneReset does not support set_param")


class PulseResetCfg(AbsResetCfg):
    type: Literal["reset/pulse"] = "reset/pulse"
    pulse_cfg: PulseOrRef

    def build(self, name: str) -> PulseReset:
        return PulseReset(name, self)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        self.pulse_cfg.set_param(name, value)


class TwoPulseResetCfg(AbsResetCfg):
    type: Literal["reset/two_pulse"] = "reset/two_pulse"
    pulse1_cfg: PulseOrRef
    pulse2_cfg: PulseOrRef

    def build(self, name: str) -> TwoPulseReset:
        return TwoPulseReset(name, self)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if name in ["gain1", "freq1"]:
            self.pulse1_cfg.set_param(name.replace("1", ""), value)
        elif name in ["gain2", "freq2"]:
            self.pulse2_cfg.set_param(name.replace("2", ""), value)
        elif name == "length":
            self.pulse1_cfg.set_param("length", value)
            self.pulse2_cfg.set_param("length", value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


class BathResetCfg(AbsResetCfg):
    type: Literal["reset/bath"] = "reset/bath"
    cavity_tone_cfg: PulseOrRef
    qubit_tone_cfg: PulseOrRef
    pi2_cfg: PulseOrRef

    def build(self, name: str) -> BathReset:
        return BathReset(name, self)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if name in ["qub_gain", "qub_freq"]:
            self.qubit_tone_cfg.set_param(name.replace("qub_", ""), value)
        elif name in ["res_gain", "res_freq"]:
            self.cavity_tone_cfg.set_param(name.replace("res_", ""), value)
        elif name == "qub_length":
            self.qubit_tone_cfg.set_param("length", value)
        elif name == "res_length":
            self.cavity_tone_cfg.set_param("length", value)
        elif name == "pi2_phase":
            self.pi2_cfg.set_param("phase", value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


ResetCfg: TypeAlias = Annotated[
    Union[NoneResetCfg, PulseResetCfg, TwoPulseResetCfg, BathResetCfg],
    Field(discriminator="type"),
]


class AbsReset(Module):
    @abstractmethod
    def total_length(self, prog: ModularProgramV2) -> Union[float, QickParam]: ...

    def allow_rerun(self) -> bool:
        return True


def Reset(name: str, cfg: Optional[AbsResetCfg]) -> AbsReset:
    """Factory: dispatch a reset cfg to its concrete impl. None → NoneReset."""
    if cfg is None:
        cfg = NoneResetCfg(desc="Auto derived from None")
    return cfg.build(name)


class NoneReset(AbsReset):
    def __init__(self, name: str, cfg: NoneResetCfg) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)

    def init(self, prog: ModularProgramV2) -> None: ...

    def total_length(self, _prog: ModularProgramV2) -> Union[float, QickParam]:
        return 0.0

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return t


class PulseReset(AbsReset):
    def __init__(self, name: str, cfg: PulseResetCfg) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)
        self.reset_pulse = Pulse(name=f"{name}_pulse", cfg=self.cfg.pulse_cfg)

    def init(self, prog: ModularProgramV2) -> None:
        self.reset_pulse.init(prog)

    def total_length(self, prog: ModularProgramV2) -> Union[float, QickParam]:
        return self.reset_pulse.total_length(prog)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.reset_pulse.run(prog, t)


class TwoPulseReset(AbsReset):
    def __init__(self, name: str, cfg: TwoPulseResetCfg) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)
        self.reset_pulse1 = Pulse(name=f"{name}_pulse1", cfg=self.cfg.pulse1_cfg)
        self.reset_pulse2 = Pulse(name=f"{name}_pulse2", cfg=self.cfg.pulse2_cfg)

    def init(self, prog: ModularProgramV2) -> None:
        self.reset_pulse1.init(prog)
        self.reset_pulse2.init(prog)

    def total_length(self, prog: ModularProgramV2) -> Union[float, QickParam]:
        return calc_max_length(
            self.reset_pulse1.total_length(prog), self.reset_pulse2.total_length(prog)
        )

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        pulse1_t = self.reset_pulse1.run(prog, t)
        pulse2_t = self.reset_pulse2.run(prog, t)
        return calc_max_length(pulse1_t, pulse2_t)


class BathReset(AbsReset):
    def __init__(self, name: str, cfg: BathResetCfg) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)
        self.res_pulse = Pulse(name=f"{name}_res_pulse", cfg=self.cfg.cavity_tone_cfg)
        self.qub_pulse = Pulse(name=f"{name}_qub_pulse", cfg=self.cfg.qubit_tone_cfg)
        self.pi2_pulse = Pulse(name=f"{name}_pi2_pulse", cfg=self.cfg.pi2_cfg)

    def init(self, prog: ModularProgramV2) -> None:
        self.res_pulse.init(prog)
        self.qub_pulse.init(prog)
        self.pi2_pulse.init(prog)

    def total_length(self, prog: ModularProgramV2) -> Union[float, QickParam]:
        return self.res_pulse.total_length(prog) + self.pi2_pulse.total_length(prog)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        res_t = self.res_pulse.run(prog, t)
        self.qub_pulse.run(prog, t)
        end_t = self.pi2_pulse.run(prog, res_t)
        return end_t
