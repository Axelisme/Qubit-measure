from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy

from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Self,
    TypeAlias,
    Union,
)

from .base import Module, ModuleCfg
from .pulse import Pulse, PulseCfg
from .util import calc_max_length

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.program.v2.modular import ModularProgramV2


@ModuleCfg.bind_handler
class NoneResetCfg(ModuleCfg):
    type: Literal["reset/none"] = "reset/none"

    @classmethod
    def from_dict(cls, raw_cfg: dict[str, Any], ml: "ModuleLibrary") -> Self:
        return cls.model_validate(raw_cfg)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        raise ValueError("NoneReset does not support set_param")


@ModuleCfg.bind_handler
class PulseResetCfg(ModuleCfg):
    type: Literal["reset/pulse"] = "reset/pulse"
    pulse_cfg: PulseCfg

    @classmethod
    def from_dict(cls, raw_cfg: dict[str, Any], ml: "ModuleLibrary") -> Self:
        raw_cfg = deepcopy(raw_cfg)

        pulse_cfg = raw_cfg.get("pulse_cfg")
        if isinstance(pulse_cfg, str):
            raw_cfg["pulse_cfg"] = ml.get_module(pulse_cfg)

        return cls.model_validate(raw_cfg)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        self.pulse_cfg.set_param(name, value)


@ModuleCfg.bind_handler
class TwoPulseResetCfg(ModuleCfg):
    type: Literal["reset/two_pulse"] = "reset/two_pulse"
    pulse1_cfg: PulseCfg
    pulse2_cfg: PulseCfg

    @classmethod
    def from_dict(cls, raw_cfg: dict[str, Any], ml: "ModuleLibrary") -> Self:
        raw_cfg = deepcopy(raw_cfg)

        pulse1_cfg = raw_cfg.get("pulse1_cfg")
        if isinstance(pulse1_cfg, str):
            raw_cfg["pulse1_cfg"] = ml.get_module(pulse1_cfg)
        pulse2_cfg = raw_cfg.get("pulse2_cfg")
        if isinstance(pulse2_cfg, str):
            raw_cfg["pulse2_cfg"] = ml.get_module(pulse2_cfg)

        return cls.model_validate(raw_cfg)

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


@ModuleCfg.bind_handler
class BathResetCfg(ModuleCfg):
    type: Literal["reset/bath"] = "reset/bath"
    cavity_tone_cfg: PulseCfg
    qubit_tone_cfg: PulseCfg
    pi2_cfg: PulseCfg

    @classmethod
    def from_dict(cls, raw_cfg: dict[str, Any], ml: "ModuleLibrary") -> Self:
        raw_cfg = deepcopy(raw_cfg)

        cavity_tone_cfg = raw_cfg.get("cavity_tone_cfg")
        if isinstance(cavity_tone_cfg, str):
            raw_cfg["cavity_tone_cfg"] = ml.get_module(cavity_tone_cfg)
        qubit_tone_cfg = raw_cfg.get("qubit_tone_cfg")
        if isinstance(qubit_tone_cfg, str):
            raw_cfg["qubit_tone_cfg"] = ml.get_module(qubit_tone_cfg)
        pi2_cfg = raw_cfg.get("pi2_cfg")
        if isinstance(pi2_cfg, str):
            raw_cfg["pi2_cfg"] = ml.get_module(pi2_cfg)

        return cls.model_validate(raw_cfg)

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


ResetCfg: TypeAlias = Union[NoneResetCfg, PulseResetCfg, TwoPulseResetCfg, BathResetCfg]


class AbsReset(Module):
    @abstractmethod
    def total_length(self, prog: ModularProgramV2) -> Union[float, QickParam]: ...


class Reset(AbsReset):
    _supported_reset: ClassVar[dict[str, type["AbsReset"]]] = {}

    def __init__(self, name: str, cfg: Optional[ResetCfg]) -> None:
        if cfg is None:
            cfg = NoneResetCfg(desc="Auto derived from None")
        cfg_type = cfg.type
        if cfg_type not in self._supported_reset:
            raise ValueError(f"Unknown reset type: {cfg_type}")
        self.reset = self._supported_reset[cfg_type](name, cfg)

    @classmethod
    def bind_reset(cls, id_name: str) -> Callable[[type["AbsReset"]], type["AbsReset"]]:
        if id_name in cls._supported_reset:
            raise ValueError(
                f"Reset {id_name} already registered by {cls._supported_reset[id_name].__name__}"
            )

        def decorator(sub_cls: type["AbsReset"]) -> type["AbsReset"]:
            cls._supported_reset[id_name] = sub_cls
            return sub_cls

        return decorator

    @property
    def name(self) -> str:
        return self.reset.name

    def init(self, prog: ModularProgramV2) -> None:
        self.reset.init(prog)

    def total_length(self, prog: ModularProgramV2) -> Union[float, QickParam]:
        return self.reset.total_length(prog)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        return self.reset.run(prog, t)


@Reset.bind_reset(NoneResetCfg.module_type())
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


@Reset.bind_reset(PulseResetCfg.module_type())
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


@Reset.bind_reset(TwoPulseResetCfg.module_type())
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


@Reset.bind_reset(BathResetCfg.module_type())
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
