from __future__ import annotations

import warnings
from abc import abstractmethod
from copy import deepcopy

from pydantic import BeforeValidator, Field, model_validator
from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Optional,
    TypeAlias,
    Union,
)

from .base import AbsModuleCfg, Module, resolve_module_ref
from .pulse import Pulse, PulseCfg
from .util import calc_max_length, merge_max_length, round_timestamp

if TYPE_CHECKING:
    from zcu_tools.program.v2.ir.builder import IRBuilder
    from zcu_tools.program.v2.modular import ModularProgramV2


class AbsReadoutCfg(AbsModuleCfg):
    @abstractmethod
    def build(self, name: str) -> AbsReadout: ...


class DirectReadoutCfg(AbsReadoutCfg):
    type: Literal["readout/direct"] = "readout/direct"
    ro_ch: int
    ro_length: Union[float, QickParam]
    ro_freq: Union[float, QickParam]
    trig_offset: Union[float, QickParam] = 0.0
    gen_ch: Optional[int] = None

    def build(self, name: str) -> DirectReadout:
        return DirectReadout(name, self)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if name == "ro_freq":
            self.ro_freq = value
        elif name == "ro_length":
            self.ro_length = value
        else:
            raise ValueError(f"Unknown parameter: {name}")


class PulseReadoutCfg(AbsReadoutCfg):
    type: Literal["readout/pulse"] = "readout/pulse"
    pulse_cfg: Annotated[PulseCfg, BeforeValidator(resolve_module_ref)]
    ro_cfg: DirectReadoutCfg

    @model_validator(mode="before")
    @classmethod
    def _autoderive_ro_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        pulse_cfg = data.get("pulse_cfg")
        ro_cfg = data.get("ro_cfg")
        if isinstance(pulse_cfg, dict) and isinstance(ro_cfg, dict):
            if "ch" in pulse_cfg:
                ro_cfg.setdefault("gen_ch", pulse_cfg["ch"])
            if "freq" in pulse_cfg:
                ro_cfg.setdefault("ro_freq", pulse_cfg["freq"])
        return data

    def build(self, name: str) -> PulseReadout:
        return PulseReadout(name, self)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if name == "gain":
            self.pulse_cfg.set_param("gain", value)
        elif name == "freq":
            self.pulse_cfg.set_param("freq", value)
            self.ro_cfg.set_param("ro_freq", value)
        elif name == "ro_freq":
            self.ro_cfg.set_param("ro_freq", value)
        elif name == "length":
            self.pulse_cfg.set_param("length", value)
        elif name == "ro_length":
            self.ro_cfg.set_param("ro_length", value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


ReadoutCfg: TypeAlias = Annotated[
    Union[DirectReadoutCfg, PulseReadoutCfg],
    Field(discriminator="type"),
]


class AbsReadout(Module):
    @abstractmethod
    def total_length(self, prog: ModularProgramV2) -> Union[float, QickParam]: ...

    def allow_rerun(self) -> bool:
        return True


def Readout(name: str, cfg: AbsReadoutCfg) -> AbsReadout:
    """Factory: dispatch a readout cfg to its concrete impl."""
    return cfg.build(name)


class DirectReadout(AbsReadout):
    def __init__(self, name: str, cfg: DirectReadoutCfg) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)

    def init(self, prog: ModularProgramV2) -> None:
        prog.declare_readout(ch=self.cfg.ro_ch, length=self.cfg.ro_length)

        readout_kwargs = dict()
        if self.cfg.gen_ch is not None:
            readout_kwargs["gen_ch"] = self.cfg.gen_ch
        prog.add_readoutconfig(
            ch=self.cfg.ro_ch,
            name=self.name,
            freq=self.cfg.ro_freq,
            **readout_kwargs,
        )

    def total_length(self, prog: ModularProgramV2) -> Union[float, QickParam]:
        return round_timestamp(
            prog,
            round_timestamp(prog, self.cfg.trig_offset)
            + round_timestamp(prog, self.cfg.ro_length, ro_ch=self.cfg.ro_ch),
        )

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        ro_ch = self.cfg.ro_ch
        trig_offset = self.cfg.trig_offset
        prog.send_readoutconfig(ro_ch, self.name, t=t)  # type: ignore
        prog.trigger([ro_ch], t=t + trig_offset)
        return t

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
    ) -> Union[float, QickParam]:
        ro_ch = str(self.cfg.ro_ch)
        builder.ir_readout(
            ch=ro_ch,
            ro_chs=(ro_ch,),
            pulse_name=self.name,
            t=t + self.cfg.trig_offset,
        )
        return t


class PulseReadout(AbsReadout):
    def __init__(self, name: str, cfg: PulseReadoutCfg) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)
        ro_ch = self.cfg.pulse_cfg.ro_ch
        if ro_ch is None:
            ro_ch = self.cfg.ro_cfg.ro_ch
            self.cfg.pulse_cfg.ro_ch = ro_ch
        if ro_ch != self.cfg.ro_cfg.ro_ch:
            warnings.warn(
                f"{name} pulse_cfg.ro_ch is {ro_ch}, this may not be what you want"
            )
        self.pulse = Pulse(name=f"{name}_pulse", cfg=self.cfg.pulse_cfg)
        self.ro_window = DirectReadout(name=f"{name}_adc", cfg=self.cfg.ro_cfg)

    def init(self, prog: ModularProgramV2) -> None:
        self.pulse.init(prog)
        self.ro_window.init(prog)

    def total_length(self, prog: ModularProgramV2) -> Union[float, QickParam]:
        return calc_max_length(
            self.ro_window.total_length(prog), self.pulse.total_length(prog)
        )

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        self.ro_window.run(prog, t)
        self.pulse.run(prog, t)
        return t + self.total_length(prog)

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
    ) -> Union[float, QickParam]:
        self.ro_window.ir_run(builder, t)
        self.pulse.ir_run(builder, t)

        prog = self.pulse._prog
        ro_end = t + self.ro_window.total_length(prog)
        pulse_end = t + self.pulse.total_length(prog)
        end_t = merge_max_length(ro_end, pulse_end)

        builder.ir_delay(end_t)
        builder.ir_delay_auto(0.0)
        return 0.0
