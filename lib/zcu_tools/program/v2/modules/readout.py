from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

from pydantic import BeforeValidator, Field, model_validator
from qick.asm_v2 import QickParam

from ..macro.wmem import PatchWmemFromDmem
from .base import AbsModuleCfg, Module, resolve_module_ref
from .dmem import LoadWord
from .pulse import Pulse, PulseCfg
from .util import calc_max_length, round_timestamp

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2


class AbsReadoutCfg(AbsModuleCfg):
    @abstractmethod
    def build(self, name: str) -> AbsReadout: ...


class DirectReadoutCfg(AbsReadoutCfg):
    type: Literal["readout/direct"] = "readout/direct"
    ro_ch: int
    ro_length: float | QickParam
    ro_freq: float | QickParam
    trig_offset: float | QickParam = 0.0
    gen_ch: int | None = None

    def build(self, name: str) -> DirectReadout:
        return DirectReadout(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
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

    def set_param(self, name: str, value: float | QickParam) -> None:
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
    DirectReadoutCfg | PulseReadoutCfg,
    Field(discriminator="type"),
]


class AbsReadout(Module):
    @abstractmethod
    def total_length(self, prog: ModularProgramV2) -> float | QickParam: ...

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

    def total_length(self, prog: ModularProgramV2) -> float | QickParam:
        return round_timestamp(
            prog,
            round_timestamp(prog, self.cfg.trig_offset)
            + round_timestamp(prog, self.cfg.ro_length, ro_ch=self.cfg.ro_ch),
        )

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        ro_ch = self.cfg.ro_ch
        trig_offset = self.cfg.trig_offset
        prog.send_readoutconfig(ro_ch, self.name, t=t)  # type: ignore
        prog.trigger([ro_ch], t=t + trig_offset)
        return t


class PulseReadout(AbsReadout):
    def __init__(
        self,
        name: str,
        cfg: PulseReadoutCfg,
        *,
        gain_val: str | None = None,
        freq_val: str | None = None,
        ro_freq_val: str | None = None,
    ) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)
        self.gain_val = gain_val
        self.freq_val = freq_val
        self.ro_freq_val = ro_freq_val
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
        self._runtime_pulse: Pulse | None = None
        self._runtime_ro_name: str | None = None

    def init(self, prog: ModularProgramV2) -> None:
        self.pulse.init(prog)
        self.ro_window.init(prog)
        if self._uses_runtime_pulse:
            self._runtime_pulse = Pulse(
                name=f"{self.name}_runtime_pulse",
                cfg=self.cfg.pulse_cfg,
                pulse_id=f"{self.name}_runtime_pulse",
            )
            self._runtime_pulse.init(prog)
        if self.ro_freq_val is not None:
            self._runtime_ro_name = f"{self.name}_adc_runtime"
            readout_kwargs: dict[str, int] = {}
            if self.cfg.ro_cfg.gen_ch is not None:
                readout_kwargs["gen_ch"] = self.cfg.ro_cfg.gen_ch
            prog.add_readoutconfig(
                ch=self.cfg.ro_cfg.ro_ch,
                name=self._runtime_ro_name,
                freq=self.cfg.ro_cfg.ro_freq,
                **readout_kwargs,
            )

    def total_length(self, prog: ModularProgramV2) -> float | QickParam:
        return calc_max_length(
            self.ro_window.total_length(prog), self.pulse.total_length(prog)
        )

    @property
    def _uses_runtime_pulse(self) -> bool:
        return self.gain_val is not None or self.freq_val is not None

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        if self.ro_freq_val is None:
            self.ro_window.run(prog, t)
        else:
            runtime_ro_name = self._runtime_ro_name
            assert runtime_ro_name is not None
            self._run_readout_config_from_regs(prog, runtime_ro_name, t)

        if not self._uses_runtime_pulse:
            self.pulse.run(prog, t)
        else:
            runtime_pulse = self._runtime_pulse
            assert runtime_pulse is not None
            pulse_id = runtime_pulse.pulse_id
            assert pulse_id is not None
            pulse_cfg = runtime_pulse.cfg
            assert pulse_cfg is not None
            prog.pulse_from_regs(
                pulse_cfg.ch,
                pulse_id,
                t=t + pulse_cfg.pre_delay,
                tag=runtime_pulse.tag,
                freq_reg=self.freq_val,
                gain_reg=self.gain_val,
            )
        return t + self.total_length(prog)

    def _run_readout_config_from_regs(
        self, prog: ModularProgramV2, name: str, t: float | QickParam
    ) -> None:
        ro_ch = self.cfg.ro_cfg.ro_ch
        prog.send_readoutconfig_from_regs(ro_ch, name, t=t, freq_reg=self.ro_freq_val)
        prog.trigger([ro_ch], t=t + self.cfg.ro_cfg.trig_offset)


class TablePulseReadout(AbsReadout):
    """Sweep a pulse readout through exact dmem frequency-word tables.

    The current point is played through QICK's ordinary wmem-backed pulse and
    readout-config path. At the sweep loop's exec-after hook, the next point is
    loaded from rotated dmem tables and persisted into dedicated templates. The
    final table entry wraps back to the first point for the next outer repetition.
    """

    def __init__(
        self,
        name: str,
        cfg: PulseReadoutCfg,
        *,
        idx_reg: str,
        freq_words: Sequence[int],
        ro_freq_words: Sequence[int],
    ) -> None:
        self.name = name
        self.idx_reg = idx_reg
        self.freq_words = [int(word) for word in freq_words]
        self.ro_freq_words = [int(word) for word in ro_freq_words]
        if len(self.freq_words) == 0:
            raise ValueError("TablePulseReadout requires at least one frequency word")
        if len(self.freq_words) != len(self.ro_freq_words):
            raise ValueError(
                "TablePulseReadout generator/readout tables must have equal length"
            )

        self._readout = PulseReadout(name, cfg)
        self.cfg = self._readout.cfg
        self._readout.pulse = Pulse(
            name=f"{name}_pulse",
            cfg=self.cfg.pulse_cfg,
            pulse_id=f"{name}_table_pulse",
        )
        self._addr_reg = f"{name}_table_addr"

        self._freq_loader = LoadWord(
            name=f"{name}_next_freq",
            values=self.freq_words[1:] + self.freq_words[:1],
            idx_reg=idx_reg,
            val_reg=f"{name}_next_freq_word",
        )
        self._ro_freq_loader = LoadWord(
            name=f"{name}_next_ro_freq",
            values=self.ro_freq_words[1:] + self.ro_freq_words[:1],
            idx_reg=idx_reg,
            val_reg=f"{name}_next_ro_freq_word",
        )

    def init(self, prog: ModularProgramV2) -> None:
        self._validate_loop_count(prog)
        self._readout.init(prog)
        self._freq_loader.init(prog)
        self._ro_freq_loader.init(prog)
        prog.add_reg(self._addr_reg)
        pulse_id = self._readout.pulse.pulse_id
        assert pulse_id is not None
        prog.append_loop_after(
            self.idx_reg,
            PatchWmemFromDmem(
                name=pulse_id,
                idx_reg=self.idx_reg,
                addr_reg=self._addr_reg,
                val_reg=self._freq_loader.val_reg,
                dmem_offset=self._freq_loader.offset,
            ),
            PatchWmemFromDmem(
                name=self._readout.ro_window.name,
                idx_reg=self.idx_reg,
                addr_reg=self._addr_reg,
                val_reg=self._ro_freq_loader.val_reg,
                dmem_offset=self._ro_freq_loader.offset,
            ),
        )

    def total_length(self, prog: ModularProgramV2) -> float | QickParam:
        return self._readout.total_length(prog)

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        return self._readout.run(prog, t)

    def allow_rerun(self) -> bool:
        return True

    def _validate_loop_count(self, prog: ModularProgramV2) -> None:
        sweep_dict = prog.sweep_dict
        if sweep_dict is None:
            raise ValueError(
                f"TablePulseReadout idx_reg={self.idx_reg!r} requires a sweep loop"
            )
        matches = [spec for name, spec in sweep_dict if name == self.idx_reg]
        if len(matches) != 1:
            raise ValueError(
                f"TablePulseReadout idx_reg={self.idx_reg!r} must match exactly "
                "one sweep loop"
            )
        spec = matches[0]
        count = spec if isinstance(spec, int) else spec.expts
        if count != len(self.freq_words):
            raise ValueError(
                f"TablePulseReadout table length {len(self.freq_words)} does not "
                f"match loop {self.idx_reg!r} count {count}"
            )
