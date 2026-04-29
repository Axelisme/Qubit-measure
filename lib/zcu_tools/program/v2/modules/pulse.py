from __future__ import annotations

from copy import deepcopy

from pydantic import BeforeValidator
from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Annotated, Any, Literal, Optional, Union

from .base import AbsModuleCfg, Module
from .util import round_timestamp
from .waveform import AbsWaveform, WaveformCfg, resolve_waveform_ref

if TYPE_CHECKING:
    from zcu_tools.program.v2.ir.builder import IRBuilder
    from zcu_tools.program.v2.modular import ModularProgramV2


class PulseCfg(AbsModuleCfg):
    type: Literal["pulse"] = "pulse"
    waveform: Annotated[WaveformCfg, BeforeValidator(resolve_waveform_ref)]
    ch: int
    nqz: Literal[1, 2]
    freq: Union[float, QickParam]
    phase: Union[float, QickParam] = 0.0
    gain: Union[float, QickParam]
    pre_delay: Union[float, QickParam] = 0.0
    post_delay: Union[float, QickParam] = 0.0

    mixer_freq: Optional[float] = None
    mux_freqs: Optional[list[float]] = None
    mux_gains: Optional[list[float]] = None
    mux_phases: Optional[list[float]] = None
    mask: Optional[list[int]] = None
    outsel: Optional[int] = None
    ro_ch: Optional[int] = None

    def build(self, name: str) -> Pulse:
        return Pulse(name, self)

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        if name == "length":
            self.waveform.set_param(name, value)
        elif name in {"gain", "freq", "phase"}:
            setattr(self, name, value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


class Pulse(Module):
    def __init__(
        self,
        name: str,
        cfg: Optional[PulseCfg],
        tag: Optional[str] = None,
        block_mode: bool = True,
        pulse_id: Optional[str] = None,
    ) -> None:
        self.name = name
        self.cfg = deepcopy(cfg) if cfg is not None else None
        self.tag = tag
        self.block_mode = block_mode
        self.pulse_id = pulse_id

    def init(self, prog: ModularProgramV2) -> None:
        self._prog = prog
        if self.cfg is None:
            return

        self.waveform: AbsWaveform = self.cfg.waveform.build(f"{self.name}_waveform")
        if self.pulse_id is None:
            self.pulse_id = prog.pulse_registry.calc_name(self.cfg)

            # auto reuse pulse
            if prog.pulse_registry.register(self.name, self.cfg):
                self.init_pulse(prog, self.pulse_id)
        else:  # if provided pulse_id, always init pulse (no reuse)
            self.init_pulse(prog, self.pulse_id)

    def init_pulse(self, prog: ModularProgramV2, pulse_id: str) -> None:
        cfg = self.cfg
        assert cfg is not None

        ro_ch = cfg.ro_ch if cfg.mixer_freq is not None else None
        prog.declare_gen(
            cfg.ch,
            nqz=cfg.nqz,
            mixer_freq=cfg.mixer_freq,
            mux_freqs=cfg.mux_freqs,
            mux_gains=cfg.mux_gains,
            mux_phases=cfg.mux_phases,
            ro_ch=ro_ch,
        )

        self.waveform.create(prog, cfg.ch)
        pulse_kwargs = dict[str, Any](freq=cfg.freq, phase=cfg.phase, gain=cfg.gain)
        if cfg.mask is not None:
            pulse_kwargs["mask"] = cfg.mask
        if cfg.outsel is not None:
            pulse_kwargs["outsel"] = cfg.outsel

        prog.add_pulse(
            cfg.ch,
            pulse_id,
            ro_ch=cfg.ro_ch,
            **pulse_kwargs,
            **self.waveform.to_wav_kwargs(),
        )

    def total_length(self, prog: ModularProgramV2) -> Union[float, QickParam]:
        if self.cfg is None:
            return 0.0
        return round_timestamp(
            prog,
            (
                round_timestamp(prog, self.cfg.pre_delay)
                + round_timestamp(prog, self.waveform.length, gen_ch=self.cfg.ch)
                + round_timestamp(prog, self.cfg.post_delay)
            ),
        )

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        cfg = self.cfg
        if cfg is None:
            return t

        prog.pulse(cfg.ch, self.pulse_id, t=t + cfg.pre_delay, tag=self.tag)
        if self.block_mode:
            return t + self.total_length(prog)
        return t

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
    ) -> Union[float, QickParam]:
        cfg = self.cfg
        if cfg is None or self.pulse_id is None:
            return t

        builder.ir_pulse(str(cfg.ch), self.pulse_id, t=t + cfg.pre_delay, tag=self.tag)

        if not self.block_mode:
            return t

        if hasattr(self, "_prog"):
            return t + self.total_length(self._prog)
        # unit-test path: prog not available, skip rounding
        return t + cfg.pre_delay + cfg.waveform.length + cfg.post_delay

    def allow_rerun(self) -> bool:
        return True
