from __future__ import annotations

import math
from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BeforeValidator
from qick.asm_v2 import QickParam

from .base import AbsModuleCfg, Module
from .delay import DelayAuto
from .dmem import LoadValue
from .util import round_timestamp
from .waveform import (
    AbsWaveform,
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    WaveformCfg,
    resolve_waveform_ref,
)

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2


class PulseCfg(AbsModuleCfg):
    type: Literal["pulse"] = "pulse"
    waveform: Annotated[WaveformCfg, BeforeValidator(resolve_waveform_ref)]
    ch: int
    nqz: Literal[1, 2]
    freq: float | QickParam
    phase: float | QickParam = 0.0
    gain: float | QickParam
    pre_delay: float | QickParam = 0.0
    post_delay: float | QickParam = 0.0

    mixer_freq: float | None = None
    mux_freqs: list[float] | None = None
    mux_gains: list[float] | None = None
    mux_phases: list[float] | None = None
    mask: list[int] | None = None
    outsel: int | None = None
    ro_ch: int | None = None

    def build(self, name: str) -> Pulse:
        return Pulse(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
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
        cfg: PulseCfg | None,
        tag: str | None = None,
        block_mode: bool = True,
        pulse_id: str | None = None,
    ) -> None:
        self.name = name
        self.cfg = deepcopy(cfg) if cfg is not None else None
        self.tag = tag
        self.block_mode = block_mode
        self.pulse_id = pulse_id

    def init(self, prog: ModularProgramV2) -> None:
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

    def total_length(self, prog: ModularProgramV2) -> float | QickParam:
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
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        cfg = self.cfg
        if cfg is None:
            return t

        prog.pulse(cfg.ch, self.pulse_id, t=t + cfg.pre_delay, tag=self.tag)
        if self.block_mode:
            return t + self.total_length(prog)
        return t

    def allow_rerun(self) -> bool:
        return True


class _RuntimeLengthPulse(Module):
    def __init__(
        self,
        name: str,
        *,
        template: Pulse,
        length_reg: str,
        flat_top: bool,
    ) -> None:
        self.name = name
        self.template = template
        self.length_reg = length_reg
        self.flat_top = flat_top

    def init(self, prog: ModularProgramV2) -> None:
        pass

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        cfg = self.template.cfg
        assert cfg is not None
        assert self.template.pulse_id is not None
        prog.pulse_from_length_reg(
            cfg.ch,
            self.template.pulse_id,
            length_reg=self.length_reg,
            flat_top=self.flat_top,
            t=t + cfg.pre_delay,
        )
        return t

    def allow_rerun(self) -> bool:
        return True


class TableLengthPulse(Module):
    """Sweep const/flat-top pulse length from dmem with one wmem template.

    The hardware loop counter selects generator-length and tProcessor-duration
    tables. Every entry must be strictly positive and satisfy the generator's
    three-cycle minimum. Readout alignment uses the
    runtime duration table, so shorter pulses are not padded to the longest point.
    """

    def __init__(
        self,
        name: str,
        cfg: PulseCfg,
        *,
        lengths: Iterable[float],
        idx_reg: str,
    ) -> None:
        self.name = name
        self.cfg = deepcopy(cfg)
        self.lengths = [float(length) for length in lengths]
        self.idx_reg = idx_reg

        if not self.lengths:
            raise ValueError("TableLengthPulse requires at least one length")
        if any(not math.isfinite(length) or length <= 0.0 for length in self.lengths):
            raise ValueError("TableLengthPulse lengths must be finite and > 0")
        if not isinstance(self.cfg.waveform, (ConstWaveformCfg, FlatTopWaveformCfg)):
            raise NotImplementedError(
                "TableLengthPulse supports only const and flat_top waveforms"
            )

        prefix = f"{name}_table_length"
        self._length_reg = f"{prefix}_gen_cycles"
        self._duration_reg = f"{prefix}_tproc_cycles"
        self._template = Pulse(f"{name}_template", self.cfg)
        self._runtime_pulse = _RuntimeLengthPulse(
            f"{name}_runtime",
            template=self._template,
            length_reg=self._length_reg,
            flat_top=isinstance(self.cfg.waveform, FlatTopWaveformCfg),
        )

    def _wave_length_cycles(self, prog: ModularProgramV2, length: float) -> int:
        wave_length = length
        if isinstance(self.cfg.waveform, FlatTopWaveformCfg):
            wave_length -= float(self.cfg.waveform.raise_waveform.length)
            if wave_length <= 0.0:
                raise ValueError(
                    f"TableLengthPulse flat_top length {length} us must exceed "
                    f"the fixed ramp length {self.cfg.waveform.raise_waveform.length} us"
                )

        cycles = int(prog.us2cycles(gen_ch=self.cfg.ch, us=wave_length))
        if cycles < 3:
            raise ValueError(
                f"TableLengthPulse non-zero length {length} us resolves to "
                f"{cycles} generator cycles; hardware requires at least 3"
            )
        if cycles >= 2**16:
            raise ValueError(
                f"TableLengthPulse length {length} us resolves to {cycles} "
                "generator cycles; hardware limit is less than 2**16"
            )
        return cycles

    def _duration_cycles(self, prog: ModularProgramV2, length: float) -> int:
        pre = round_timestamp(prog, self.cfg.pre_delay)
        wave = round_timestamp(prog, length, gen_ch=self.cfg.ch)
        post = round_timestamp(prog, self.cfg.post_delay)
        total = round_timestamp(prog, pre + wave + post)
        if isinstance(total, QickParam):
            raise ValueError("TableLengthPulse pre_delay/post_delay must not be swept")
        return int(prog.us2cycles(us=float(total)))

    def init(self, prog: ModularProgramV2) -> None:
        wave_cycles = [
            self._wave_length_cycles(prog, length) for length in self.lengths
        ]
        duration_cycles = [
            self._duration_cycles(prog, length) for length in self.lengths
        ]
        template_length = self.lengths[0]
        template_cfg = self._template.cfg
        assert template_cfg is not None
        template_cfg.set_param("length", template_length)
        self._template.init(prog)

        self._length_loader = LoadValue(
            f"{self.name}_load_gen_length",
            wave_cycles,
            idx_reg=self.idx_reg,
            val_reg=self._length_reg,
        )
        self._duration_loader = LoadValue(
            f"{self.name}_load_duration",
            duration_cycles,
            idx_reg=self.idx_reg,
            val_reg=self._duration_reg,
        )
        for module in (
            self._length_loader,
            self._duration_loader,
            self._runtime_pulse,
        ):
            module.init(prog)

        self._wait = DelayAuto(f"{self.name}_runtime_duration", t=self._duration_reg)
        self._wait.init(prog)

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        cur_t = t
        for module in (
            self._length_loader,
            self._duration_loader,
            self._runtime_pulse,
            self._wait,
        ):
            cur_t = module.run(prog, cur_t)
        return cur_t

    def allow_rerun(self) -> bool:
        return True
