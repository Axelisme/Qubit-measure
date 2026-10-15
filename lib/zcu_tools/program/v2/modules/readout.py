from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

from pydantic import BeforeValidator
from qick.asm_v2 import QickParam

from ..macro.wmem import PatchWmemFromDmem
from .base import AbsModuleCfg, Module
from .dmem import LoadWord
from .pulse import Pulse, PulseCfg
from .util import calc_max_length, round_timestamp
from .waveform import WaveformCfg, resolve_waveform_ref

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2


class AbsReadoutCfg(AbsModuleCfg):
    @abstractmethod
    def build(self, name: str) -> AbsReadout: ...


class PulseReadoutCfg(AbsReadoutCfg):
    """Native generator/ADC readout, with one shared semantic frequency.

    gen_ch/ro_ch select generator/ADC channels; nqz is Nyquist zone 1 or 2.
    freq/mixer_freq are MHz, gain is normalized, phase is degrees. waveform is
    an expanded native envelope (arb holds its absolute asset path). pre_delay
    and post_delay bracket generator playback in us; ro_length is the ADC window
    and trig_offset the trigger delay in us. Scalar knobs except mixer_freq accept
    QICK sweeps. desc is optional prose. type is readout/pulse.
    """

    type: Literal["readout/pulse"] = "readout/pulse"
    gen_ch: int
    nqz: Literal[1, 2]
    freq: float | QickParam
    gain: float | QickParam
    phase: float | QickParam = 0.0
    waveform: Annotated[WaveformCfg, BeforeValidator(resolve_waveform_ref)]
    mixer_freq: float | None = None
    pre_delay: float | QickParam = 0.0
    post_delay: float | QickParam = 0.0
    ro_ch: int
    ro_length: float | QickParam
    trig_offset: float | QickParam = 0.0

    def build(self, name: str) -> PulseReadout:
        """Capture this cfg as a named, blocking QICK readout module."""
        return PulseReadout(name, self)

    def set_param(self, name: str, value: float | QickParam) -> None:
        """Set shared freq (MHz), gain, waveform length or ro_length (us).

        Unsupported names raise ValueError; waveform length restrictions apply.
        """
        if name == "length":
            self.waveform.set_param(name, value)
        elif name in {"freq", "gain", "ro_length"}:
            setattr(self, name, value)
        else:
            raise ValueError(f"Unknown parameter: {name}")


ReadoutCfg: TypeAlias = PulseReadoutCfg


class AbsReadout(Module):
    @abstractmethod
    def total_length(self, prog: ModularProgramV2) -> float | QickParam: ...

    def allow_rerun(self) -> bool:
        return True


def Readout(name: str, cfg: AbsReadoutCfg) -> AbsReadout:
    """Factory: dispatch a readout cfg to its concrete impl."""
    return cfg.build(name)


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
        """Capture a flat readout and optional hardware-word register inputs.

        name identifies ADC and pulse templates. cfg is copied and not mutated.
        gain_val names a hardware gain-word register; freq_val/ro_freq_val are
        paired generator/ADC frequency-word registers, already lowered for mixer
        and downconversion sign. Unspecified registers use cfg scalar/sweep values.
        init must precede run. Invalid QICK settings fail during initialization.
        """
        self.name = name
        self.cfg = deepcopy(cfg)
        self.gain_val = gain_val
        self.freq_val = freq_val
        self.ro_freq_val = ro_freq_val
        pulse_cfg = PulseCfg(
            ch=self.cfg.gen_ch,
            nqz=self.cfg.nqz,
            freq=self.cfg.freq,
            gain=self.cfg.gain,
            phase=self.cfg.phase,
            waveform=self.cfg.waveform,
            mixer_freq=self.cfg.mixer_freq,
            pre_delay=self.cfg.pre_delay,
            post_delay=self.cfg.post_delay,
        )
        self.pulse = Pulse(name=f"{name}_pulse", cfg=pulse_cfg, ro_ch=self.cfg.ro_ch)
        self._runtime_pulse: Pulse | None = None
        self._runtime_ro_name: str | None = None

    @property
    def adc_name(self) -> str:
        """Registered ADC template name, also used for table-backed word updates."""
        return f"{self.name}_adc"

    def init(self, prog: ModularProgramV2) -> None:
        """Declare the generator, ADC window and fixed/runtime configuration templates."""
        self.pulse.init(prog)
        prog.declare_readout(ch=self.cfg.ro_ch, length=self.cfg.ro_length)
        prog.add_readoutconfig(
            ch=self.cfg.ro_ch,
            name=self.adc_name,
            freq=self.cfg.freq,
            gen_ch=self.cfg.gen_ch,
        )
        if self._uses_runtime_pulse:
            self._runtime_pulse = Pulse(
                name=f"{self.name}_runtime_pulse",
                cfg=self.pulse.cfg,
                ro_ch=self.cfg.ro_ch,
                pulse_id=f"{self.name}_runtime_pulse",
            )
            self._runtime_pulse.init(prog)
        if self.ro_freq_val is not None:
            self._runtime_ro_name = f"{self.name}_adc_runtime"
            prog.add_readoutconfig(
                ch=self.cfg.ro_ch,
                name=self._runtime_ro_name,
                freq=self.cfg.freq,
                gen_ch=self.cfg.gen_ch,
            )

    def total_length(self, prog: ModularProgramV2) -> float | QickParam:
        """Return blocking us after init, covering generator delays and ADC window.

        Trigger offset and ADC length follow their respective clock rounding;
        generator playback uses Pulse's registered scalar/sweep duration.
        """
        return calc_max_length(
            round_timestamp(
                prog,
                round_timestamp(prog, self.cfg.trig_offset)
                + round_timestamp(prog, self.cfg.ro_length, ro_ch=self.cfg.ro_ch),
            ),
            self.pulse.total_length(prog),
        )

    @property
    def _uses_runtime_pulse(self) -> bool:
        return self.gain_val is not None or self.freq_val is not None

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        """Send configuration at t, trigger at t+offset, and pulse at t+pre_delay.

        t and the returned blocking end time are us; scalar and QICK sweeps are
        supported after init. Re-running leaves captured cfg/templates unchanged.
        """
        if self.ro_freq_val is None:
            prog.send_readoutconfig(self.cfg.ro_ch, self.adc_name, t=t)
            prog.trigger([self.cfg.ro_ch], t=t + self.cfg.trig_offset)
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
        ro_ch = self.cfg.ro_ch
        prog.send_readoutconfig_from_regs(ro_ch, name, t=t, freq_reg=self.ro_freq_val)
        prog.trigger([ro_ch], t=t + self.cfg.trig_offset)


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
        """Capture flat cfg and paired uint32 words for a named sweep loop.

        idx_reg must match exactly one declared loop with the same point count.
        Empty, unequal, or invalid uint32 tables raise ValueError. The first pair
        must encode cfg.freq; each later pair represents one semantic frequency.
        Dedicated templates isolate updates from shared pulses. The final update
        wraps to the first pair before the next outer repetition.
        """
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
            cfg=self._readout.pulse.cfg,
            ro_ch=self.cfg.ro_ch,
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
                name=self._readout.adc_name,
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
