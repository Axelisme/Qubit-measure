from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import BeforeValidator
from qick.asm_v2 import QickParam

from zcu_tools.cfg_model import MuxReadoutCfg as CoreMuxReadoutCfg

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

    def to_pulse_cfg(self) -> PulseCfg:
        """Return a fresh outer PulseCfg containing only generator settings.

        gen_ch maps to ch; ADC channel, window and trigger settings are omitted.
        Pass ro_ch separately to Pulse when ADC frequency matching is needed.
        The waveform and any QickParam values are borrowed, not deep-copied;
        callers must not mutate these shared nested objects through the result.
        PulseCfg validation runs as usual and its errors propagate unchanged.
        """
        return PulseCfg(
            ch=self.gen_ch,
            nqz=self.nqz,
            freq=self.freq,
            gain=self.gain,
            phase=self.phase,
            waveform=self.waveform,
            mixer_freq=self.mixer_freq,
            pre_delay=self.pre_delay,
            post_delay=self.post_delay,
        )

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


class MuxReadoutCfg(CoreMuxReadoutCfg, AbsReadoutCfg):
    """Native fixed mux settings; numeric fields and validation belong to cfg_model.

    Inherited fields retain MHz, degrees, normalized gain and us units. build
    captures a copy for mixmux8/static PFBv4; every set_param call is rejected.
    """

    def build(self, name: str) -> MuxReadout:
        """Capture these numeric settings as a named fixed mux readout module."""
        return MuxReadout(name, self)


ReadoutCfg: TypeAlias = PulseReadoutCfg | MuxReadoutCfg


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
        pulse_cfg = self.cfg.to_pulse_cfg()
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

    @property
    def initial_template(self) -> PulseReadout:
        """Borrow the initial scalar/swept template used before exec-after updates.

        The returned Module is owned by this table readout and must not be
        mutated by consumers. init(prog) initializes it and its dedicated pulse
        identity. It is not the current mutable hardware wmem contents; semantic
        interpreters own their own per-execution template-word update state.
        """
        return self._readout

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


class MuxReadout(AbsReadout):
    """Fixed mixmux8 playback and simultaneous static PFBv4 acquisition.

    Only explicit tone channels are used, in their original order. Initialization
    rejects incompatible hardware, duplicate routes and sub-cycle windows. No
    envelope, tProc readout configuration or hardware parameter update is used.
    """

    def __init__(self, name: str, cfg: MuxReadoutCfg) -> None:
        """Copy numeric cfg; name identifies the const mask pulse and diagnostics."""
        self.name = name
        self.cfg = deepcopy(cfg)
        self._program: ModularProgramV2 | None = None

    def init(self, prog: ModularProgramV2) -> None:
        """Validate cached hardware/routes, then declare all tones and ADC windows.

        Raises ValueError with the module name for incompatible wiring/capacity
        or illegal rounded durations. QICK frequency/format errors propagate.
        """
        self._validate_generator(prog)
        self._validate_readouts(prog)
        cfg = self.cfg
        prog.declare_gen(
            ch=cfg.gen_ch,
            nqz=cfg.nqz,
            mixer_freq=cfg.mixer_freq,
            ro_ch=cfg.tones[0].ro_ch,
            mux_freqs=[tone.freq for tone in cfg.tones],
            mux_gains=[tone.gain for tone in cfg.tones],
            mux_phases=[tone.phase for tone in cfg.tones],
        )
        for tone in cfg.tones:
            prog.declare_readout(
                ch=tone.ro_ch,
                length=cfg.ro_length,
                freq=tone.freq,
                phase=0,
                gen_ch=cfg.gen_ch,
            )
        prog.add_pulse(
            ch=cfg.gen_ch,
            name=self.name,
            style="const",
            mask=list(range(len(cfg.tones))),
            length=cfg.length,
        )
        prog.protect_readout_routes(
            self.name, cfg.gen_ch, [tone.ro_ch for tone in cfg.tones]
        )
        self._program = prog

    def _validate_generator(self, prog: ModularProgramV2) -> None:
        ch = self.cfg.gen_ch
        gens = prog.soccfg["gens"]
        if ch >= len(gens):
            raise ValueError(f"{self.name}.gen_ch: missing generator {ch}")
        gen = gens[ch]
        if (
            gen["type"] != "axis_sg_mixmux8_v1"
            or not gen["has_mixer"]
            or not gen["has_gain"]
            or not gen["has_phase"]
            or len(self.cfg.tones) > gen["n_tones"]
        ):
            raise ValueError(
                f"{self.name}.gen_ch: generator {ch} requires mixmux8 mixer/gain/phase capacity"
            )
        if ch in prog.gen_chs:
            raise ValueError(
                f"{self.name}.gen_ch: generator route {ch} already declared"
            )
        cycles = prog.us2cycles(self.cfg.length, gen_ch=ch)
        if not 3 <= cycles < 2**32:
            raise ValueError(
                f"{self.name}.length: generator {ch} rounded length must be 3..2**32-1 cycles"
            )

    def _validate_readouts(self, prog: ModularProgramV2) -> None:
        channels: set[int] = set()
        ports: set[int] = set()
        path: str | None = None
        readouts = prog.soccfg["readouts"]
        for i, tone in enumerate(self.cfg.tones):
            ch = tone.ro_ch
            where = f"{self.name}.tones[{i}].ro_ch: ADC {ch}"
            if ch >= len(readouts) or ch in channels or ch in prog.ro_chs:
                raise ValueError(f"{where} missing or duplicate readout route")
            ro = readouts[ch]
            if (
                ro["ro_type"] != "axis_pfb_readout_v4"
                or "tproc_ctrl" in ro
                or not ro.get("ro_fullpath")
                or not ro["pfb_dds_on_output"]
            ):
                raise ValueError(f"{where} requires static PFBv4")
            if path is not None and path != ro["ro_fullpath"]:
                raise ValueError(f"{where} must use the same PFB ro_fullpath")
            path = ro["ro_fullpath"]
            port = ro["pfb_port"]
            if (
                len(self.cfg.tones) > ro["pfb_nout"]
                or not 0 <= port < ro["pfb_nout"]
                or port in ports
            ):
                raise ValueError(f"{where} invalid or duplicate PFB output/capacity")
            if prog.us2cycles(self.cfg.ro_length, ro_ch=ch) <= 0:
                raise ValueError(
                    f"{self.name}.ro_length: ADC {ch} rounded window must be positive"
                )
            channels.add(ch)
            ports.add(port)

    def total_length(self, prog: ModularProgramV2) -> float:
        """Return blocking us covering actual rounded playback and every ADC window.

        init must have succeeded. Start/post delays round upward on the processor
        clock; playback and ADC windows use QICK's actual channel-cycle rounding.
        """
        if self._program is not prog:
            raise ValueError(
                f"{self.name}: initialize this module with this program first"
            )
        cfg = self.cfg
        duration = prog.cycles2us(
            prog.us2cycles(cfg.length, gen_ch=cfg.gen_ch), gen_ch=cfg.gen_ch
        )
        gen_end = round_timestamp(
            prog,
            round_timestamp(prog, cfg.pre_delay)
            + duration
            + round_timestamp(prog, cfg.post_delay),
        )
        adc_duration = max(
            prog.cycles2us(prog.us2cycles(cfg.ro_length, ro_ch=t.ro_ch), ro_ch=t.ro_ch)
            for t in cfg.tones
        )
        adc_end = round_timestamp(
            prog, round_timestamp(prog, cfg.trig_offset) + adc_duration
        )
        return float(max(gen_end, adc_end))

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        """Play all tones at t+pre_delay and trigger all chains at t+trig_offset.

        init must precede run; returns blocking end in us. Repeated fixed
        playback is supported, but cfg/program must be rebuilt for any update.
        """
        length = self.total_length(prog)
        prog.pulse(ch=self.cfg.gen_ch, name=self.name, t=t + self.cfg.pre_delay)
        prog.trigger(
            [tone.ro_ch for tone in self.cfg.tones], t=t + self.cfg.trig_offset
        )
        return t + length

    def collect_raw_iq(
        self,
        prog: ModularProgramV2,
        *,
        read_index: int | None = None,
    ) -> NDArray[np.float64]:
        """Return last-round raw IQ in tone order, normalized by ADC length.

        prog must be the program that initialized this module. Call after its
        integrated acquire(), or in a completed-round hook. This reads get_raw()
        from that program, never acquires or samples a new round. A simulation
        getter may materialize retained shots using that round's fixed identity.
        Multiple rounds are NOT concatenated: get_raw() retains only the last
        round. Collect in each completed-round hook to retain every round.

        Input buffers must be int64 QICK sums shaped (*prog.loop_dims,reads,2)
        in actual ADC declaration order. The returned independent float64 array
        has shape (*prog.loop_dims,reads,N,2), with N in cfg.tones order and the
        final axis [I,Q]. Every loop axis, including reps at prog.avg_level, keeps
        its original position and shot order. No preparation meaning is assigned.
        read_index=None keeps every read; a nonnegative int selects the same read
        on all tones and returns (*prog.loop_dims,N,2). No shots are averaged.

        Values are ADC units per integration sample: sums divided by each ADC's
        declared, rounded length in samples. DC offsets remain, unlike QICK's
        default averaged acquisition. No rotation or classification is applied.
        Hardware and simulation get_raw() share this normalization contract.

        ValueError names this module for the wrong/uninitialized program, absent
        raw buffers/channels, count/dtype/loop/read/IQ shape mismatches or invalid
        read_index. Unrelated channels' shapes are ignored. Getter errors propagate.
        """
        if self._program is not prog:
            raise ValueError(
                f"{self.name}: use the program that initialized this module"
            )
        if read_index is not None and (type(read_index) is not int or read_index < 0):
            raise ValueError(
                f"{self.name}: read_index must be None or a nonnegative integer"
            )
        positions = {ch: i for i, ch in enumerate(prog.ro_chs)}
        for i, tone in enumerate(self.cfg.tones):
            if tone.ro_ch not in positions:
                raise ValueError(f"{self.name}: tone {i} missing ro_ch {tone.ro_ch}")
        raw = prog.get_raw()
        if raw is None:
            raise ValueError(
                f"{self.name}: no raw buffers; acquire integrated IQ first"
            )
        if len(raw) != len(positions):
            raise ValueError(
                f"{self.name}: raw item count {len(raw)} != channels {len(positions)}"
            )
        if prog.loop_dims is None:
            raise ValueError(f"{self.name}: raw loop dimensions are unavailable")
        reads = prog.ro_chs[self.cfg.tones[0].ro_ch]["trigs"]
        expected_shape = (*prog.loop_dims, reads, 2)
        selected: list[NDArray[np.float64]] = []
        for i, tone in enumerate(self.cfg.tones):
            pos = positions[tone.ro_ch]
            array = raw[pos]
            ro = prog.ro_chs[tone.ro_ch]
            where = f"{self.name}: tone {i} ro_ch {tone.ro_ch} raw[{pos}]"
            if array.dtype != np.int64:
                raise ValueError(f"{where} must be int64 QICK sums")
            if reads <= 0 or ro["trigs"] != reads or array.shape != expected_shape:
                raise ValueError(f"{where} shape {array.shape} != {expected_shape}")
            if read_index is not None and read_index >= reads:
                raise ValueError(f"{where} read_index {read_index} out of bounds")
            iq = array.astype(np.float64) / ro["length"]
            selected.append(iq if read_index is None else iq[..., read_index, :])
        return np.stack(selected, axis=-2)

    def collect_iq(
        self,
        prog: ModularProgramV2,
        acquired: Sequence[NDArray[np.float64]],
        *,
        read_index: int = 0,
    ) -> NDArray[np.float64]:
        """Select one averaged read in tone order, returning (*sweep_shape,N,2).

        Caller must supply the same initialized program's same acquire result:
        an unclassified averaged list of float64 arrays (reads,*sweep_shape,2).
        The list carries no source identity, so another acquisition's same-shaped
        arrays cannot be detected. Raw, single-shot, classified and decimated
        outputs are not supported. Values (including NaN), normalization and
        units are preserved; no complex conversion, interpretation or I/O occurs.

        ValueError names this module and the invalid position/tone/channel for
        missing initialization/channels, list count, read_index, dtype or shape.
        Unrelated channels' shapes are ignored.
        """
        if self._program is None:
            raise ValueError(f"{self.name}: module is not initialized")
        positions = {ch: i for i, ch in enumerate(prog.ro_chs)}
        for i, tone in enumerate(self.cfg.tones):
            if tone.ro_ch not in positions:
                raise ValueError(f"{self.name}: tone {i} missing ro_ch {tone.ro_ch}")
        if self._program is not prog:
            raise ValueError(
                f"{self.name}: use the program that initialized this module"
            )
        if len(acquired) != len(positions):
            raise ValueError(
                f"{self.name}: acquired item count {len(acquired)} != channels {len(positions)}"
            )
        if type(read_index) is not int or read_index < 0:
            raise ValueError(f"{self.name}: read_index must be a nonnegative integer")
        selected: list[NDArray[np.float64]] = []
        shape: tuple[int, ...] | None = None
        for i, tone in enumerate(self.cfg.tones):
            pos = positions[tone.ro_ch]
            array = acquired[pos]
            where = f"{self.name}: tone {i} ro_ch {tone.ro_ch} acquired[{pos}]"
            if array.dtype != np.float64 or array.ndim < 2 or array.shape[-1] != 2:
                raise ValueError(f"{where} must be float64 (reads,*sweep_shape,2)")
            if shape is not None and array.shape != shape:
                raise ValueError(f"{where} read/sweep shape {array.shape} != {shape}")
            if read_index >= array.shape[0]:
                raise ValueError(f"{where} read_index {read_index} out of bounds")
            shape = array.shape
            selected.append(array[read_index])
        return np.stack(selected, axis=-2)
