"""Integration tests for all Module subclasses using a real ModularProgramV2.

Each test constructs a real program (triggering real QICK compilation) against
the mock QickConfig fixture, verifying that the module's init/run pipeline
integrates with the actual QICK ASM backend without hardware.
"""

from __future__ import annotations

from typing import Literal

import pytest
from qick.asm_v2 import Label, WriteReg
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.macro.loop import OpenInnerLoop
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.control import Branch, Repeat
from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, Join, SoftDelay
from zcu_tools.program.v2.modules.dmem import LoadValue, ScanWith
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import (
    DirectReadout,
    DirectReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
    Readout,
)
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    NoneResetCfg,
    PulseResetCfg,
    Reset,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from .conftest import make_mock_soccfg

# ---------------------------------------------------------------------------
# Constants — valid for the mock soccfg
# GEN: axis_sg_int4_v1, f_dds=6553.6 MHz → nqz=1 max ~3276 MHz
# RO:  axis_readout_v2, f_dds=1000.0 MHz → max ~500 MHz
# ---------------------------------------------------------------------------

GEN_CH = 0
GEN_CH2 = 1
# axis_signal_gen_v6: f_dds=6144 MHz, nqz=1 range 0–3072 MHz
GEN_FREQ = 1000.0
GEN_GAIN = 0.5
GEN_NQZ = 1
WAVEFORM_LEN = 0.5  # µs

RO_CH = 0
# axis_readout_v2: f_dds=2457.6 MHz, range 0–1228.8 MHz
RO_FREQ = 100.0
RO_LENGTH = 1.0  # µs


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_prog(modules=None, sweep=None, n_gens=2, n_readouts=1, **cfg_kwargs):
    soccfg = make_mock_soccfg(n_gens=n_gens, n_readouts=n_readouts)
    cfg = ProgramV2Cfg(**cfg_kwargs)
    return ModularProgramV2(soccfg, cfg, modules=modules or [], sweep=sweep)


def _label_addr(labels: dict[str, str], name: str) -> int:
    return int(labels[name].removeprefix("&"))


def _pulse_cfg(
    ch=GEN_CH,
    freq=GEN_FREQ,
    gain=GEN_GAIN,
    nqz: "Literal[1, 2]" = GEN_NQZ,
    length=WAVEFORM_LEN,
    pre_delay=0.0,
    post_delay=0.0,
):
    return PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=ch,
        nqz=nqz,
        freq=freq,
        gain=gain,
        pre_delay=pre_delay,
        post_delay=post_delay,
    )


def _direct_ro_cfg(ro_ch=RO_CH, ro_freq=RO_FREQ, ro_length=RO_LENGTH):
    return DirectReadoutCfg(ro_ch=ro_ch, ro_length=ro_length, ro_freq=ro_freq)


# ---------------------------------------------------------------------------
# ModularProgram temp_reg scope
# ---------------------------------------------------------------------------


class TestTempRegScope:
    def test_acquire_temp_reg_with_scope(self):
        prog = _make_prog(modules=[])

        with prog.acquire_temp_reg(1) as regs:
            assert len(regs) == 1
            assert regs[0] == "temp_reg_0"

    def test_acquire_temp_reg_nested_scope_is_disjoint(self):
        prog = _make_prog(modules=[])

        with prog.acquire_temp_reg(2) as outer_regs:
            with prog.acquire_temp_reg(1) as inner_regs:
                # nested scope must not clobber outer regs
                assert set(inner_regs).isdisjoint(outer_regs)
                assert inner_regs == ["temp_reg_2"]
                assert outer_regs == ["temp_reg_0", "temp_reg_1"]

    def test_acquire_temp_reg_zero_returns_empty_list(self):
        prog = _make_prog(modules=[])

        with prog.acquire_temp_reg(0) as regs:
            assert regs == []

    def test_acquire_temp_reg_negative_raises(self):
        prog = _make_prog(modules=[])

        with pytest.raises(ValueError, match="greater than or equal to 0"):
            with prog.acquire_temp_reg(-1):
                pass


# ---------------------------------------------------------------------------
# Delay modules
# ---------------------------------------------------------------------------


class TestDelayIntegration:
    def test_delay_compiles(self):
        prog = _make_prog(modules=[Delay("d", 1.0)])
        assert prog.binprog is not None

    def test_softdelay_compiles(self):
        prog = _make_prog(modules=[SoftDelay("s", 0.5)])
        assert prog.binprog is not None

    def test_delayauto_compiles(self):
        prog = _make_prog(modules=[DelayAuto("a")])
        assert prog.binprog is not None

    def test_delayauto_gens_only(self):
        prog = _make_prog(modules=[DelayAuto("a", gens=True, ros=False)])
        assert prog.binprog is not None

    def test_join_two_softdelays(self):
        prog = _make_prog(modules=[Join([SoftDelay("a", 0.2)], [SoftDelay("b", 0.3)])])
        assert prog.binprog is not None

    def test_delay_resets_time(self):
        modules = [SoftDelay("pre", 0.1), Delay("d", 1.0), SoftDelay("post", 0.2)]
        prog = _make_prog(modules=modules)
        assert prog.binprog is not None

    def test_multiple_delays_sequence(self):
        modules = [Delay("d1", 0.5), Delay("d2", 1.0), Delay("d3", 0.5)]
        prog = _make_prog(modules=modules)
        assert prog.binprog is not None


# ---------------------------------------------------------------------------
# Pulse module
# ---------------------------------------------------------------------------


class TestPulseIntegration:
    def test_pulse_const_waveform_compiles(self):
        prog = _make_prog(modules=[Pulse("p", _pulse_cfg())])
        assert prog.binprog is not None

    def test_pulse_none_cfg_compiles(self):
        prog = _make_prog(modules=[Pulse("p", None)])
        assert prog.binprog is not None

    def test_pulse_nqz2_compiles(self):
        # nqz=2 uses the upper Nyquist zone; freq > f_dds/2 is allowed
        prog = _make_prog(modules=[Pulse("p", _pulse_cfg(nqz=2, freq=4000.0))])
        assert prog.binprog is not None

    def test_pulse_pre_post_delay_compiles(self):
        prog = _make_prog(
            modules=[Pulse("p", _pulse_cfg(pre_delay=0.1, post_delay=0.1))]
        )
        assert prog.binprog is not None

    def test_pulse_two_pulses_diff_channels(self):
        modules = [
            Pulse("p0", _pulse_cfg(ch=GEN_CH)),
            Pulse("p1", _pulse_cfg(ch=GEN_CH2)),
        ]
        prog = _make_prog(modules=modules, n_gens=2)
        assert prog.binprog is not None

    def test_pulse_non_blocking_mode(self):
        prog = _make_prog(modules=[Pulse("p", _pulse_cfg(), block_mode=False)])
        assert prog.binprog is not None


# ---------------------------------------------------------------------------
# Readout module
# ---------------------------------------------------------------------------


class TestReadoutIntegration:
    def test_direct_readout_compiles(self):
        prog = _make_prog(modules=[DirectReadout("ro", _direct_ro_cfg())])
        assert prog.binprog is not None

    def test_direct_readout_with_gen_ch(self):
        # gen_ch links the readout to a declared generator for frequency matching.
        # Both the gen and readout must be initialised; wrap in separate modules.
        ro_cfg = DirectReadoutCfg(
            ro_ch=RO_CH, ro_length=RO_LENGTH, ro_freq=RO_FREQ, gen_ch=GEN_CH
        )
        pulse_cfg = _pulse_cfg(ch=GEN_CH)
        modules = [Pulse("p", pulse_cfg), DirectReadout("ro", ro_cfg)]
        prog = _make_prog(modules=modules)
        assert prog.binprog is not None

    def test_pulse_readout_compiles(self):
        cfg = PulseReadoutCfg(
            pulse_cfg=_pulse_cfg(ch=GEN_CH),
            ro_cfg=_direct_ro_cfg(ro_ch=GEN_CH),
        )
        prog = _make_prog(modules=[PulseReadout("ro", cfg)])
        assert prog.binprog is not None

    def test_readout_factory_direct_compiles(self):
        prog = _make_prog(modules=[Readout("ro", _direct_ro_cfg())])
        assert prog.binprog is not None

    def test_readout_factory_pulse_compiles(self):
        cfg = PulseReadoutCfg(
            pulse_cfg=_pulse_cfg(ch=GEN_CH),
            ro_cfg=_direct_ro_cfg(ro_ch=GEN_CH),
        )
        prog = _make_prog(modules=[Readout("ro", cfg)])
        assert prog.binprog is not None

    def test_pulse_plus_readout_compiles(self):
        modules = [
            Pulse("drive", _pulse_cfg(ch=GEN_CH)),
            DirectReadout("ro", _direct_ro_cfg()),
        ]
        prog = _make_prog(modules=modules)
        assert prog.binprog is not None


# ---------------------------------------------------------------------------
# Reset module
# ---------------------------------------------------------------------------


class TestResetIntegration:
    def test_none_reset_compiles(self):
        prog = _make_prog(modules=[Reset("r", NoneResetCfg())])
        assert prog.binprog is not None

    def test_reset_none_cfg_compiles(self):
        prog = _make_prog(modules=[Reset("r", None)])
        assert prog.binprog is not None

    def test_pulse_reset_compiles(self):
        cfg = PulseResetCfg(pulse_cfg=_pulse_cfg(ch=GEN_CH))
        prog = _make_prog(modules=[Reset("r", cfg)])
        assert prog.binprog is not None

    def test_two_pulse_reset_compiles(self):
        cfg = TwoPulseResetCfg(
            pulse1_cfg=_pulse_cfg(ch=GEN_CH, length=0.2),
            pulse2_cfg=_pulse_cfg(ch=GEN_CH2, length=0.1),
        )
        prog = _make_prog(modules=[Reset("r", cfg)], n_gens=2)
        assert prog.binprog is not None

    def test_bath_reset_compiles(self):
        # BathReset requires 3 separate pulse configs on up to 3 channels
        cfg = BathResetCfg(
            cavity_tone_cfg=_pulse_cfg(ch=GEN_CH, freq=GEN_FREQ),
            qubit_tone_cfg=_pulse_cfg(ch=GEN_CH2, freq=GEN_FREQ),
            pi2_cfg=_pulse_cfg(ch=GEN_CH, freq=GEN_FREQ),
        )
        prog = _make_prog(modules=[Reset("r", cfg)], n_gens=2)
        assert prog.binprog is not None


# ---------------------------------------------------------------------------
# Control module
# ---------------------------------------------------------------------------


class TestControlIntegration:
    def test_open_inner_loop_expands_counter_init_before_start_label(self):
        class _Prog:
            def _get_reg(self, value):
                return value

        macros = OpenInnerLoop(name="r", counter_reg="r", n=4).expand(_Prog())

        assert isinstance(macros[1], WriteReg)
        assert isinstance(macros[2], Label)

    def test_repeat_zero_compiles(self):
        r = Repeat("r", 0)
        r.add_content(SoftDelay("d", 0.1))
        prog = _make_prog(modules=[r])
        assert prog.binprog is not None

    def test_repeat_n_compiles(self):
        r = Repeat("r", 3)
        r.add_content(SoftDelay("d", 0.1))
        prog = _make_prog(modules=[r])
        assert prog.binprog is not None

    def test_repeat_with_pulse(self):
        r = Repeat("r", 5)
        r.add_content(Pulse("p", _pulse_cfg()))
        prog = _make_prog(modules=[r])
        assert prog.binprog is not None

    def test_repeat_register_driven_compiles(self):
        # Register-driven loop: Repeat(name, n_reg) where name is the counter register
        # and n_reg is the name of a pre-existing register holding the count.
        # Use sweep to create "n_count" register, then Repeat("r_cnt", "n_count") creates
        # a separate "r_cnt" counter register. No name collision.
        r = Repeat("r_cnt", "n_count")
        r.add_content(SoftDelay("d", 0.1))
        prog = _make_prog(modules=[r], sweep=[("n_count", 4)])
        assert prog.binprog is not None

    def test_repeat_register_driven_loop_roundtrip_shape(self):
        from zcu_tools.program.v2.ir.pipeline import DEFAULT_PIPELINE_CONFIG

        old_val = DEFAULT_PIPELINE_CONFIG.enable_unroll_loop
        DEFAULT_PIPELINE_CONFIG.enable_unroll_loop = False
        try:
            r = Repeat("r_cnt", "n_count")
            r.add_content(SoftDelay("d", 0.1))
            prog = _make_prog(modules=[r], sweep=[("n_count", 4)])

            start_addr = _label_addr(prog.labels, "r_cnt_start")
            end_addr = _label_addr(prog.labels, "r_cnt_end")

            # Counter init (REG_WR imm #0) sits between the runtime guard and
            # the start label.
            init_inst = max(
                (
                    inst
                    for inst in prog.prog_list
                    if inst.get("CMD") == "REG_WR"
                    and inst.get("SRC") == "imm"
                    and inst.get("LIT") == "#0"
                    and inst["P_ADDR"] < start_addr
                ),
                key=lambda inst: inst["P_ADDR"],
            )
            assert init_inst["P_ADDR"] < start_addr

            # Runtime guard: cond-jump with IF=Z over the loop, sitting before
            # init (so `n_count == 0` skips the entire body).
            guard_jump = next(
                inst
                for inst in prog.prog_list
                if inst.get("CMD") == "JUMP"
                and inst.get("IF") == "Z"
                and inst["P_ADDR"] < init_inst["P_ADDR"]
            )
            assert "OP" in guard_jump

            # Back-edge: condensed cond-jump with IF=NS sitting just before
            # end_addr (replaces the old TEST + unconditional JUMP pair).
            back_jump = next(
                inst
                for inst in prog.prog_list
                if inst.get("CMD") == "JUMP"
                and inst.get("IF") == "NS"
                and inst["P_ADDR"] < end_addr
                and inst["P_ADDR"] > start_addr
            )
            assert "OP" in back_jump

            # No unconditional JUMP back to start any more — increment +
            # cond-jump combined into one back-edge.
            unconditional_jumps = [
                inst
                for inst in prog.prog_list
                if inst.get("CMD") == "JUMP"
                and "IF" not in inst
                and inst["P_ADDR"] < end_addr
                and inst["P_ADDR"] > start_addr
            ]
            assert unconditional_jumps == []

            # Per-iteration counter increment lives just before the back-edge.
            incr_inst = next(
                inst
                for inst in prog.prog_list
                if inst.get("CMD") == "REG_WR"
                and inst.get("SRC") == "op"
                and inst.get("OP", "").endswith(" + #1")
                and inst["P_ADDR"] < back_jump["P_ADDR"]
                and inst["P_ADDR"] > start_addr
            )
            assert incr_inst["P_ADDR"] < back_jump["P_ADDR"]
        finally:
            DEFAULT_PIPELINE_CONFIG.enable_unroll_loop = old_val

    def test_nested_repeat_compiles(self):
        inner = Repeat("inner", 2)
        inner.add_content(SoftDelay("d", 0.1))
        outer = Repeat("outer", 3)
        outer.add_content(inner)
        prog = _make_prog(modules=[outer])
        assert prog.binprog is not None


# ---------------------------------------------------------------------------
# Dmem module
# ---------------------------------------------------------------------------


class TestDmemIntegration:
    def _scan_values_small(self):
        return list(range(5))

    def _scan_values_large(self):
        return list(range(40))

    def test_load_value_small_uncompressed(self):
        # 5 values → below compression threshold (30), stays uncompressed
        lv = LoadValue(
            "lv", self._scan_values_small(), idx_reg="myloop", val_reg="myval"
        )
        r = Repeat("myloop", len(self._scan_values_small()))
        r.add_content(lv)
        prog = _make_prog(modules=[r])
        assert prog.binprog is not None
        assert not lv._is_compressed

    def test_load_value_large_compressed(self):
        # 40 values → triggers auto_compress
        vals = self._scan_values_large()
        lv = LoadValue("lv", vals, idx_reg="myloop", val_reg="myval")
        r = Repeat("myloop", len(vals))
        r.add_content(lv)
        prog = _make_prog(modules=[r])
        assert prog.binprog is not None
        assert lv._is_compressed

    def test_load_value_negative_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            LoadValue("lv", [-1, 0, 1], idx_reg="myloop", val_reg="myval")

    def test_compile_datamem_not_none(self):
        vals = self._scan_values_large()
        lv = LoadValue("lv", vals, idx_reg="myloop", val_reg="myval")
        r = Repeat("myloop", len(vals))
        r.add_content(lv)
        prog = _make_prog(modules=[r])
        assert prog.compile_datamem() is not None

    def test_scan_with_compiles(self):
        vals = list(range(10))
        s = ScanWith("s", vals, "myval")
        s.add_content(SoftDelay("d", 0.1))
        prog = _make_prog(modules=[s])
        assert prog.binprog is not None

    def test_scan_with_large_values_compiles(self):
        vals = list(range(40))
        s = ScanWith("s", vals, "myval")
        s.add_content(SoftDelay("d", 0.1))
        prog = _make_prog(modules=[s])
        assert prog.binprog is not None
        assert prog.compile_datamem() is not None
