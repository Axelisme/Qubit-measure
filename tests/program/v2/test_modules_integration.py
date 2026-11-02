"""Integration tests for all Module subclasses using a real TimelineProgramV2.

Each test constructs a real program (triggering real QICK compilation) against
the mock QickConfig fixture, verifying that the module's init/run pipeline
integrates with the actual QICK ASM backend without hardware.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from qick.asm_v2 import Label, QickParam, WriteReg
from zcu_tools.program.v2 import (
    SweepCfg,
    TimelineBuilder,
    TimelineProgramV2,
    sweep2param,
)
from zcu_tools.program.v2.macro.loop import OpenInnerLoop
from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, SoftDelay
from zcu_tools.program.v2.modules.dmem import LoadValue, LoadWord
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import (
    PulseReadout,
    PulseReadoutCfg,
    Readout,
)
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    PulseResetCfg,
    Reset,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
)

from tests.program._profiles import load_soccfg
from tests.program._timeline import compile_modules

# ---------------------------------------------------------------------------
# Constants — valid for the mock soccfg
# GEN: axis_signal_gen_v6, f_dds=12288.0 MHz → nqz=1 range 0–6144 MHz
# RO:  axis_readout_v2,    f_dds=2457.6 MHz → range 0–1228.8 MHz
# ---------------------------------------------------------------------------

GEN_CH = 0
GEN_CH2 = 1
# axis_signal_gen_v6: f_dds=12288 MHz, nqz=1 range 0–6144 MHz
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
    soccfg = load_soccfg(n_gens=n_gens, n_readouts=n_readouts)
    return compile_modules(soccfg, modules=modules or [], sweep=sweep, **cfg_kwargs)


def _pulse_cfg(
    ch=GEN_CH,
    freq=GEN_FREQ,
    gain=GEN_GAIN,
    nqz: Literal[1, 2] = GEN_NQZ,
    length: float | QickParam = WAVEFORM_LEN,
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


def _readout_cfg(ro_ch=RO_CH, freq=RO_FREQ, ro_length=RO_LENGTH):
    return PulseReadoutCfg.model_validate(
        {
            "gen_ch": GEN_CH,
            "nqz": GEN_NQZ,
            "freq": freq,
            "gain": GEN_GAIN,
            "waveform": ConstWaveformCfg(length=WAVEFORM_LEN),
            "ro_ch": ro_ch,
            "ro_length": ro_length,
        }
    )


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

        with (
            prog.acquire_temp_reg(2) as outer_regs,
            prog.acquire_temp_reg(1) as inner_regs,
        ):
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

        with (
            pytest.raises(ValueError, match="greater than or equal to 0"),
            prog.acquire_temp_reg(-1),
        ):
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

    def test_swept_blocking_alignment_leaves_no_generator_residual_sweep(self) -> None:
        grid = SweepCfg(start=0.04, stop=1.2, expts=9, step=1.16 / 8)
        pulse = Pulse("drive", _pulse_cfg(length=sweep2param("length", grid)))
        from zcu_tools.program.v2 import TimelineBuilder

        builder = TimelineBuilder(load_soccfg())
        builder.add_loop("length", grid.expts)
        builder.emit(pulse)
        builder.delay_by_timeline()
        builder.delay_auto(gens=True, ros=False, tag="residual")
        builder.emit(Readout("readout", _readout_cfg()))
        prog = builder.build()
        residual = prog.get_time_param("residual", "t")
        assert isinstance(residual, float)
        assert residual == 0.0

    def test_pulse_none_cfg_compiles(self):
        prog = _make_prog(modules=[Pulse("p", None)])
        assert prog.binprog is not None

    def test_pulse_nqz2_compiles(self):
        # nqz=2 selects the upper Nyquist zone; the cfg still compiles there
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


# ---------------------------------------------------------------------------
# Readout module
# ---------------------------------------------------------------------------


class TestReadoutIntegration:
    @pytest.mark.parametrize("runtime", [False, True])
    def test_same_named_readouts_keep_program_owned_templates(self, runtime):
        cfg = _readout_cfg()

        def readout_module(gain):
            return PulseReadout(
                "r1",
                cfg,
                gain=gain,
                gain_val="gain_word" if runtime else None,
                ro_freq_val="ro_freq_word" if runtime else None,
            )

        first = readout_module(cfg.gain)
        second = readout_module(0.0)

        def build(modules):
            builder = TimelineBuilder(load_soccfg())
            builder.add_loop("choice", 2)
            if runtime:
                for name in ("gain_word", "ro_freq_word"):
                    builder.emit(LoadWord(name, [0, 0], idx_reg="choice", val_reg=name))
            with builder.branch(by="choice", align_timeline=True) as cases:
                for index, module in enumerate(modules):
                    with cases.case(index):
                        builder.emit(module)
            return builder.build()

        prog = build([first, second])
        names = (first.adc_name(prog), second.adc_name(prog))
        assert names[0] != names[1]
        assert all(name in prog.pulses for name in names)
        other = build([second, first])
        assert first.adc_name(other) != second.adc_name(other)
        assert (first.adc_name(prog), second.adc_name(prog)) == names
        assert prog.binprog is not None

    def test_pulse_readout_compiles(self):
        cfg = _readout_cfg()
        prog = _make_prog(modules=[PulseReadout("ro", cfg)])
        assert prog.binprog is not None

    def test_pulse_readout_runtime_freq_words_compile(self):
        cfg = _readout_cfg(freq=GEN_FREQ)
        modules = [
            LoadWord(
                "load_freq",
                values=[1, 2],
                idx_reg="freq",
                val_reg="freq_word",
            ),
            LoadWord(
                "load_ro_freq",
                values=[3, 4],
                idx_reg="freq",
                val_reg="ro_freq_word",
            ),
            PulseReadout(
                "ro",
                cfg,
                freq_val="freq_word",
                ro_freq_val="ro_freq_word",
            ),
        ]

        prog = _make_prog(modules=modules, sweep=[("freq", 2)])

        assert prog.binprog is not None
        runtime_ports = [
            inst
            for inst in prog.prog_list
            if inst.get("CMD") == "WPORT_WR" and inst.get("SRC") == "r_wave"
        ]
        assert len(runtime_ports) == 2
        assert not any(inst.get("CMD") == "WMEM_WR" for inst in prog.prog_list)

    def test_readout_factory_pulse_compiles(self):
        cfg = _readout_cfg()
        prog = _make_prog(modules=[Readout("ro", cfg)])
        assert prog.binprog is not None

    def test_pulse_plus_readout_compiles(self):
        modules = [
            Pulse("drive", _pulse_cfg(ch=GEN_CH)),
            Readout("ro", _readout_cfg()),
        ]
        prog = _make_prog(modules=modules)
        assert prog.binprog is not None


# ---------------------------------------------------------------------------
# Reset module
# ---------------------------------------------------------------------------


class TestResetIntegration:
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


# ---------------------------------------------------------------------------
# Dmem module
# ---------------------------------------------------------------------------


class TestDmemIntegration:
    def _scan_values_small(self):
        return list(range(5))

    def _scan_values_large(self):
        return list(range(40))

    def test_load_value_negative_rejected(self):
        with pytest.raises(ValueError, match=r"\[0,"):
            LoadValue("lv", [-1, 0, 1], idx_reg="myloop", val_reg="myval")

    def test_binprog_rejects_non_vector_dmem(self):
        class TwoDimensionalDmemProgram(TimelineProgramV2):
            def compile_datamem(self):
                return np.asarray([[1, 2]], dtype=np.int32)

        soccfg = load_soccfg(n_gens=1, n_readouts=1)

        with pytest.raises(ValueError, match="dmem must be one-dimensional"):
            TwoDimensionalDmemProgram(soccfg)


@pytest.mark.parametrize("mixer", [None, 500.0])
def test_pulse_runtime_adc_identity_keeps_distinct_qick_definitions(mixer):
    soccfg = load_soccfg(n_gens=1, n_readouts=2)
    soccfg["gens"][0]["has_mixer"] = mixer is not None
    cfg = _pulse_cfg(freq=700.0)
    cfg.mixer_freq = mixer
    pulses = [Pulse(f"p{idx}", cfg, ro_ch=ch) for idx, ch in enumerate((None, 0, 0, 1))]
    prog = compile_modules(soccfg, modules=pulses)
    assert prog.pulse_registry.count == 3
    assert pulses[1].pulse_name(prog) == pulses[2].pulse_name(prog)
    assert len({pulses[idx].pulse_name(prog) for idx in (0, 1, 3)}) == 3
    for pulse, ch in zip(pulses, (None, 0, 0, 1), strict=True):
        assert prog.pulses[pulse.pulse_name(prog)].params["ro_ch"] == ch


@pytest.mark.parametrize("swept", [False, True])
def test_readout_generator_and_adc_compile_one_frequency(swept):
    cfg = _readout_cfg(freq=700.0)
    if swept:
        cfg.set_param("freq", QickParam(700.0, {"freq": 20.0}))
    prog = _make_prog(
        modules=[Readout("ro", cfg)], sweep=[("freq", 3)] if swept else None
    )
    names = [name for name in prog.pulses if name != "ro_adc"]
    assert len(names) == 1
    gen_freq = prog.get_pulse_param(names[0], "freq", as_array=swept)
    ro_freq = prog.get_pulse_param("ro_adc", "freq", as_array=swept)
    np.testing.assert_allclose(
        gen_freq, [700.0, 710.0, 720.0] if swept else 700.0, atol=1e-4
    )
    np.testing.assert_allclose(ro_freq, gen_freq, atol=1e-4)


def test_unprotected_routes_keep_qick_redeclaration_semantics():
    prog = _make_prog()
    prog.declare_gen(0, nqz=1)
    prog.declare_gen(0, nqz=2)
    prog.declare_readout(0, length=0.413)
    prog.declare_readout(0, length=0.827)
    assert prog.get_time_axis(0, length_only=True) == prog.us2cycles(0.827, ro_ch=0)
