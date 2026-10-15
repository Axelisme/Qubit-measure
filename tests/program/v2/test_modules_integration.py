"""Integration tests for all Module subclasses using a real ModularProgramV2.

Each test constructs a real program (triggering real QICK compilation) against
the mock QickConfig fixture, verifying that the module's init/run pipeline
integrates with the actual QICK ASM backend without hardware.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from qick.asm_v2 import Label, QickParam, WriteReg
from zcu_tools.program.v2 import SweepCfg, sweep2param
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.macro.loop import OpenInnerLoop
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.control import Repeat
from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, Join, SoftDelay
from zcu_tools.program.v2.modules.dmem import LoadValue, LoadWord, ScanWith
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import (
    PulseReadout,
    PulseReadoutCfg,
    Readout,
    TablePulseReadout,
)
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    NoneResetCfg,
    PulseResetCfg,
    Reset,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
from zcu_tools.program.v2.utils import readout_freq_words

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
    soccfg = make_mock_soccfg(n_gens=n_gens, n_readouts=n_readouts)
    cfg = ProgramV2Cfg(**cfg_kwargs)
    return ModularProgramV2(soccfg, cfg, modules=modules or [], sweep=sweep)


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

    @pytest.mark.parametrize("flat_top", [False, True])
    @pytest.mark.parametrize("start,stop", [(0.04, 1.2), (1.2, 0.04)])
    def test_blocking_length_sweep_tracks_registered_pulse_duration(
        self, flat_top: bool, start: float, stop: float
    ) -> None:
        # Both scan directions resolve on the generator clock.
        grid = SweepCfg(start=start, stop=stop, expts=9, step=(stop - start) / 8)
        length = sweep2param("length", grid)
        waveform = (
            FlatTopWaveformCfg(
                length=length,
                raise_waveform=GaussWaveformCfg(length=0.02, sigma=0.004),
            )
            if flat_top
            else ConstWaveformCfg(length=length)
        )
        pulse = Pulse(
            "drive",
            PulseCfg(
                waveform=waveform,
                ch=GEN_CH,
                nqz=GEN_NQZ,
                freq=GEN_FREQ,
                gain=GEN_GAIN,
            ),
            pulse_id="drive",
        )
        prog = _make_prog(
            modules=[pulse, Readout("readout", _readout_cfg())],
            sweep=[("length", grid)],
        )
        actual = np.asarray(
            prog.get_pulse_param("drive", "total_length", as_array=True),
            dtype=np.float64,
        )
        blocking = pulse.total_length(prog)
        assert isinstance(blocking, QickParam)
        timestamps = np.asarray(blocking.to_array(prog.loop_dict), dtype=np.float64)
        assert actual.shape == timestamps.shape == (grid.expts,)
        # Per-step processor rounding can accumulate conservative overrun.
        # It must never start the following readout before actual playback ends.
        assert np.all(timestamps >= actual - 1e-12)
        assert prog.binprog is not None

    def test_swept_blocking_alignment_leaves_no_generator_residual_sweep(self) -> None:
        grid = SweepCfg(start=0.04, stop=1.2, expts=9, step=1.16 / 8)
        pulse = Pulse("drive", _pulse_cfg(length=sweep2param("length", grid)))
        prog = _make_prog(
            modules=[
                pulse,
                Delay("advance", 0.0),
                DelayAuto("residual", gens=True, ros=False, tag="residual"),
                Readout("readout", _readout_cfg()),
            ],
            sweep=[("length", grid)],
        )
        residual = prog.get_time_param("residual", "t")
        assert isinstance(residual, float)
        assert residual == 0.0

    @pytest.mark.parametrize("block_mode", [False, True])
    def test_scalar_pre_post_blocking_timing_keeps_existing_value(
        self, block_mode: bool
    ) -> None:
        pulse = Pulse(
            "drive",
            _pulse_cfg(length=0.5, pre_delay=0.101, post_delay=0.203),
            block_mode=block_mode,
            pulse_id="drive",
        )
        prog = _make_prog(modules=[pulse, Readout("readout", _readout_cfg())])
        total = pulse.total_length(prog)
        assert isinstance(total, float)
        # Captured from the unchanged scalar path before the D190 correction.
        assert prog.us2cycles(us=total) == 348
        end = pulse.run(prog, t=0.25)
        assert end == pytest.approx(0.25 + total if block_mode else 0.25)
        assert prog.binprog is not None

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

    def test_pulse_non_blocking_mode(self):
        prog = _make_prog(modules=[Pulse("p", _pulse_cfg(), block_mode=False)])
        assert prog.binprog is not None


# ---------------------------------------------------------------------------
# Readout module
# ---------------------------------------------------------------------------


class TestReadoutIntegration:
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

    def test_table_pulse_readout_prepares_next_wmem_after_standard_playback(self):
        soccfg = make_mock_soccfg(n_gens=1, n_readouts=1)
        freqs = [1000.0, 1010.0, 1020.0]
        freq_words, ro_freq_words = readout_freq_words(
            soccfg,
            freqs,
            gen_ch=GEN_CH,
            ro_ch=RO_CH,
            mixer_freq=None,
            nqz=GEN_NQZ,
        )
        cfg = _readout_cfg(freq=freqs[0])
        prog = ModularProgramV2(
            soccfg,
            ProgramV2Cfg(),
            modules=[
                TablePulseReadout(
                    "ro",
                    cfg,
                    idx_reg="freq",
                    freq_words=freq_words,
                    ro_freq_words=ro_freq_words,
                )
            ],
            sweep=[("freq", len(freqs))],
        )

        ports = [
            (idx, inst)
            for idx, inst in enumerate(prog.prog_list)
            if inst.get("CMD") == "WPORT_WR"
        ]
        updates = [
            (idx, inst)
            for idx, inst in enumerate(prog.prog_list)
            if inst.get("CMD") == "WMEM_WR"
        ]
        assert len(ports) == 2
        assert all(inst.get("SRC") == "wmem" for _, inst in ports)
        assert len(updates) == 2
        assert min(idx for idx, _ in updates) > max(idx for idx, _ in ports)
        time_indices = [
            idx for idx, inst in enumerate(prog.prog_list) if inst.get("CMD") == "TIME"
        ]
        test_indices = [
            idx for idx, inst in enumerate(prog.prog_list) if inst.get("CMD") == "TEST"
        ]
        assert any(
            max(idx for idx, _ in ports) < idx < min(idx for idx, _ in updates)
            for idx in time_indices
        )
        assert max(idx for idx, _ in updates) < min(test_indices)
        assert not any("runtime" in wave.name for wave in prog.waves)
        assert not any(wave.conf & 0b010000 for wave in prog.waves)

        for pulse_name, expected_word in (
            ("ro_table_pulse", freq_words[0]),
            ("ro_adc", ro_freq_words[0]),
        ):
            wave_names = prog.pulses[pulse_name].get_wavenames(exclude_special=True)
            assert len(wave_names) == 1
            wave = prog.waves[prog.wave2idx[wave_names[0]]]
            raw_freq = wave.freq.start if hasattr(wave.freq, "start") else wave.freq
            assert int(raw_freq) % 2**32 == expected_word

        dmem = prog.binprog["dmem"]
        assert dmem is not None
        assert [int(word) % 2**32 for word in dmem] == (
            freq_words[1:] + freq_words[:1] + ro_freq_words[1:] + ro_freq_words[:1]
        )

    def test_table_pulse_readout_rejects_loop_count_mismatch(self):
        cfg = _readout_cfg(freq=GEN_FREQ)
        with pytest.raises(ValueError, match="does not match loop"):
            _make_prog(
                modules=[
                    TablePulseReadout(
                        "ro",
                        cfg,
                        idx_reg="freq",
                        freq_words=[1, 2],
                        ro_freq_words=[3, 4],
                    )
                ],
                sweep=[("freq", 3)],
                n_gens=1,
            )

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
        assert not lv.is_compressed

    def test_load_value_large_compressed(self):
        # 40 values → triggers auto_compress
        vals = self._scan_values_large()
        lv = LoadValue("lv", vals, idx_reg="myloop", val_reg="myval")
        r = Repeat("myloop", len(vals))
        r.add_content(lv)
        prog = _make_prog(modules=[r])
        assert prog.binprog is not None
        assert lv.is_compressed

    def test_load_value_negative_rejected(self):
        with pytest.raises(ValueError, match=r"\[0,"):
            LoadValue("lv", [-1, 0, 1], idx_reg="myloop", val_reg="myval")

    def test_compile_datamem_not_none(self):
        vals = self._scan_values_large()
        lv = LoadValue("lv", vals, idx_reg="myloop", val_reg="myval")
        r = Repeat("myloop", len(vals))
        r.add_content(lv)
        prog = _make_prog(modules=[r])
        assert prog.compile_datamem() is not None

    def test_load_word_compiles_uint32_words_as_int32_datamem(self):
        vals = [0, 2**31, 2**32 - 1]
        lw = LoadWord("lw", vals, idx_reg="myloop", val_reg="myword")
        r = Repeat("myloop", len(vals))
        r.add_content(lw)
        prog = _make_prog(modules=[r])

        dmem = prog.compile_datamem()

        assert dmem is not None
        assert dmem.dtype == np.int32
        assert dmem.tolist() == [0, -(2**31), -1]

    def test_load_word_binprog_uses_builtin_ints_for_transport(self):
        vals = [0, 2**31, 2**32 - 1]
        lw = LoadWord("lw", vals, idx_reg="myloop", val_reg="myword")
        r = Repeat("myloop", len(vals))
        r.add_content(lw)

        prog = _make_prog(modules=[r])

        dmem = prog.binprog["dmem"]
        assert dmem == [0, -(2**31), -1]
        assert all(type(word) is int for word in dmem)

    def test_binprog_rejects_non_vector_dmem(self):
        class TwoDimensionalDmemProgram(ModularProgramV2):
            def compile_datamem(self):
                return np.asarray([[1, 2]], dtype=np.int32)

        soccfg = make_mock_soccfg(n_gens=1, n_readouts=1)

        with pytest.raises(ValueError, match="dmem must be one-dimensional"):
            TwoDimensionalDmemProgram(soccfg, ProgramV2Cfg(), modules=[])

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


@pytest.mark.parametrize("mixer", [None, 500.0])
def test_pulse_runtime_adc_identity_keeps_distinct_qick_definitions(mixer):
    soccfg = make_mock_soccfg(n_gens=1, n_readouts=2)
    soccfg["gens"][0]["has_mixer"] = mixer is not None
    cfg = _pulse_cfg(freq=700.0)
    cfg.mixer_freq = mixer
    pulses = [Pulse(f"p{idx}", cfg, ro_ch=ch) for idx, ch in enumerate((None, 0, 0, 1))]
    prog = ModularProgramV2(soccfg, ProgramV2Cfg(), modules=pulses)
    assert prog.pulse_registry.count == 3
    assert pulses[1].pulse_id == pulses[2].pulse_id
    assert len({pulses[idx].pulse_id for idx in (0, 1, 3)}) == 3
    for pulse, ch in zip(pulses, (None, 0, 0, 1), strict=True):
        assert prog.pulses[pulse.pulse_id].params["ro_ch"] == ch


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


@pytest.mark.parametrize("sweep", [None, [("other", 2)]])
def test_table_readout_requires_named_loop(sweep):
    ro = TablePulseReadout(
        "table", _readout_cfg(), idx_reg="freq", freq_words=[1, 2], ro_freq_words=[3, 4]
    )
    with pytest.raises(
        ValueError, match="requires a sweep loop|exactly one sweep loop"
    ):
        _make_prog(modules=[ro], sweep=sweep)


def test_table_readout_does_not_share_fixed_pulse_template():
    cfg = _readout_cfg()
    table = TablePulseReadout(
        "table", cfg, idx_reg="freq", freq_words=[1, 2], ro_freq_words=[3, 4]
    )
    prog = _make_prog(
        modules=[Readout("fixed", cfg), table], sweep=[("freq", 2)], reps=2
    )
    pulse_names = [name for name in prog.pulses if not name.endswith("_adc")]
    assert len(pulse_names) == 2
    assert len({tuple(prog.pulses[name].get_wavenames()) for name in pulse_names}) == 2


def test_unprotected_routes_keep_qick_redeclaration_semantics():
    prog = _make_prog()
    prog.declare_gen(0, nqz=1)
    prog.declare_gen(0, nqz=2)
    prog.declare_readout(0, length=0.413)
    prog.declare_readout(0, length=0.827)
    assert prog.get_time_axis(0, length_only=True) == prog.us2cycles(0.827, ro_ch=0)


def test_protected_routes_report_owner_and_reject_overwrite():
    prog = _make_prog()
    prog.declare_gen(0, nqz=1)
    prog.declare_readout(0, length=0.413)
    assert hasattr(prog, "protect_readout_routes"), (
        "program must protect declared routes"
    )
    prog.protect_readout_routes("owner", 0, [0])
    with pytest.raises(ValueError, match="owner.*generator.*route.*0"):
        prog.declare_gen(0, nqz=2)
    with pytest.raises(ValueError, match="owner.*readout.*route.*0"):
        prog.declare_readout(0, length=0.827)


@pytest.mark.parametrize("declared", ["none", "generator"])
def test_route_protection_requires_successful_declarations(declared):
    prog = _make_prog()
    if declared == "generator":
        prog.declare_gen(0, nqz=1)
    assert hasattr(prog, "protect_readout_routes"), (
        "program must validate route protection"
    )
    with pytest.raises(ValueError, match="owner.*(generator|readout).*0"):
        prog.protect_readout_routes("owner", 0, [0])
