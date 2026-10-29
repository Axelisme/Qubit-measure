"""Composite declarations and runtime playback at the public compiler seam."""

from collections.abc import Callable, Sequence

import pytest
from qick import QickConfig
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import Module, PulseCfg, TableLengthPulse
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.readout import (
    MuxReadout,
    PulseReadoutCfg,
    TablePulseReadout,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2
from zcu_tools.program.v2.utils import readout_freq_words

from tests.program.v2._mux_support import mux_soccfg, native_mux
from tests.program.v2.timing_support import shots, waves


def _program(
    soccfg: QickConfig,
    modules: Sequence[Module],
    body: Callable[[TimelineProgramV2], None],
    initialize: Callable[[TimelineProgramV2], None] | None = None,
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            if initialize is not None:
                initialize(self)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            body(self)

    return Program(soccfg, reps=2, initial_delay=0, relax_delay=0.02)


@pytest.mark.parametrize("length, spacing", [(0.101, 50), (0.201, 91)])
def test_mux_end_covers_all_tones_and_adc_windows_plus_one_post(length, spacing):
    readout = MuxReadout(
        "mux",
        native_mux(
            length=length,
            pre_delay=0.01,
            post_delay=0.0005,
            ro_length=0.1,
            trig_offset=0.01,
        ),
    )

    def body(prog):
        prog.fork("default", into="read")
        prog.emit(readout, timeline="read")
        prog.emit(readout, timeline="read")

    prog = _program(mux_soccfg(), [readout], body)
    result = shots(prog)
    # Gen39/77 clocks at384MHz =>43.68/86.24 tProc cycles. Start rounds4.
    # Both ADCs round .1us to8 samples at76.8MHz =>44.8 cycles; end48.8.
    # ceil(max(47.68/90.24,48.8)+.21504) =50/91, no per-field ceil.
    assert [waves(shot) for shot in result] == [[4, spacing + 4]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        2 * spacing
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        2 * spacing + 9
    ] * 2
    assert [
        [(e.cycle - shot.start_cycle, e.value) for e in shot.events if e.kind == "data"]
        for shot in result
    ] == [[(4, 18), (14, 0), (spacing + 4, 18), (spacing + 14, 0)]] * 2
    assert list(prog.ro_chs) == [4, 1]
    assert prog.gen_chs[0]["mux_tones"][0]["gain_int"] > 0
    prog.compile()
    assert shots(prog) == result


def _table_readout() -> TablePulseReadout:
    soccfg = make_mock_soccfg(1, 1)
    freq_words, ro_words = readout_freq_words(
        soccfg, [1000.0, 1010.0, 1020.0], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
    )
    return TablePulseReadout(
        "table_read",
        PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.101),
            pre_delay=0.01,
            post_delay=0.0005,
            ro_length=0.1,
            trig_offset=0.01,
        ),
        idx_reg="frequency",
        freq_words=freq_words,
        ro_freq_words=ro_words,
    )


def test_table_readout_uses_loop_hook_and_wraps_paired_words_without_padding():
    readout = _table_readout()
    prog = _program(
        make_mock_soccfg(1, 1),
        [readout],
        lambda p: p.emit(readout),
        lambda p: p.add_loop("frequency", 3),
    )
    result = shots(prog)
    # ADC .1us rounds25 clocks at250MHz =>43.008cycles, pulse39=>43.68.
    # Config at0; trigger/pulse at4; ceil(4+43.68+.21504)=48.
    assert [waves(shot) for shot in result] == [[0, 4]] * 6
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [48] * 6
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [57] * 6
    # The loop owns paired updates after body closure, before the next point.
    writes = [i for i, inst in enumerate(prog.prog_list) if inst["CMD"] == "WMEM_WR"]
    body_end = next(
        i
        for i, inst in enumerate(prog.prog_list)
        if inst["P_ADDR"] == prog.timing_layout.body_end
    )
    assert len(writes) == 2
    assert min(writes) >= body_end
    dmem = prog.compile_datamem()
    assert dmem is not None
    assert [int(word) % 2**32 for word in dmem] == (
        readout.freq_words[1:]
        + readout.freq_words[:1]
        + readout.ro_freq_words[1:]
        + readout.ro_freq_words[:1]
    )
    assert [(loop.name, loop.count) for loop in prog.sweep_loops] == [
        ("reps", 2),
        ("frequency", 3),
    ]
    prog.compile()
    assert shots(prog) == result


def _pulse_cfg(length: float, *, ch: int = 0) -> PulseCfg:
    return PulseCfg(
        ch=ch,
        nqz=1,
        freq=1000,
        gain=0.5,
        waveform=ConstWaveformCfg(length=length),
        pre_delay=0.01,
        post_delay=0.0005,
    )


def test_computed_pulse_end_follows_selected_waveform_without_maximum_padding():
    pulse = ComputedPulse(
        "selected",
        val_reg="selector",
        pulses=[_pulse_cfg(39 / 384), _pulse_cfg(89 / 384)],
    )

    def body(prog):
        prog.emit(pulse)
        prog.emit(pulse)

    prog = _program(
        make_mock_soccfg(1, 0),
        [pulse],
        body,
        lambda p: p.add_loop("selector", 2),
    )
    # Actual ends: pre4 + generator43.68/99.68 + post.21504 =>ceil48/104.
    # Two emissions select the same index, but each starts at the previous end.
    result = shots(prog)
    assert [waves(shot) for shot in result] == [[4, 52], [4, 108]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [96, 208] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [105, 217] * 2
    prog.compile()
    assert shots(prog) == result


def test_computed_pulse_uses_selected_fractional_post_not_first_or_maximum():
    first = _pulse_cfg(39 / 384)
    second = first.model_copy(update={"post_delay": 0.001})
    pulse = ComputedPulse("selected_post", val_reg="selector", pulses=[first, second])
    prog = _program(
        make_mock_soccfg(1, 0),
        [pulse],
        lambda p: p.emit(pulse),
        lambda p: p.add_loop("selector", 2),
    )
    result = shots(prog)
    # Same waveform47.68 cycle end. Posts .21504/.43008 select ceil48/49.
    assert [waves(shot) for shot in result] == [[4]] * 4
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [48, 49] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [57, 58] * 2


def test_table_length_uses_actual_playback_end_and_counts_pre_post_once():
    pulse = TableLengthPulse(
        "length_table",
        _pulse_cfg(39 / 384),
        lengths=[39 / 384, 64 / 384, 89 / 384],
        idx_reg="index",
    )

    def body(prog):
        prog.emit(pulse)
        prog.emit(pulse)

    prog = _program(
        make_mock_soccfg(1, 0),
        [pulse],
        body,
        lambda p: p.add_loop("index", 3),
    )
    result = shots(prog)
    # ceil(pre4 + gen43.68/71.68/99.68 + post.21504) =48/76/104.
    assert [waves(shot) for shot in result] == [[4, 52], [4, 80], [4, 108]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        96,
        152,
        208,
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [105, 161, 217] * 2
    prog.compile()
    assert shots(prog) == result


@pytest.mark.parametrize("count", [None, 2])
def test_table_readout_rejects_missing_or_mismatched_public_loop(count):
    readout = _table_readout()

    def initialize(prog):
        if count is not None:
            prog.add_loop("frequency", count)

    with pytest.raises(ValueError, match="(sweep loop|does not match loop)"):
        _program(
            make_mock_soccfg(1, 1), [readout], lambda p: p.emit(readout), initialize
        )
