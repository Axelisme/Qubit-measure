"""Composite declarations and runtime playback at the public compiler seam."""

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from qick import QickConfig
from qick.asm_v2 import AsmInst, Macro, QickSweep1D, WriteReg
from zcu_tools.program.v2._timeline_bounds import CycleBounds
from zcu_tools.program.v2._timeline_selected import PlaybackChoice, SelectedPlayback
from zcu_tools.program.v2.macro.pluse_reg import PulseByReg
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import Module, Pulse, PulseCfg, TableLengthPulse
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.dmem import LoadValue
from zcu_tools.program.v2.modules.readout import (
    MuxReadout,
    PulseReadoutCfg,
    TablePulseReadout,
)
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
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


@pytest.mark.parametrize("length, spacing", [(0.101, 51), (0.201, 92)])
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
    # Scalar pre-delay .01 us ceils to 5 tProc ticks before adding playback/post.
    assert [waves(shot) for shot in result] == [[5, spacing + 5]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        2 * spacing
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        2 * spacing + 9
    ] * 2
    assert [
        [(e.cycle - shot.start_cycle, e.value) for e in shot.events if e.kind == "data"]
        for shot in result
    ] == [[(5, 18), (15, 0), (spacing + 5, 18), (spacing + 15, 0)]] * 2
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
    # ADC .1us rounds31 clocks at307.2MHz =>43.4 cycles, pulse39=>43.68.
    # Config at0; trigger/pulse at5; ceil(5+43.68+.21504)=49.
    assert [waves(shot) for shot in result] == [[0, 5]] * 6
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [49] * 6
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [58] * 6
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


def test_integer_candidate_selection_uses_only_author_waveform_table():
    pulse = ComputedPulse(
        "selected_integer",
        val_reg="selector",
        pulses=[
            _pulse_cfg(25 / 384).model_copy(update={"pre_delay": 0, "post_delay": 0}),
            _pulse_cfg(50 / 384).model_copy(update={"pre_delay": 0, "post_delay": 0}),
        ],
    )
    prog = _program(
        make_mock_soccfg(1, 0),
        [pulse],
        lambda p: p.emit(pulse),
        lambda p: p.add_loop("selector", 2),
    )
    result = shots(prog)
    assert [waves(s) for s in result] == [[0]] * 4
    assert [s.body_end_cycle - s.start_cycle for s in result] == [28, 56] * 2
    memory = prog.compile_datamem()
    # Only the author-declared two waveform addresses occupy data memory.
    assert memory is not None and len(memory) == 2
    assert set(int(word) for word in memory) <= set(prog.wave2idx.values())


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
    # First start is ceil(.01 * 430.08) = 5; selected spacing is 49 or 105.
    result = shots(prog)
    assert [waves(shot) for shot in result] == [[5, 54], [5, 110]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [98, 210] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [107, 219] * 2
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
    # Scalar pre-delay .01 us ceils to 5 tProc ticks before adding playback/post.
    assert [waves(shot) for shot in result] == [[5]] * 4
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [49, 50] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [58, 59] * 2


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
    # Scalar pre-delay .01 us ceils to 5 tProc ticks before adding playback/post.
    assert [waves(shot) for shot in result] == [[5, 54], [5, 82], [5, 110]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        98,
        154,
        210,
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [107, 163, 219] * 2
    prog.compile()
    assert shots(prog) == result


def _runtime_selected(kind: str, *, pre_delay: float = 0.01) -> Module:
    first = _pulse_cfg(39 / 384).model_copy(update={"pre_delay": pre_delay})
    second = _pulse_cfg(89 / 384).model_copy(update={"pre_delay": pre_delay})
    if kind == "computed":
        return ComputedPulse("selected", val_reg="index", pulses=[first, second])
    return TableLengthPulse(
        "selected", first, lengths=[39 / 384, 89 / 384], idx_reg="index"
    )


@pytest.mark.parametrize("kind", ["computed", "table"])
def test_shared_selected_pulse_keeps_each_programs_addresses_and_loader(kind):
    soccfg = make_mock_soccfg(1, 0)
    shared = _runtime_selected(kind)
    others = []

    def build(pulse, *, padded=False, interleave=False):
        def initialize(prog):
            prog.add_loop("index", 2)
            if padded:
                prog.add_dmem([91, 83, 79])
                prog.declare(Pulse("padding", _pulse_cfg(0.3)))

        def body(prog):
            if interleave:
                others.append(build(shared, padded=True))
            prog.emit(pulse)

        return _program(soccfg, [pulse], body, initialize)

    first = build(shared, interleave=True)
    fresh = build(_runtime_selected(kind))
    other_fresh = build(_runtime_selected(kind), padded=True)
    assert first.prog_list == fresh.prog_list
    assert shots(first) == shots(fresh)
    assert others[0].prog_list == other_fresh.prog_list
    np.testing.assert_array_equal(
        others[0].compile_datamem(), other_fresh.compile_datamem()
    )
    first.compile()
    assert first.prog_list == fresh.prog_list


@pytest.mark.parametrize("kind", ["computed", "table"])
def test_selected_branch_paths_inside_repeat_use_actual_ends(kind):
    pulse = _runtime_selected(kind)

    def initialize(prog):
        prog.add_loop("path", 2)
        prog.add_reg("index")

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("twice", 2):
            with prog.branch(by="path", align_timeline=False) as branch:
                for index in (0, 1):
                    with branch.case(index):
                        prog.write_reg("index", index)
                        prog.emit(pulse)
            prog.delay_by_timeline()

    prog = _program(make_mock_soccfg(1, 0), [pulse], body, initialize)
    result = shots(prog)
    assert [waves(s) for s in result] == [[5, 54], [5, 110]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [98, 210] * 2
    assert [s.end_cycle - s.start_cycle for s in result] == [107, 219] * 2


@pytest.mark.parametrize("kind", ["computed", "table"])
@pytest.mark.parametrize("index", [-1, 2, 2**31])
def test_selected_playback_requires_full_selector_bounds(kind, index):
    pulse = _runtime_selected(kind)

    def body(prog):
        prog.write_reg("index", index)
        prog.emit(pulse)
        prog.write_reg("index", 0)  # A reset cannot hide an invalid earlier lookup.

    def initialize(prog):
        prog.add_reg("index")

    with pytest.raises(ValueError, match="index"):
        _program(make_mock_soccfg(1, 0), [pulse], body, initialize)


@pytest.mark.parametrize("kind", ["computed", "table"])
def test_selected_runtime_origin_rejects_possible_signed_overflow(kind):
    pulse = _runtime_selected(kind)

    def initialize(prog):
        prog.add_loop("index", 2)
        prog.add_reg("origin")

    def body(prog):
        prog.write_reg("origin", 2**31 - 41)
        prog.timeline_inc("origin", timeline="default")
        prog.emit(pulse)

    with pytest.raises(ValueError, match="overflow"):
        _program(make_mock_soccfg(1, 0), [pulse], body, initialize)


@pytest.mark.parametrize("kind", ["computed", "table"])
@pytest.mark.parametrize("negative_pre", [False, True])
def test_selected_channel_overlap_cannot_borrow_coverage(kind, negative_pre):
    pulse = _runtime_selected(kind, pre_delay=-0.01 if negative_pre else 0.01)

    def body(prog):
        if negative_pre:
            prog.timeline_inc(100 / 430.08, timeline="default")
            prog.emit(pulse)
            prog.emit(pulse)
        else:
            prog.fork("default", into="other")
            prog.emit(pulse)
            prog.emit(pulse, timeline="other")

    with pytest.raises(ValueError, match="channel 0 pulse overlap"):
        _program(
            make_mock_soccfg(1, 0), [pulse], body, lambda p: p.add_loop("index", 2)
        )


def _negative_selected(kind: str, cycles: tuple[int, int] = (39, 64)) -> Module:
    candidates = [
        _pulse_cfg(n / 384).model_copy(update={"post_delay": -60 / 430.08})
        for n in cycles
    ]
    if kind == "computed":
        return ComputedPulse("negative", val_reg="index", pulses=candidates)
    return TableLengthPulse(
        "negative", candidates[0], lengths=[n / 384 for n in cycles], idx_reg="index"
    )


@pytest.mark.parametrize("kind", ["computed", "table"])
@pytest.mark.parametrize("runtime", [False, True])
def test_selected_negative_post_rounds_signed_offset_without_clamping_origin(
    kind, runtime
):
    pulse = _negative_selected(kind)

    def initialize(prog):
        prog.add_loop("index", 2)
        prog.add_reg("origin")

    def body(prog):
        prog.write_reg("origin", 100)
        prog.timeline_inc("origin" if runtime else 100 / 430.08, timeline="default")
        prog.emit(pulse)

    prog = _program(make_mock_soccfg(1, 0), [pulse], body, initialize)
    result = shots(prog)
    # Integer origin 100 plus ceil(.01 * 430.08) = 105.
    assert [waves(s) for s in result] == [[105]] * 4
    assert [s.body_end_cycle - s.start_cycle for s in result] == [89, 117] * 2


@pytest.mark.parametrize("kind", ["computed", "table"])
@pytest.mark.parametrize("wait", [False, True])
def test_selected_negative_post_retains_physical_channel_end(kind, wait):
    pulse = _negative_selected(kind, (39, 39))
    following = Pulse("following", _pulse_cfg(39 / 384))

    def initialize(prog):
        prog.add_loop("index", 2)
        prog.add_reg("origin")

    def body(prog):
        prog.write_reg("origin", 100)
        prog.timeline_inc("origin", timeline="default")
        prog.emit(pulse)
        if wait:
            prog.timeline_inc(60 / 430.08, timeline="default")
        prog.emit(following)

    if not wait:
        with pytest.raises(ValueError, match="channel 0 pulse overlap"):
            _program(make_mock_soccfg(1, 0), [pulse, following], body, initialize)
    else:
        prog = _program(make_mock_soccfg(1, 0), [pulse, following], body, initialize)
        assert [waves(s) for s in shots(prog)] == [[105, 154]] * 4
        assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [198] * 4


@pytest.mark.parametrize("kind", ["computed", "table"])
def test_selected_negative_post_native_auto_alignment_uses_exact_physical_end(kind):
    pulse = _negative_selected(kind)
    following = Pulse("following", _pulse_cfg(39 / 384))

    def initialize(prog):
        prog.add_loop("index", 2)
        prog.add_reg("origin")

    def body(prog):
        prog.write_reg("origin", 100)
        prog.timeline_inc("origin", timeline="default")
        prog.emit(pulse)
        prog.delay_auto(0, gens=True, ros=False)
        prog.emit(following)

    prog = _program(make_mock_soccfg(1, 0), [pulse, following], body, initialize)
    result = shots(prog)
    # Scalar pre-delay .01 us ceils to 5 tProc ticks before adding playback/post.
    assert [waves(s) for s in result] == [[105, 154], [105, 182]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [198, 226] * 2


@pytest.mark.parametrize("kind", ["computed", "table"])
@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("advances", [(60,), (30, 30), (59,), (30, -1, 30)])
def test_selected_negative_post_direct_advances_recover_only_proven_physical_boundary(
    kind, runtime, advances
):
    pulse = _negative_selected(kind)
    following = Pulse(
        "following", _pulse_cfg(39 / 384).model_copy(update={"pre_delay": 0.0})
    )

    def initialize(prog):
        prog.add_loop("index", 2)
        prog.add_reg("origin")

    def body(prog):
        prog.write_reg("origin", 100)
        prog.timeline_inc("origin" if runtime else 100 / 430.08, timeline="default")
        prog.emit(pulse)
        for cycles in advances:
            prog.timeline_inc(cycles / 430.08, timeline="default")
        prog.emit(following)

    if sum(advances) < 60:
        with pytest.raises(ValueError, match="channel 0 pulse overlap"):
            _program(make_mock_soccfg(1, 0), [pulse, following], body, initialize)
    else:
        prog = _program(make_mock_soccfg(1, 0), [pulse, following], body, initialize)
        result = shots(prog)
        assert [waves(s) for s in result] == [[105, 149], [105, 177]] * 2
        assert [s.body_end_cycle - s.start_cycle for s in result] == [193, 221] * 2


@pytest.mark.parametrize("kind", ["computed", "table"])
@pytest.mark.parametrize("delay_by_timeline", [False, True])
def test_deferred_coverage_does_not_cover_a_later_emission_or_reused_register(
    kind, delay_by_timeline
):
    pulse = _negative_selected(kind)
    following = Pulse(
        "following", _pulse_cfg(39 / 384).model_copy(update={"pre_delay": 0.0})
    )

    def initialize(prog):
        prog.add_loop("index", 2)
        prog.add_reg("origin")

    def body(prog):
        prog.write_reg("origin", 100)
        prog.timeline_inc("origin", timeline="default")
        prog.emit(pulse)
        prog.timeline_inc(60 / 430.08, timeline="default")
        prog.fork("default", into="stale")
        if delay_by_timeline:
            prog.delay_by_timeline()
        prog.emit(following)
        prog.emit(following, timeline="stale")

    with pytest.raises(ValueError, match="channel 0 pulse overlap"):
        _program(make_mock_soccfg(1, 0), [pulse, following], body, initialize)


@pytest.mark.parametrize("kind", ["computed", "table"])
def test_deferred_coverage_branch_does_not_hide_underpaid_path(kind):
    pulse = _negative_selected(kind)
    following = Pulse(
        "following", _pulse_cfg(39 / 384).model_copy(update={"pre_delay": 0.0})
    )

    def initialize(prog):
        prog.add_loop("index", 2)
        prog.add_loop("path", 2)
        prog.add_reg("origin")

    def body(prog):
        prog.write_reg("origin", 100)
        prog.timeline_inc("origin", timeline="default")
        prog.emit(pulse)
        with prog.branch(by="path", align_timeline=False) as branch:
            for index, amount in enumerate((60, 59)):
                with branch.case(index):
                    prog.timeline_inc(amount / 430.08)
        prog.emit(following)

    with pytest.raises(ValueError, match="channel 0 pulse overlap"):
        _program(make_mock_soccfg(1, 0), [pulse, following], body, initialize)


@pytest.mark.parametrize("kind", ["computed", "table"])
@pytest.mark.parametrize("runtime_selector", [False, True])
def test_deferred_coverage_through_runtime_maximum_does_not_pad_short_selection(
    kind, runtime_selector
):
    pulse = _negative_selected(kind)
    following = Pulse(
        "following", _pulse_cfg(39 / 384).model_copy(update={"pre_delay": 0.0})
    )

    def initialize(prog):
        prog.add_loop("scan" if runtime_selector else "index", 2)
        if runtime_selector:
            prog.add_reg("index")
        prog.add_reg("origin")

    def body(prog):
        if runtime_selector:
            # Bitwise author arithmetic has a certified range, not known points.
            with prog.time_producer("index", CycleBounds(0, 1)):
                prog.write_reg_op("index", "scan", "AND", 1)
        prog.write_reg("origin", 100)
        prog.timeline_inc("origin", timeline="default")
        prog.fork("default", into="other")
        prog.emit(pulse)
        prog.join("other", into="default")
        prog.timeline_inc(60 / 430.08, timeline="default")
        prog.emit(following)

    prog = _program(make_mock_soccfg(1, 0), [pulse, following], body, initialize)
    result = shots(prog)
    # max(89/117,100) +60 =>160/177. No sweep-wide max177.
    assert [waves(s) for s in result] == [[105, 160], [105, 177]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [204, 221] * 2


@pytest.mark.parametrize("kind", ["computed", "table"])
def test_deferred_coverage_repeat_backedge_and_next_shot_reload_selection(kind):
    pulse = _negative_selected(kind)

    def initialize(prog):
        prog.add_loop("index", 2)
        prog.add_reg("origin")

    def body(prog):
        prog.delay_by_timeline()
        with prog.repeat("twice", 2):
            prog.write_reg("origin", 100)
            prog.timeline_inc("origin", timeline="default")
            prog.emit(pulse)
            prog.timeline_inc(60 / 430.08, timeline="default")
            prog.delay_by_timeline()

    prog = _program(make_mock_soccfg(1, 0), [pulse], body, initialize)
    result = shots(prog)
    assert [waves(s) for s in result] == [[105, 254], [105, 282]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [298, 354] * 2
    prog.compile()
    assert shots(prog) == result


def _flat_selected(kind: str) -> Module:
    def cfg(cycles):
        return _pulse_cfg((cycles + 16) / 384).model_copy(
            update={
                "waveform": FlatTopWaveformCfg(
                    length=(cycles + 16) / 384,
                    raise_waveform=GaussWaveformCfg(length=16 / 384, sigma=4 / 384),
                )
            }
        )

    if kind == "computed":
        return ComputedPulse("flat", val_reg="index", pulses=[cfg(39), cfg(89)])
    return TableLengthPulse(
        "flat", cfg(39), lengths=[55 / 384, 105 / 384], idx_reg="index"
    )


@pytest.mark.parametrize("kind", ["computed", "table"])
def test_flat_selected_runtime_origin_repeat_backedge_and_shot_reset(kind):
    pulse = _flat_selected(kind)

    def initialize(prog):
        prog.add_reg("origin")
        prog.add_loop("index", 2)

    def body(prog):
        prog.write_reg("origin", 7)
        prog.timeline_inc("origin", timeline="default")
        prog.emit(pulse)
        prog.delay_by_timeline()
        with prog.repeat("twice", 2):
            prog.emit(pulse)
            prog.delay_by_timeline()

    prog = _program(make_mock_soccfg(1, 0), [pulse], body, initialize)
    result = shots(prog)
    # Origin 7 plus pre-delay 5 starts at 12; selected spacing is 67 or 123.
    assert [waves(s) for s in result] == [
        [12] * 3 + [79] * 3 + [146] * 3,
        [12] * 3 + [135] * 3 + [258] * 3,
    ] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [208, 376] * 2
    assert [s.end_cycle - s.start_cycle for s in result] == [217, 385] * 2
    prog.compile()
    assert shots(prog) == result


def test_selected_parallel_join_keeps_crossing_pointwise_ends_and_runtime_origin():
    left = TableLengthPulse(
        "left",
        _pulse_cfg(39 / 384),
        lengths=[39 / 384, 64 / 384, 89 / 384],
        idx_reg="index",
    )
    right = TableLengthPulse(
        "right",
        _pulse_cfg(39 / 384, ch=1),
        lengths=[89 / 384, 64 / 384, 39 / 384],
        idx_reg="index",
    )

    def initialize(prog):
        prog.add_loop("index", 3)
        prog.add_reg("origin")

    def body(prog):
        prog.write_reg("origin", 7)
        prog.timeline_inc("origin", timeline="default")
        prog.fork("default", into=["left", "right"])
        prog.emit(left, timeline="left")
        prog.emit(right, timeline="right")
        prog.join("left", "right", into="default")

    prog = _program(make_mock_soccfg(2, 0), [left, right], body, initialize)
    result = shots(prog)
    # Scalar pre-delay .01 us ceils to 5 tProc ticks before adding playback/post.
    assert [waves(s) for s in result] == [[12, 12]] * 6
    assert [s.body_end_cycle - s.start_cycle for s in result] == [112, 84, 112] * 2
    assert [s.end_cycle - s.start_cycle for s in result] == [121, 93, 121] * 2


@pytest.mark.parametrize("constant_second", [False, True])
def test_arbitrary_selected_posts_keep_independent_native_sweep_grid(constant_second):
    first = _pulse_cfg(39 / 384).model_copy(
        update={"post_delay": QickSweep1D("post", 0.0005, 0.0015)}
    )
    second = _pulse_cfg(39 / 384).model_copy(
        update={
            "post_delay": 0.0015
            if constant_second
            else QickSweep1D("post", 0.0015, 0.0005)
        }
    )
    pulse = ComputedPulse("swept_posts", val_reg="selector", pulses=[first, second])

    def initialize(prog):
        prog.add_reg("selector")
        prog.add_loop("post", 3)

    def body(prog):
        prog.write_reg("selector", 0)
        prog.emit(pulse)
        prog.write_reg("selector", 1)
        prog.emit(pulse)

    prog = _program(make_mock_soccfg(1, 0), [pulse], body, initialize)
    result = shots(prog)
    # Native subcycle post steps truncate to zero in both input branches.
    assert [waves(s) for s in result] == [[5, 54], [5, 54], [5, 54]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in result] == [99, 99, 99] * 2


class _OverwriteSelected(Macro):
    def translate(self, prog):
        AsmInst(
            inst={
                "CMD": "REG_WR",
                "DST": prog._get_reg("length"),
                "SRC": "imm",
                "LIT": "#89",
            },
            addr_inc=1,
        ).translate(prog)


def _change_selection(prog, loader, other, change):
    if change == "constant":
        prog.write_reg("length", 89)
    elif change == "other_table":
        prog.emit(other)
    elif change == "selector":
        prog.write_reg("index", 1)
    elif change == "translate":
        prog.append_macro(_OverwriteSelected())
    elif change == "constant_branch":
        with prog.branch(by="path", align_timeline=True) as branch:
            with branch.case(0):
                prog.write_reg("length", 89)
            with branch.case(1):
                pass
    elif change == "constant_backedge":
        prog.delay_by_timeline()
        with prog.repeat("overwrite", 2):
            prog.write_reg("length", 89)
            prog.delay_by_timeline()
    elif change == "constant_reset":
        prog.write_reg("length", 89)
        loader.lower_selected(prog)


@pytest.mark.parametrize("kind", ["computed", "table"])
@pytest.mark.parametrize("flat", [False, True])
def test_selected_actual_playback_words_match_independent_generator_lengths(kind, flat):
    pulse = _flat_selected(kind) if flat else _runtime_selected(kind)

    class Observed(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_reg("output", addr=15)
            self.add_loop("index", 2)
            self.declare(pulse)

        def _body(self, cfg):
            self.emit(pulse)

        def record_word(self, register):
            self.write_reg("output", register)
            self.asm_inst(
                {
                    "CMD": "DPORT_WR",
                    "DST": "0",
                    "SRC": "reg",
                    "DATA": "r15",
                    "TIME": "@0",
                }
            )

        def pulse_by_reg(self, ch, addr_reg, *args, **kwargs):
            super().pulse_by_reg(ch, addr_reg, *args, **kwargs)
            self.record_word(addr_reg)

        def pulse_from_length_reg(self, ch, name, *, length_reg, **kwargs):
            super().pulse_from_length_reg(ch, name, length_reg=length_reg, **kwargs)
            self.record_word(length_reg)
            self.record_word("w4")

    prog = Observed(make_mock_soccfg(1, 0), reps=2, initial_delay=0, relax_delay=0)
    words = [[e.value for e in shot.events if e.kind == "data"] for shot in shots(prog)]
    if kind == "table":
        # Observe both the true producer and the patched waveform field consumed
        # by WPORT. The timing table alone could not establish this promise.
        assert words == [[39, 39], [89, 89]] * 2
    else:
        lengths = []
        for row in words:
            address = row[0]
            assert address is not None
            lengths.append(
                sum(
                    int(w.length)
                    for w in prog.waves[address : address + (3 if flat else 1)]
                )
            )
        assert lengths == ([55, 105] if flat else [39, 89]) * 2


@pytest.mark.parametrize("flat", [False, True])
def test_native_address_playback_keeps_deferred_register_resolution(flat):
    pulse = (
        _flat_selected("computed") if flat else Pulse("native", _pulse_cfg(39 / 384))
    )

    def body(prog):
        prog.write_reg("late", 0)
        prog.pulse_by_reg(0, "late", 0.01, flat_top_pulse=flat)
        prog.add_reg("late")

    prog = _program(make_mock_soccfg(1, 0), [pulse], body)
    assert [waves(shot) for shot in shots(prog)] == [[4] * (3 if flat else 1)] * 2


class _FlatAddressObservation(PulseByReg):
    corrupt: bool = False

    def translate(self, prog):
        # A native writer at the public macro boundary, after preparation.
        if self.corrupt:
            WriteReg(dst=self.addr_regs[1], src=0).translate(prog)
        for register in self.addr_regs:
            WriteReg(dst="output", src=register).translate(prog)
            AsmInst(
                inst={
                    "CMD": "DPORT_WR",
                    "DST": "0",
                    "SRC": "reg",
                    "DATA": "r15",
                    "TIME": "@0",
                },
                addr_inc=1,
            ).translate(prog)
        super().translate(prog)


@pytest.mark.parametrize(
    "managed, corrupt", [(True, False), (True, True), (False, False)]
)
def test_flat_top_actual_segment_address_producers_are_preserved_or_reject_overwrite(
    managed, corrupt
):
    pulse = _flat_selected("computed")

    class Observed(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_reg("output", addr=15)
            self.add_loop("index", 2)
            self.declare(pulse)
            self.expected_segments = [
                [self.wave2idx[wave] for wave in self.list_pulse_waveforms(name)]
                for name in self.pulses
            ]
            self.loader = LoadValue(
                "native_addresses",
                [row[0] for row in self.expected_segments],
                idx_reg="index",
                val_reg="native_base",
            )
            if not managed:
                self.declare(self.loader)

        def _body(self, cfg):
            if managed:
                self.emit(pulse)
            else:
                self.emit(self.loader)
                self.pulse_by_reg(0, "native_base", t=0.01, flat_top_pulse=True)

        def append_macro(self, macro):
            if type(macro) is PulseByReg:
                macro = _FlatAddressObservation(
                    ch=macro.ch,
                    t=macro.t,
                    addr_regs=list(macro.addr_regs),
                    flat_top_pulse=getattr(macro, "flat_top_pulse", False),
                    corrupt=corrupt,
                )
            super().append_macro(macro)

    try:
        prog = Observed(make_mock_soccfg(1, 0), reps=2, initial_delay=0, relax_delay=0)
    except ValueError as exc:
        assert corrupt and "unregistered compiler time write at instruction" in str(exc)
        return
    result = shots(prog)
    actual = [
        [event.value for event in shot.events if event.kind == "data"]
        for shot in result
    ]
    # Candidate order and public declared wmem resources supply the expected
    # addresses. The observer never prepares or repairs a segment address.
    assert actual == prog.expected_segments * 2
    assert [waves(shot) for shot in result] == [[(5 if managed else 4)] * 3] * 4
    if managed:
        assert [s.body_end_cycle - s.start_cycle for s in result] == [67, 123] * 2
    lengths = []
    for row in actual:
        assert all(address is not None for address in row)
        lengths.append(
            [int(prog.waves[address].length) for address in row if address is not None]
        )
    assert lengths == [[8, 39, 8], [8, 89, 8]] * 2
    writes = [i for i, inst in enumerate(prog.prog_list) if inst["CMD"] == "WPORT_WR"]
    assert writes == list(range(writes[0], writes[0] + 3))
    assert len({prog.prog_list[i].get("TIME") for i in writes}) == 1


def _selected_length_with_change(change: str) -> TimelineProgramV2:
    template = Pulse("template", _pulse_cfg(39 / 384), pulse_id="template")
    loader = LoadValue("lengths", [39, 89], idx_reg="index", val_reg="length")
    other = LoadValue(
        "other", [89, 39], idx_reg="index", val_reg="length", use_existed=True
    )

    class Selection(Module):
        name = "selection"

        def init(self, prog):
            raise NotImplementedError("native-only fixture")

        def run(self, prog, t=0.0):
            raise NotImplementedError("native-only fixture")

        def declare(self, prog):
            if change == "constant_branch":
                prog.add_loop("path", 2)
            prog.add_reg("index")
            prog.declare(template)
            prog.declare(loader)
            prog.declare(other)

        def lower(self, prog, *, timeline, tag):
            prog.write_reg("index", 0)
            relation = loader.lower_selected(prog)
            _change_selection(prog, loader, other, change)
            prog.pulse_from_length_reg(
                0,
                "template",
                length_reg="length",
                flat_top=False,
                t=0.01,
                timeline="default",
                selection=SelectedPlayback(
                    relation.selector,
                    "length",
                    (
                        PlaybackChoice("template", length_cycles=39),
                        PlaybackChoice("template", length_cycles=89),
                    ),
                    relation,
                ),
            )
            if change == "selector_after_playback":
                prog.write_reg("index", 1)
            prog.timeline_inc(0.0005, timeline="default")

    return _selection_program(Selection())


def _selection_program(selection: Module) -> TimelineProgramV2:
    return _program(make_mock_soccfg(1, 0), [selection], lambda p: p.emit(selection))


@pytest.mark.parametrize(
    "change",
    [
        "constant",
        "other_table",
        "translate",
        "constant_branch",
        "constant_backedge",
        "constant_reset",
    ],
)
def test_selected_playback_rejects_in_range_writes_without_table_relation(change):
    with pytest.raises(ValueError, match=r"length: .*selection.*instruction"):
        _selected_length_with_change(change)


@pytest.mark.parametrize("change", ["selector", "selector_after_playback"])
def test_selected_playback_captures_index_before_later_in_range_change(change):
    prog = _selected_length_with_change(change)
    # The first length remains39 despite the author changing index to1.
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [49] * 2


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
