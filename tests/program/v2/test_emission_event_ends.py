"""Actual waveform/ADC ends govern module emission at the public compiler seam."""

from collections.abc import Callable, Sequence

import pytest
from qick.asm_v2 import Macro, QickParam, QickSpan, QickSweep1D, WriteReg
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import LoadValue, Module, Pulse, PulseCfg
from zcu_tools.program.v2.modules.delay import Delay, SoftDelay
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program.v2.timing_support import shots, waves


def _pulse_cfg(
    ch: int = 0,
    *,
    length: float | QickParam = 0.101,
    pre: float | QickParam = 0.01,
    post: float | QickParam = 0.0005,
) -> PulseCfg:
    return PulseCfg(
        ch=ch,
        nqz=1,
        freq=1000,
        gain=0.5,
        waveform=ConstWaveformCfg(length=length),
        pre_delay=pre,
        post_delay=post,
    )


def _program(
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

    return Program(make_mock_soccfg(3, 1), reps=2, initial_delay=0, relax_delay=0.02)


@pytest.mark.parametrize("post, spacing", [(0.0005, 48), (0.001, 49)])
@pytest.mark.parametrize("block_mode", [False, True])
@pytest.mark.parametrize("timeline", [None, "drive"])
def test_pulse_ceil_covers_actual_playback_plus_post_once(
    post, spacing, block_mode, timeline
):
    pulse = Pulse("drive", _pulse_cfg(post=post), block_mode=block_mode)

    def body(prog):
        if timeline is not None:
            prog.fork("default", into=timeline)
        prog.emit(pulse, timeline=timeline, tag="first")
        prog.emit(pulse, timeline=timeline, tag="second")

    prog = _program([pulse], body)
    # 39 generator clocks at 384 MHz = 43.68 tProc cycles.
    # pre rounds to 4. Posts are .21504/.43008 cycles: final ceil is 48/49.
    # Separately ceiling end and post gives 49 in both cases (wrong for .0005).
    # Nearest rounding 48.11008 gives 48, omitting part of the .001 post.
    # Rule a instead ceils pre/playback/post separately: 5+44+1=50.
    result = shots(prog)
    assert [waves(shot) for shot in result] == [[4, spacing + 4]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        2 * spacing
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        2 * spacing + 9
    ] * 2
    assert prog.get_time_param("second", "t") * 430.08 == pytest.approx(spacing + 4)
    prog.compile()
    assert shots(prog) == result


class _AllocateDuration(Macro):
    """Declare a native QICK register at preprocessing, before its writes."""

    def preprocess(self, prog):
        prog.add_reg("duration")

    def translate(self, prog):
        pass


@pytest.mark.parametrize("delay_type", [Delay, SoftDelay])
def test_fractional_delay_from_register_origin_rounds_up_without_reference_advance(
    delay_type,
):
    delay = delay_type("gap", 0.0005)

    def body(prog):
        prog.write_reg("duration", 40)
        prog.delay_timeline("duration")
        prog.emit(delay)

    result = shots(
        _program([delay], body, lambda prog: prog.append_macro(_AllocateDuration()))
    )
    # The register origin is an integer40 cycles; ceil(40+.21504)=41.
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [41] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [50] * 2


@pytest.mark.parametrize("swept_post", [False, True])
@pytest.mark.parametrize("kind", ["pulse", "readout"])
def test_runtime_origin_keeps_fractional_event_tail_until_post_is_added(
    kind, swept_post
):
    post = QickSweep1D("scan", 0.0005, 0.0005 + 4 / 430.08) if swept_post else 0.0005
    if kind == "pulse":
        owner = Pulse("drive", _pulse_cfg(post=post))
    else:
        owner = PulseReadout(
            "read",
            PulseReadoutCfg(
                gen_ch=0,
                nqz=1,
                freq=1000,
                gain=0.5,
                waveform=ConstWaveformCfg(length=0.051),
                pre_delay=0.01,
                post_delay=post,
                ro_ch=0,
                ro_length=0.1,
                trig_offset=0.01,
            ),
        )

    def initialize(prog):
        prog.append_macro(_AllocateDuration())
        if swept_post:
            prog.add_loop("scan", 3)

    def body(prog):
        prog.write_reg("duration", 40)
        prog.delay_timeline("duration")
        prog.emit(owner)

    result = shots(_program([owner], body, initialize))
    # Integer origin40 + ceil(pre4 + tail43.68/43.4 + post.21504)=88.
    # A swept post adds exactly0/2/4 cycles; separate tail/post ceilings give89.
    # Rule a at this origin gives90 for Pulse and89 for ADC-dominant Readout.
    ends = [88, 90, 92] if swept_post else [88]
    assert [waves(shot) for shot in result] == (
        [[40, 44]] if kind == "readout" else [[44]]
    ) * (2 * len(ends))
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == ends * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        end + 9 for end in ends
    ] * 2
    if kind == "readout":
        assert [
            [
                (event.cycle - shot.start_cycle, event.value)
                for event in shot.events
                if event.kind == "data"
            ]
            for shot in result
        ] == [[(44, 1), (54, 0)]] * (2 * len(ends))


@pytest.mark.parametrize("swept_post", [False, True])
def test_affine_waveform_and_post_sweeps_ceil_complete_end(swept_post):
    post = QickSweep1D("scan", 0.0005, 0.0005 + 4 / 430.08) if swept_post else 0.0005
    pulse = Pulse(
        "drive",
        _pulse_cfg(length=QickSweep1D("scan", 39 / 384, 89 / 384), post=post),
    )

    def body(prog):
        prog.emit(pulse)
        prog.emit(pulse)

    prog = _program([pulse], body, lambda prog: prog.add_loop("scan", 3))
    # Registered lengths39,64,89 generator clocks =>43.68,71.68,99.68.
    # pre=4, post=.21504 (+0/2/4 when swept): ends48,76,104 or48,78,108.
    # Rule a uses pre5, waveform start44+steps28, post1(+0/2/4):50,78,106
    # or50,80,110; it retains two extra cycles per emission.
    ends = [48, 78, 108] if swept_post else [48, 76, 104]
    result = shots(prog)
    assert prog.compile_datamem() is None
    assert [waves(shot) for shot in result] == [[4, end + 4] for end in ends] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        2 * end for end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        2 * end + 9 for end in ends
    ] * 2


def test_nonaffine_sweep_ceil_uses_exact_registered_playback_at_each_point():
    pulse = Pulse(
        "drive",
        _pulse_cfg(length=QickSweep1D("scan", 0.101, 0.301), pre=0, post=0),
        pulse_id="drive",
    )
    tail = Pulse("tail", _pulse_cfg(1, pre=0, post=0))

    def body(prog):
        prog.emit(pulse)
        prog.emit(tail)

    prog = _program([pulse, tail], body, lambda prog: prog.add_loop("scan", 3))
    # QICK resolves 39,77,115 generator clocks. At clock ratio28/25,
    # ends43.68,86.24,128.8 ceil to44,87,129, not affine44,87,130.
    assert prog.get_pulse_param("drive", "total_length", as_array=True) * 384 == (
        pytest.approx([39, 77, 115])
    )
    result = shots(prog)
    assert [waves(shot) for shot in result] == [[0, 44], [0, 87], [0, 129]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        88,
        131,
        173,
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [97, 140, 182] * 2
    assert prog.compile_datamem() is not None
    prog.compile()
    assert shots(prog) == result


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("swept_post", [False, True])
def test_nonaffine_post_and_repeated_emissions_keep_exact_same_channel_spacing(
    descending, swept_post
):
    start, stop = (115 / 384, 39 / 384) if descending else (39 / 384, 115 / 384)
    post = QickSweep1D("scan", 0.0002, 0.0002 + 0.1 / 430.08) if swept_post else 0.0002
    pulse = Pulse(
        "drive", _pulse_cfg(length=QickSweep1D("scan", start, stop), pre=0, post=post)
    )

    def body(prog):
        prog.emit(pulse)
        prog.emit(pulse)

    prog = _program([pulse], body, lambda prog: prog.add_loop("scan", 3))
    # Ends43.68,86.24,128.8 plus scalar.086016 or swept.086016/.136016/.186016.
    # Descending playback reverses these ends, not the increasing post.
    ends = [129, 87, 44] if descending else [44, 87, 129]
    result = shots(prog)
    assert [waves(shot) for shot in result] == [[0, end] for end in ends] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        2 * end for end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        2 * end + 9 for end in ends
    ] * 2


def test_multi_axis_nonaffine_sweep_reenters_nested_repeat_at_the_same_point():
    pulse = Pulse(
        "drive",
        _pulse_cfg(
            length=39 / 384 + QickSpan("scan", 76 / 384) + QickSpan("fine", 2 / 384),
            pre=0,
            post=0,
        ),
    )

    def initialize(prog):
        prog.add_loop("scan", 3)
        prog.add_loop("fine", 3)

    def body(prog):
        with prog.repeat("outer", 2), prog.repeat("inner", 2):
            prog.emit(pulse)

    prog = _program([pulse], body, initialize)
    # Generator grids39/40/41,77/78/79,115/116/117 at ratio28/25.
    # Each native sweep point executes four exact local bodies, not four
    # advances of a shared sequential dmem cursor.
    ends = [44, 45, 46, 87, 88, 89, 129, 130, 132]
    result = shots(prog)
    assert [waves(shot) for shot in result] == [
        [0, end, 2 * end, 3 * end] for end in ends
    ] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        4 * end for end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        4 * end + 9 for end in ends
    ] * 2


@pytest.mark.parametrize("swept_post", [False, True])
@pytest.mark.parametrize("length, end", [(0.051, 48), (0.201, 91)])
def test_readout_post_follows_maximum_of_generator_and_adc(length, end, swept_post):
    readout = PulseReadout(
        "read",
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=length),
            pre_delay=0.01,
            post_delay=QickSweep1D("scan", 0.0005, 0.0005 + 4 / 430.08)
            if swept_post
            else 0.0005,
            ro_ch=0,
            ro_length=0.1,
            trig_offset=0.01,
        ),
    )
    tail = Pulse("tail", _pulse_cfg(2, pre=0, post=0))

    def body(prog):
        prog.emit(readout)
        prog.emit(tail)

    prog = _program(
        [readout, tail],
        body,
        (lambda prog: prog.add_loop("scan", 3)) if swept_post else None,
    )
    result = shots(prog)
    ends = [end, end + 2, end + 4] if swept_post else [end]
    # ADC: 31/307.2 us = 43.4 cycles, starting at4 =>47.4.
    # Generator: nearest(.051/.201 *384)=20/77 clocks =>22.4/86.24 cycles.
    # ceil(max(26.4/90.24,47.4)+.21504) =>48/91.
    # Rule a: ADC ceil(ceil(offset)=5 +43.4)=49; pulse 29/94.
    # Rule-a owner max=49/94; it did not add post after ADC max.
    assert [waves(shot) for shot in result] == [
        [0, 4, point_end] for point_end in ends
    ] * 2
    assert [
        [
            (event.cycle - shot.start_cycle, event.value)
            for event in shot.events
            if event.kind == "data"
        ]
        for shot in result
    ] == [[(4, 1), (14, 0)]] * (2 * len(ends))
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        point_end + 44 for point_end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        point_end + 53 for point_end in ends
    ] * 2


def test_nonaffine_reference_debit_and_runtime_max_use_selected_point_not_padding():
    pulse = Pulse(
        "drive",
        _pulse_cfg(length=QickSweep1D("scan", 39 / 384, 115 / 384), pre=0, post=0),
    )
    tail = Pulse("tail", _pulse_cfg(2, pre=0, post=0))
    loader = LoadValue("other", [60, 20, 100], "scan", "other_end")

    def body(prog):
        prog.fork("default", into=["drive", "other"])
        prog.emit(pulse, timeline="drive")
        prog.emit(loader)
        prog.delay_timeline("other_end", timeline="other")
        prog.delay(50 / 430.08)
        prog.join("drive", "other", into="default")
        prog.emit(tail)

    prog = _program([pulse, tail, loader], body, lambda prog: prog.add_loop("scan", 3))
    result = shots(prog)
    # max(0,44/87/129-50) versus max(0,60/20/100-50) is10/37/79.
    assert [waves(shot) for shot in result] == [[0, 60], [0, 87], [0, 129]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        104,
        131,
        173,
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [113, 140, 182] * 2


class _OverwriteSweepSelector(Macro):
    def translate(self, prog):
        # The selector controls sweep iteration counts and therefore the table's
        # bounds proof, even though the index is a separate native sweep.
        WriteReg(dst="scan", src=0).translate(prog)


def test_nonaffine_lookup_rejects_unregistered_sweep_selector_writes():
    pulse = Pulse(
        "drive",
        _pulse_cfg(length=QickSweep1D("scan", 39 / 384, 115 / 384), pre=0, post=0),
    )

    def body(prog):
        prog.emit(pulse)
        prog.append_macro(_OverwriteSweepSelector())

    with pytest.raises(ValueError, match="scan.*unregistered.*instruction"):
        _program([pulse], body, lambda prog: prog.add_loop("scan", 3))


def test_nonaffine_boundary_does_not_certify_a_negative_same_channel_offset():
    pulse = Pulse(
        "drive",
        _pulse_cfg(length=QickSweep1D("scan", 39 / 384, 115 / 384), pre=0, post=0),
        pulse_id="drive",
    )

    def body(prog):
        prog.emit(pulse)
        prog.pulse(0, "drive", t=-1 / 430.08, timeline="default")

    # At the first point,44-1 starts before actual playback ends at43.68.
    with pytest.raises(ValueError, match="channel 0 pulse overlap"):
        _program([pulse], body, lambda prog: prog.add_loop("scan", 3))


def test_sweep_end_requires_a_discovered_native_axis_before_using_its_bounds():
    with pytest.raises(ValueError, match="missing.*native sweep axis"):
        _program(
            [], lambda prog: prog.delay_timeline(QickSweep1D("missing", 0.101, 0.301))
        )


def test_exact_sweep_delay_can_fill_the_remaining_dmem_capacity():
    def initialize(prog):
        prog.add_loop("scan", 3)
        prog.add_dmem([0] * 2045)

    prog = _program(
        [],
        lambda prog: prog.delay_timeline(
            QickSweep1D("scan", 43.68 / 430.08, 128.8 / 430.08)
        ),
        initialize,
    )
    result = shots(prog)
    memory = prog.compile_datamem()
    assert memory is not None
    assert len(memory) == 2048
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        44,
        87,
        129,
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [53, 96, 138] * 2


def test_crossing_fractional_generator_and_adc_ends_add_post_before_ceil():
    readout = PulseReadout(
        "read",
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=QickSweep1D("scan", 40 / 384, 116 / 384)),
            pre_delay=0,
            post_delay=0.0002,
            ro_ch=0,
            ro_length=0.2,
            trig_offset=0,
        ),
    )
    tail = Pulse("tail", _pulse_cfg(2, pre=0, post=0))

    def body(prog):
        prog.emit(readout)
        prog.emit(tail)

    result = shots(
        _program([readout, tail], body, lambda prog: prog.add_loop("scan", 3))
    )
    # ADC61 clocks gives85.4 tProc cycles. Generator44.8/87.36/129.92
    # crosses that end. Add post.086016 before rounding:86/88/131.
    # Premature conversion of this sweep rounds start45 and step42;
    # adding post to that integer max instead gives130 at the last point.
    assert [waves(shot) for shot in result] == [[0, 0, 86], [0, 0, 88], [0, 0, 131]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        130,
        132,
        175,
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [139, 141, 184] * 2


def test_nonaffine_table_capacity_error_identifies_the_emission_and_required_words():
    pulse = Pulse(
        "drive",
        _pulse_cfg(length=QickSweep1D("scan", 39 / 384, 115 / 384), pre=0, post=0),
    )

    def initialize(prog):
        prog.add_loop("scan", 3)
        prog.add_dmem([0] * 2047)

    with pytest.raises(
        ValueError, match=r"drive emission 0.*requires 3 dmem words.*1 available"
    ):
        _program([pulse], lambda prog: prog.emit(pulse), initialize)


@pytest.mark.parametrize("kind", ["two", "bath"])
def test_nonaffine_composite_aggregate_and_dependent_placement(kind):
    cavity = _pulse_cfg(
        length=QickSweep1D("scan", 39 / 384, 115 / 384), pre=0, post=0.0002
    )
    qubit = _pulse_cfg(1, length=77 / 384, pre=0, post=0.0002)
    reset = (
        TwoPulseResetCfg(pulse1_cfg=cavity, pulse2_cfg=qubit)
        if kind == "two"
        else BathResetCfg(
            cavity_tone_cfg=cavity,
            qubit_tone_cfg=qubit,
            pi2_cfg=_pulse_cfg(0, length=20 / 384, pre=0, post=0.0002),
        )
    ).build("reset")

    prog = _program(
        [reset], lambda prog: prog.emit(reset), lambda prog: prog.add_loop("scan", 3)
    )
    result = shots(prog)
    # Cavity ends43.68/86.24/128.8 plus.086016 =>44/87/129.
    # Qubit ends86.24+.086016 =>87. Pi2 placement is the ceiled
    # cavity boundary; length22.4+.086016 adds23, once per placement.
    cycles = [[0, 0]] * 3 if kind == "two" else [[0, 0, end] for end in [44, 87, 129]]
    ends = [87, 87, 129] if kind == "two" else [87, 110, 152]
    assert [waves(shot) for shot in result] == cycles * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == ends * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        end + 9 for end in ends
    ] * 2


@pytest.mark.parametrize("origin", [0, 40])
@pytest.mark.parametrize(
    "kind, expected_waves, end",
    [
        ("pulse", [4], 48),
        ("two", [4, 9], 53),
        ("bath_short", [4, 9, 52], 75),
        ("bath_long", [4, 9, 52], 183),
    ],
)
def test_reset_retains_child_posts_and_rounds_dependent_placement(
    kind, expected_waves, end, origin
):
    cavity = _pulse_cfg()
    if kind == "pulse":
        reset = PulseResetCfg(pulse_cfg=cavity).build("reset")
    elif kind == "two":
        reset = TwoPulseResetCfg(
            pulse1_cfg=cavity,
            pulse2_cfg=_pulse_cfg(1, length=0.051, pre=0.02, post=0.05),
        ).build("reset")
    else:
        reset = BathResetCfg(
            cavity_tone_cfg=cavity,
            qubit_tone_cfg=_pulse_cfg(
                1, length=0.401 if kind == "bath_long" else 0.051, pre=0.02, post=0.003
            ),
            pi2_cfg=_pulse_cfg(0, length=0.051),
        ).build("reset")

    def body(prog):
        if origin:
            prog.write_reg("duration", origin)
            prog.delay_timeline("duration")
        prog.emit(reset)

    result = shots(
        _program([reset], body, lambda prog: prog.append_macro(_AllocateDuration()))
    )
    # Cavity actual end+post=47.89504 =>48; its rule-a duration=50.
    # Two: second end+post=9+22.4+21.504=52.904 =>53 (rule a54).
    # Bath: pi2 starts48+4=52, end+post=52+22.4+.21504 =>75.
    # Qubit: short=9+22.4+1.29024 =>33; long=9+172.48+1.29024 =>183.
    # Rule a Bath ignores the qubit tail:50+29=79, pi2 starts54.
    assert [waves(shot) for shot in result] == [
        [origin + cycle for cycle in expected_waves]
    ] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        origin + end
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        origin + end + 9
    ] * 2
