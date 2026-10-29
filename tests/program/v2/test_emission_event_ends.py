"""Actual waveform/ADC ends govern module emission at the public compiler seam."""

from collections.abc import Callable, Sequence

import pytest
from qick.asm_v2 import Macro, QickParam, QickSweep1D
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import Module, Pulse, PulseCfg
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
    assert [waves(shot) for shot in result] == [[4, end + 4] for end in ends] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        2 * end for end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        2 * end + 9 for end in ends
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
