"""Actual waveform/ADC ends govern module emission at the public compiler seam."""

from collections.abc import Callable, Sequence
from fractions import Fraction
from functools import lru_cache
from math import ceil

import numpy as np
import pytest
from qick.asm_v2 import Macro, QickParam, QickSpan, QickSweep1D
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import ComputedPulse, LoadValue, Module, Pulse, PulseCfg
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
    *, fabric_clock: float = 384,
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            if initialize is not None:
                initialize(self)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            body(self)

    profile = make_mock_soccfg(3, 1)
    for generator in profile["gens"]:
        generator["f_fabric"] = fabric_clock
    return Program(profile, reps=2, initial_delay=0, relax_delay=0.02)


@pytest.mark.parametrize("post, spacing", [(0.0005, 49), (0.001, 50)])
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
    # Start ceil(.01*430.08)=5. Adding 43.68 and .21504/.43008
    # before the operation's final ceil gives 49/50, respectively.
    result = shots(prog)
    assert [waves(shot) for shot in result] == [[5, spacing + 5]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        2 * spacing
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        2 * spacing + 9
    ] * 2
    assert prog.get_time_param("second", "t") * 430.08 == pytest.approx(spacing + 5)
    prog.compile()
    assert shots(prog) == result


@pytest.mark.parametrize("origin", [0, 40])
@pytest.mark.parametrize("swept_trigger", [False, True])
@pytest.mark.parametrize(
    "offset, post, end", [(0.0013, 0.0005, 20), (0.0011, 0.0012, 19)]
)
def test_readout_end_uses_quantized_trigger_before_adding_adc_window_and_post(
    origin, swept_trigger, offset, post, end
):
    trigger = (
        QickSweep1D("scan", offset, offset + 2 / 430.08) if swept_trigger else offset
    )
    readout = PulseReadout(
        "read",
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=10 / 384),
            pre_delay=0,
            post_delay=post,
            ro_ch=0,
            ro_length=13 / 307.2,
            trig_offset=trigger,
        ),
    )

    def initialize(prog):
        prog.append_macro(_AllocateDuration())
        if swept_trigger:
            prog.add_loop("scan", 3)

    def body(prog):
        if origin:
            prog.write_reg("duration", origin)
            prog.delay_timeline("duration")
        prog.emit(readout)

    result = shots(_program([readout], body, initialize))
    # Scalar triggers ceil both .559104 and .473088 to 1. Author
    # sweeps start at native round(.559104/.473088), then step by one.
    # ADC length is 18.2 cycles. Add post before final ceil.
    steps = [0, 1, 2] if swept_trigger else [0]
    trigger_start = 1 if not swept_trigger or offset == 0.0013 else 0
    if not swept_trigger and offset == 0.0011:
        end = 20
    assert [
        [
            (event.cycle - shot.start_cycle, event.value)
            for event in shot.events
            if event.kind == "data"
        ]
        for shot in result
    ] == [
        [(origin + trigger_start + step, 1), (origin + trigger_start + step + 10, 0)]
        for step in steps
    ] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        origin + end + step for step in steps
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        origin + end + step + 9 for step in steps
    ] * 2


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
    # Integer origin40 + ceil(start5 + tail43.68/43.4 + post)=89.
    # Native post sweep starts at zero and adds 0/2/4 cycles.
    ends = [89, 91, 93] if swept_post else [89]
    assert [waves(shot) for shot in result] == (
        [[40, 45]] if kind == "readout" else [[45]]
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
        ] == [[(45, 1), (55, 0)]] * (2 * len(ends))


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
    # Start ceils to5. Scalar post is .21504; native swept post starts0.
    # Both produce ceil(5+43.68+post)=49 at the first point.
    ends = [49, 79, 109] if swept_post else [49, 77, 105]
    result = shots(prog)
    assert prog.compile_datamem() is None
    assert [waves(shot) for shot in result] == [[5, end + 5] for end in ends] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        2 * end for end in ends
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        2 * end + 9 for end in ends
    ] * 2


@pytest.mark.parametrize("post_ticks, end", [(5e-11, 28), (2e-10, 29)])
def test_scalar_static_cross_grid_ceil_applies_noise_tolerance(post_ticks, end):
    pulse = Pulse(
        "precise", _pulse_cfg(length=25 / 384, pre=0, post=post_ticks / 430.08)
    )
    prog = _program([pulse], lambda p: p.emit(pulse))
    # Exact 28-tick playback plus post: only the tail outside tolerance adds a tick.
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [end] * 2


@pytest.mark.parametrize("post_ticks, end", [(5e-11, 14), (2e-10, 15)])
def test_scalar_adc_aggregate_ceil_applies_noise_tolerance(post_ticks, end):
    readout = PulseReadout(
        "precise_adc",
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=3 / 384),
            pre_delay=0,
            post_delay=post_ticks / 430.08,
            ro_ch=0,
            ro_length=10 / 307.2,
            trig_offset=0,
        ),
    )
    prog = _program([readout], lambda p: p.emit(readout))
    # Ten ADC samples are exactly 14 ticks; apply tolerance after adding post.
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [end] * 2


def test_runtime_cross_grid_length_sweep_ceil_without_implicit_time_table():
    pulse = Pulse(
        "cross_grid",
        _pulse_cfg(length=QickSweep1D("scan", 39 / 384, 115 / 384), pre=0, post=0),
    )
    tail = Pulse("tail", _pulse_cfg(1, length=10 / 384, pre=0, post=0))

    def body(prog):
        prog.emit(pulse)
        prog.emit(tail)

    prog = _program([pulse, tail], body, lambda p: p.add_loop("scan", 3))
    # W=287: ceil([39,77,115]*287/256) = [44,87,129].
    assert [waves(shot) for shot in shots(prog)] == [[0, 44], [0, 87], [0, 129]] * 2
    assert prog.compile_datamem() is None
    assert not any(i["CMD"] in {"ARITH", "DIV"} for i in prog.prog_list)


@lru_cache(maxsize=None)
def _signed_power_cost(value: int) -> int:
    if value <= 1:
        return value
    if value % 2 == 0:
        return _signed_power_cost(value // 2)
    return 1 + min(_signed_power_cost(value // 2), _signed_power_cost(value // 2 + 1))


@pytest.mark.parametrize("clock, weight", [(384, 287), (307.2, 359), (38.4, 2872), (430.08, 256)])
def test_runtime_conversion_uses_minimum_csd_upper_weight_with_500us_budget(clock, weight):
    pulse = Pulse("drive", _pulse_cfg(length=QickSweep1D("scan", 256 / clock, 258 / clock), pre=0, post=0))
    tail = Pulse("tail", _pulse_cfg(1, length=3 / clock, pre=0, post=0))

    def body(prog):
        prog.emit(pulse)
        prog.emit(tail)

    prog = _program([pulse, tail], body, lambda p: p.add_loop("scan", 3), fabric_clock=clock)
    starts = [waves(shot)[1] for shot in shots(prog)]
    assert starts == [(n * weight + 255) // 256 for n in (256, 257, 258)] * 2
    observed = starts[0]  # At n=256 the public end reveals the multiplier exactly.
    ratio = Fraction("430.08") / Fraction(str(clock))
    counts = np.arange(int(500 * clock) + 1, dtype=np.int64)
    physical_numerator = counts * ratio.numerator
    scheduled = (counts * observed + 255) // 256
    error_numerator = scheduled * ratio.denominator - physical_numerator
    assert np.all(error_numerator >= 0)
    assert np.all(error_numerator <= 430.08 * ratio.denominator)
    admissible = []
    for candidate in range(ceil(256 * ratio), int(256 * (ratio + Fraction("430.08") / int(500 * clock))) + 2):
        error = ((counts * candidate + 255) // 256) * ratio.denominator - physical_numerator
        if np.all(error >= 0) and np.all(error <= 430.08 * ratio.denominator):
            admissible.append(candidate)
    assert observed == min(admissible, key=lambda w: (_signed_power_cost(w), w))
    shifts = [op for inst in prog.prog_list if isinstance(op := inst.get("OP"), str)
              and any(shift in op for shift in ("SL", "SR", "ASR"))]
    assert all(int(op.split("#")[-1]) <= 15 for op in shifts)
    if clock == 430.08:
        assert shifts == []


def test_runtime_cross_grid_product_overflow_fails_at_build():
    # Native waveform words fit 16 bits, and the final end fits signed32.
    # Only the required fixed-point product overflows, before ASR8.
    pulse = Pulse("large", _pulse_cfg(length=QickSweep1D("scan", 25000, 25002), pre=0, post=0))
    with pytest.raises(ValueError, match="overflow signed 32-bit"):
        _program([pulse], lambda p: p.emit(pulse), lambda p: p.add_loop("scan", 3), fabric_clock=1)


def test_selected_waveform_with_native_length_sweep_tracks_selected_end():
    selector = LoadValue("selector", [0, 1], "choice", "selected")
    playback = ComputedPulse(
        "chosen", val_reg="selected",
        pulses=[_pulse_cfg(length=QickSweep1D("scan", n / 384, (n + 76) / 384), pre=0, post=0)
                for n in (39, 40)],
    )
    tail = Pulse("tail", _pulse_cfg(1, length=10 / 384, pre=0, post=0))

    def initialize(prog):
        prog.add_loop("choice", 2)
        prog.add_loop("scan", 3)

    def body(prog):
        prog.emit(selector)
        prog.emit(playback)
        prog.emit(tail)

    prog = _program([selector, playback, tail], body, initialize)
    # Candidate-specific fabric lengths use W=287, never longest-candidate padding.
    assert [waves(shot) for shot in shots(prog)] == [
        [0, end] for end in [44, 87, 129, 45, 88, 131]
    ] * 2
    # Two selector words and two author-declared waveform addresses, no time table.
    data = prog.compile_datamem()
    assert data is not None and len(data) == 4


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
    # W=287 gives ceil(n*287/256)=44,87,129, not affine44,87,130.
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
    assert prog.compile_datamem() is None
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
    # W=287 gives tails43.72265625/86.32421875/128.92578125.
    # Scalar post .086016 crosses 129; native swept post truncates to zero.
    last = 129 if swept_post else 130
    ends = [last, 87, 44] if descending else [44, 87, last]
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
    # Each native sweep point executes four k=8 local bodies, not four
    # advances of a shared sequential dmem cursor.
    ends = [44, 45, 46, 87, 88, 89, 129, 131, 132]
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
@pytest.mark.parametrize("length, end", [(0.051, 49), (0.201, 92)])
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
    # Start5 plus ADC43.4 or generator86.24 dominates. Scalar post .21504
    # or native swept post0 gives final ceil49/92, with swept steps2.
    assert [waves(shot) for shot in result] == [
        [0, 5, point_end] for point_end in ends
    ] * 2
    assert [
        [
            (event.cycle - shot.start_cycle, event.value)
            for event in shot.events
            if event.kind == "data"
        ]
        for shot in result
    ] == [[(5, 1), (15, 0)]] * (2 * len(ends))
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


def test_nonaffine_boundary_does_not_certify_a_negative_same_channel_offset():
    pulse = Pulse(
        "drive",
        _pulse_cfg(length=QickSweep1D("scan", 25 / 384, 75 / 384), pre=0, post=0),
        pulse_id="drive",
    )

    def body(prog):
        prog.emit(pulse)
        prog.pulse(0, "drive", t=-1 / 430.08, timeline="default")

    # At the first point,28-1 starts before actual playback ends at28.
    with pytest.raises(ValueError, match="channel 0 pulse overlap"):
        _program([pulse], body, lambda prog: prog.add_loop("scan", 3))


def test_sweep_end_requires_a_discovered_native_axis_before_using_its_bounds():
    with pytest.raises(ValueError, match="missing.*native sweep axis"):
        _program(
            [], lambda prog: prog.delay_timeline(QickSweep1D("missing", 0.101, 0.301))
        )


def test_native_sweep_delay_preserves_author_declared_dmem_capacity():
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
    assert len(memory) == 2045
    # Native start rounds43.68 to44; native step truncates42.56 to42.
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        44,
        86,
        128,
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [53, 95, 137] * 2


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
    # Cavity k=8 tails plus .086016 =>44/87/130. Static qubit stays87.
    # Pi2 starts at the cavity boundary and adds ceil(22.4+.086016)=23.
    cycles = [[0, 0]] * 3 if kind == "two" else [[0, 0, end] for end in [44, 87, 130]]
    ends = [87, 87, 130] if kind == "two" else [87, 110, 153]
    assert [waves(shot) for shot in result] == cycles * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == ends * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        end + 9 for end in ends
    ] * 2


@pytest.mark.parametrize("origin", [0, 40])
@pytest.mark.parametrize(
    "kind, expected_waves, end",
    [
        ("pulse", [5], 49),
        ("two", [5, 9], 53),
        ("bath_short", [5, 9, 54], 77),
        ("bath_long", [5, 9, 54], 183),
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
    # Cavity ceil(5+43.68+.21504)=49. The second child closes53.
    # Bath pi2 starts49+5=54 and closesceil(54+22.4+.21504)=77.
    # The longer qubit still dominates atceil(9+172.48+1.29024)=183.
    assert [waves(shot) for shot in result] == [
        [origin + cycle for cycle in expected_waves]
    ] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [
        origin + end
    ] * 2
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [
        origin + end + 9
    ] * 2
